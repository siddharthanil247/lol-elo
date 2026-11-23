"""Random search for player Elo parameters with a 2024+ held-out evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lol_elo.evaluate import evaluate_predictions
from lol_elo.leagues import load_leagues
from lol_elo.player_elo import ROLES, run_player_elo
from lol_elo.priors import (
    build_debut_training_rows,
    build_initial_rookie_ratings,
    fit_rookie_priors,
    load_fitted_priors,
    load_or_build_lower_history,
    save_fitted_priors,
)
from lol_elo.project import (
    _last_org_lineups,
    _org_key,
    _project_name_catalog,
    _season_rosters_from_games,
    _team_actual_rates,
    project_team_ratings,
    run_backtest,
    season_start,
)
from lol_elo.team_elo import DEFAULT_MODEL_CONFIG_PATH, load_model_config

MODEL_TUNED_PATH = Path("config/model_tuned.yaml")
PRIORS_TUNED_PATH = Path("config/priors_tuned.yaml")


def _log_loss(predictions: pd.DataFrame) -> float:
    p = np.clip(
        predictions["predicted_blue_win_prob"].to_numpy(float), 1e-15, 1 - 1e-15
    )
    y = predictions["actual_blue_win"].to_numpy(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _random_candidate(
    rng: np.random.Generator,
    current: dict,
    trial: int,
) -> dict:
    """Draw normalized role weights and parameters; trial zero mirrors current."""
    candidate = deepcopy(current)
    raw_weights = np.array(
        [float(current.get("role_weights", {}).get(role, 1.0)) for role in ROLES]
    )
    total = float(raw_weights.sum()) or 5.0
    if trial == 0:
        weights = raw_weights / total
        candidate["k_base"] = float(current.get("k_base", 32)) * total
        candidate["k_region"] = float(current.get("k_region", 0.1)) / total
        candidate["role_weights"] = {
            role: float(weight) for role, weight in zip(ROLES, weights, strict=True)
        }
        candidate["regress_frac"] = 0.0
        candidate["new_season_k_mult"] = 1.0
        candidate["offseason_regression_enabled"] = False
        return candidate

    candidate["k_base"] = float(np.exp(rng.uniform(np.log(20), np.log(200))))
    candidate["k_min_ratio"] = float(rng.uniform(0.05, 0.75))
    candidate["k_decay"] = float(np.exp(rng.uniform(np.log(4), np.log(100))))
    candidate["blue_side_advantage"] = float(rng.uniform(0, 60))
    candidate["continuity_bonus"] = float(rng.uniform(0, 200))
    candidate["k_region"] = float(rng.uniform(0, 0.25))
    weights = rng.dirichlet(np.ones(5))
    candidate["role_weights"] = {
        role: float(weight) for role, weight in zip(ROLES, weights, strict=True)
    }
    candidate["regress_frac"] = float(rng.uniform(0, 0.15))
    candidate["new_season_k_mult"] = float(rng.uniform(1.0, 1.5))
    candidate["offseason_regression_enabled"] = candidate["regress_frac"] > 0
    candidate["use_rookie_priors"] = bool(current.get("use_rookie_priors", True))
    return candidate


def _snapshot_from_run(
    result,
    games: pd.DataFrame,
    cutoff: pd.Timestamp,
    leagues_config: dict,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Recover exact player ratings and region offsets at a date from one run."""
    history = result.rating_history.copy()
    if not history.empty:
        history["date"] = pd.to_datetime(history["date"], errors="coerce")
        history = history.loc[history["date"].lt(cutoff)]
        history = history.sort_values(["date", "gameid"], kind="stable")
        history = history.drop_duplicates("player_key", keep="last")
    ratings = (
        history[["player_key", "rating", "games_played", "region"]].copy()
        if not history.empty
        else pd.DataFrame(columns=["player_key", "rating", "games_played", "region"])
    )

    offsets = {region: 0.0 for region in leagues_config.get("regions", {})}
    predictions = result.predictions.copy()
    if predictions.empty:
        return ratings, offsets
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce")
    previous = predictions.loc[predictions["date"].lt(cutoff)]
    if previous.empty:
        return ratings, offsets
    blue = previous[["blue_region", "region_rating_change"]].rename(
        columns={"blue_region": "region", "region_rating_change": "change"}
    )
    red = previous[["red_region", "region_rating_change"]].rename(
        columns={"red_region": "region", "region_rating_change": "change"}
    )
    red["change"] *= -1
    changes = pd.concat([blue, red], ignore_index=True).dropna(subset=["region"])
    totals = changes.groupby("region")["change"].sum()
    for region, change in totals.items():
        offsets[str(region)] = float(change)
    return ratings, offsets


def _projected_game_rates(projected: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    advantage = float(config.get("blue_side_advantage", 0))
    for league, frame in projected.groupby("league", sort=False):
        frame = frame.reset_index(drop=True)
        ratings = frame["projected_rating"].to_numpy(float)
        diff = ratings[:, None] - ratings[None, :]
        blue_p = 1 / (1 + np.power(10, -(diff + advantage) / 400))
        red_p = 1 / (1 + np.power(10, -(diff - advantage) / 400))
        balanced = (blue_p + red_p) / 2
        rates = (balanced.sum(axis=1) - 0.5) / max(len(frame) - 1, 1)
        rows.extend(
            {"teamkey": _org_key(row.teamid, row.team), "projected_rate": float(rate)}
            for row, rate in zip(frame.itertuples(index=False), rates, strict=True)
        )
    return pd.DataFrame(rows)


def _candidate_backtest_mae(
    model_result,
    games: pd.DataFrame,
    players: pd.DataFrame,
    lower_history: pd.DataFrame,
    fitted: dict,
    model_config: dict,
    leagues_config: dict,
    all_players: pd.DataFrame,
    seasons: range = range(2020, 2024),
) -> float:
    domestic = {
        entry if isinstance(entry, str) else entry["name"]
        for entry in leagues_config["tier1_domestic"]
    }
    catalog = _project_name_catalog(players, all_players)
    errors = []
    for season in seasons:
        cutoff = season_start(games, season)
        prior_games = games.loc[
            pd.to_datetime(games["date"], errors="coerce").lt(cutoff)
        ]
        prior_ids = set(prior_games["gameid"].astype(str))
        prior_players = players.loc[players["gameid"].astype(str).isin(prior_ids)]
        frozen_ratings, offsets = _snapshot_from_run(
            model_result, games, cutoff, leagues_config
        )
        rosters = _season_rosters_from_games(games, players, season, domestic)
        projections, _ = project_team_ratings(
            rosters,
            frozen_ratings,
            offsets,
            _last_org_lineups(prior_games, prior_players, cutoff),
            model_config,
            leagues_config,
            lower_history,
            fitted,
            cutoff,
            catalog,
        )
        if projections.empty:
            continue
        projections["teamkey"] = projections.apply(
            lambda row: _org_key(row["teamid"], row["team"]), axis=1
        )
        rates = _projected_game_rates(projections, model_config)
        for league in projections["league"].dropna().unique():
            actual = _team_actual_rates(games, season, str(league))
            joined = rates.merge(actual[["teamkey", "actual_win_rate"]], on="teamkey")
            errors.extend(
                (joined["projected_rate"] - joined["actual_win_rate"]).abs().tolist()
            )
    return float(np.mean(errors)) if errors else float("nan")


def _fit_priors_as_of(
    games: pd.DataFrame,
    players: pd.DataFrame,
    lower_history: pd.DataFrame,
    config: dict,
    cutoff: pd.Timestamp,
    cutoff_year: int,
) -> dict:
    dates = pd.to_datetime(games["date"], errors="coerce")
    pre_games = games.loc[dates.lt(cutoff)]
    pre_ids = set(pre_games["gameid"].astype(str))
    pre_players = players.loc[players["gameid"].astype(str).isin(pre_ids)]
    flat = run_player_elo(
        pre_games,
        pre_players,
        {**config, "use_rookie_priors": False},
        record_history=True,
    )
    training = build_debut_training_rows(
        pre_players, flat.rating_history, lower_history
    )
    return fit_rookie_priors(training, cutoff_year=cutoff_year)


def _evaluate_candidate(
    candidate: dict,
    games: pd.DataFrame,
    players: pd.DataFrame,
    all_players: pd.DataFrame,
    lower_history: pd.DataFrame,
    objective_weights: dict,
) -> tuple[dict, dict]:
    holdout_start = season_start(games, 2024)
    train_games = games.loc[
        pd.to_datetime(games["date"], errors="coerce").lt(holdout_start)
    ].copy()
    train_ids = set(train_games["gameid"].astype(str))
    train_players = players.loc[players["gameid"].astype(str).isin(train_ids)].copy()
    tuning_start = season_start(games, 2019)
    training_priors = _fit_priors_as_of(
        games, players, lower_history, candidate, tuning_start, cutoff_year=2019
    )
    initial = build_initial_rookie_ratings(
        train_games, train_players, lower_history, training_priors
    )
    result = run_player_elo(
        train_games,
        train_players,
        candidate,
        initial_ratings=initial,
        record_history=True,
    )
    years = pd.to_numeric(result.predictions["year"], errors="coerce")
    scored = result.predictions.loc[years.between(2019, 2023)].copy()
    game_loss = _log_loss(scored)
    backtest_mae = _candidate_backtest_mae(
        result,
        games,
        players,
        lower_history,
        training_priors,
        candidate,
        load_leagues(),
        all_players,
    )
    w_game = float(objective_weights.get("game_log_loss", 0.7))
    w_mae = float(objective_weights.get("backtest_mae", 0.3))
    objective = w_game * game_loss + w_mae * backtest_mae
    metrics = {
        "game_log_loss": game_loss,
        "backtest_mae": backtest_mae,
        "objective": objective,
        "scored_games": len(scored),
    }
    return metrics, candidate


def _holdout_metrics(
    games: pd.DataFrame,
    players: pd.DataFrame,
    all_games: pd.DataFrame,
    all_players: pd.DataFrame,
    config: dict,
    fitted: dict,
    lower_history: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    initial = build_initial_rookie_ratings(games, players, lower_history, fitted)
    result = run_player_elo(
        games, players, config, start_year=2024, initial_ratings=initial
    )
    evaluation = evaluate_predictions(result.predictions, config, start_year=2024)
    overall = evaluation.metrics.loc[evaluation.metrics["scope"].eq("overall")].iloc[0]
    series = evaluation.series_metrics.loc[
        evaluation.series_metrics["scope"].eq("overall")
    ].iloc[0]
    return {
        "game_log_loss": float(overall["elo_log_loss"]),
        "series_log_loss": float(series["series_log_loss"]),
    }, result.role_ratings.merge(
        result.ratings[["player_key", "region"]], on="player_key", how="left"
    ).assign(
        effective_rating=lambda frame: (
            frame["rating"] + frame["region"].map(result.region_offsets).fillna(0.0)
        ),
        _active=lambda frame: frame["player_key"].isin(_active_keys(players, games)),
    ).pipe(lambda frame: frame.loc[frame["_active"]]).sort_values(
        ["role", "effective_rating"], ascending=[True, False], kind="stable"
    ).groupby("role", sort=False).head(10)


def _active_keys(players: pd.DataFrame, games: pd.DataFrame) -> set[str]:
    dates = pd.to_datetime(games["date"], errors="coerce")
    latest = dates.max()
    player_dates = pd.to_datetime(players["date"], errors="coerce")
    return set(
        players.loc[
            player_dates.between(latest - pd.Timedelta(days=365), latest), "player_key"
        ]
        .dropna()
        .astype(str)
    )


def _holdout_backtest(
    games: pd.DataFrame,
    players: pd.DataFrame,
    all_games: pd.DataFrame,
    all_players: pd.DataFrame,
    config: dict,
    fitted: dict,
    lower_history: pd.DataFrame,
) -> dict:
    table = run_backtest(
        games,
        players,
        all_games,
        all_players,
        config,
        load_leagues(),
        fitted,
        seasons=range(2024, 2027),
        lower_history=lower_history,
    )

    def get(cohort: str) -> dict:
        player = table.loc[
            table["cohort"].eq(cohort)
            & table["league"].eq("overall")
            & table["method"].eq("player projection")
        ].iloc[0]
        return {"spearman": float(player["spearman"]), "mae": float(player["mae"])}

    return {"all teams": get("all teams"), "3+ changes": get("3+ roster changes")}


def _save_yaml(data: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def to_builtin(value):
        if isinstance(value, dict):
            return {key: to_builtin(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_builtin(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    target.write_text(
        yaml.safe_dump(to_builtin(data), sort_keys=False), encoding="utf-8"
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [
            f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _yaml_builtin(value):
    """Convert nested NumPy values before writing selected settings as YAML."""
    if isinstance(value, dict):
        return {key: _yaml_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def random_search(
    games: pd.DataFrame,
    players: pd.DataFrame,
    all_games: pd.DataFrame,
    all_players: pd.DataFrame,
    config: dict,
    trials: int = 8,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    lower_history = load_or_build_lower_history(all_games, all_players, config)
    weights = config.get("tuning_objective_weights", {})
    trial_rows = []
    best_candidate = None
    best_objective = float("inf")
    for trial in range(trials):
        candidate = _random_candidate(rng, config, trial)
        metrics, candidate = _evaluate_candidate(
            candidate, games, players, all_players, lower_history, weights
        )
        trial_rows.append(
            {
                "trial": trial,
                **metrics,
                **{
                    "k_base": candidate["k_base"],
                    "k_min_ratio": candidate["k_min_ratio"],
                    "k_decay": candidate["k_decay"],
                    "blue_side_advantage": candidate["blue_side_advantage"],
                    "continuity_bonus": candidate["continuity_bonus"],
                    "k_region": candidate["k_region"],
                    "regress_frac": candidate["regress_frac"],
                    "new_season_k_mult": candidate["new_season_k_mult"],
                },
            }
        )
        print(
            f"trial {trial + 1}/{trials}: log_loss={metrics['game_log_loss']:.5f}, "
            f"backtest_mae={metrics['backtest_mae']:.5f}, "
            f"objective={metrics['objective']:.5f}",
            flush=True,
        )
        if metrics["objective"] < best_objective:
            best_objective = metrics["objective"]
            best_candidate = candidate
    assert best_candidate is not None
    return pd.DataFrame(trial_rows).sort_values("objective"), best_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune player Elo parameters")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--notes", default="notes/tuning.md")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least one")
    games = pd.read_parquet("data/processed/games.parquet")
    players = pd.read_parquet("data/processed/players.parquet")
    all_games = pd.read_parquet("data/processed/games_unfiltered.parquet")
    all_players = pd.read_parquet("data/processed/players_unfiltered.parquet")
    current = load_model_config()
    trials, best = random_search(
        games, players, all_games, all_players, current, args.trials, args.seed
    )
    _save_yaml(best, MODEL_TUNED_PATH)

    holdout_start = season_start(games, 2024)
    current_lower = load_or_build_lower_history(all_games, all_players, current)
    tuned_lower = load_or_build_lower_history(all_games, all_players, best)
    tuned_priors = _fit_priors_as_of(
        games,
        players,
        tuned_lower,
        best,
        holdout_start,
        cutoff_year=2024,
    )
    save_fitted_priors(tuned_priors, PRIORS_TUNED_PATH)
    current_holdout, current_top = _holdout_metrics(
        games,
        players,
        all_games,
        all_players,
        current,
        load_fitted_priors(),
        current_lower,
    )
    tuned_holdout, tuned_top = _holdout_metrics(
        games, players, all_games, all_players, best, tuned_priors, tuned_lower
    )
    current_backtest = _holdout_backtest(
        games,
        players,
        all_games,
        all_players,
        current,
        load_fitted_priors(),
        current_lower,
    )
    tuned_backtest = _holdout_backtest(
        games, players, all_games, all_players, best, tuned_priors, tuned_lower
    )
    weights = current.get("tuning_objective_weights", {})
    game_weight = float(weights.get("game_log_loss", 0.7))
    mae_weight = float(weights.get("backtest_mae", 0.3))
    current_score = (
        game_weight * current_holdout["game_log_loss"]
        + mae_weight * current_backtest["all teams"]["mae"]
    )
    tuned_score = (
        game_weight * tuned_holdout["game_log_loss"]
        + mae_weight * tuned_backtest["all teams"]["mae"]
    )
    tuned_wins = tuned_score < current_score
    if tuned_wins:
        _save_yaml(best, DEFAULT_MODEL_CONFIG_PATH)
        save_fitted_priors(tuned_priors)

    holdout_rows = []
    for label, game, backtest in (
        ("current", current_holdout, current_backtest),
        ("tuned", tuned_holdout, tuned_backtest),
    ):
        for cohort, values in backtest.items():
            holdout_rows.append(
                {
                    "configuration": label,
                    "cohort": cohort,
                    "game_log_loss": game["game_log_loss"],
                    "series_log_loss": game["series_log_loss"],
                    "backtest_spearman": values["spearman"],
                    "backtest_mae": values["mae"],
                    "holdout_objective": game_weight * game["game_log_loss"]
                    + mae_weight * backtest["all teams"]["mae"],
                }
            )
    holdout = pd.DataFrame(holdout_rows)
    print("\nHoldout comparison (2024 onward):")
    print(holdout.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nDefault config {'updated' if tuned_wins else 'left unchanged'}.")
    print("\nActive top 10 per role under tuned parameters:")
    for role in ROLES:
        top = tuned_top.loc[tuned_top["role"].eq(role)].head(10)
        print(f"\n{role}:")
        print(
            top[
                ["playername", "rating", "region", "effective_rating", "games_played"]
            ].to_string(index=False)
        )
    notes = Path(args.notes)
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text(
        "# Player Elo random search\n\n"
        f"Random seed {args.seed}; {args.trials} trials. The search objective uses "
        "2019–2023 walk-forward game log loss and 2020–2023 walk-forward "
        f"backtest MAE with weights {game_weight:.2f}/{mae_weight:.2f}. "
        "All prior fits and rating snapshots stop before each evaluation cutoff. "
        "2024 onward is used only for the final holdout comparison.\n\n"
        "## Trials\n\n"
        + _markdown_table(trials)
        + "\n\n## Holdout\n\n"
        + _markdown_table(holdout)
        + "\n\n## Decision\n\n"
        + (
            f"Tuned parameters win the weighted holdout objective ({tuned_score:.5f} "
            f"vs {current_score:.5f}); config/model.yaml and priors_fitted.yaml "
            "were updated.\n"
            if tuned_wins
            else f"Tuned parameters do not win the weighted holdout objective "
            f"({tuned_score:.5f} vs {current_score:.5f}); the default config "
            "remains unchanged. Tuned files are available for inspection.\n"
        )
        + "\n## Selected parameters\n\n```yaml\n"
        + yaml.safe_dump(_yaml_builtin(best), sort_keys=False)
        + "```\n",
        encoding="utf-8",
    )
    print(f"\nSaved best params to {MODEL_TUNED_PATH}; report to {notes}.")


if __name__ == "__main__":
    main()
