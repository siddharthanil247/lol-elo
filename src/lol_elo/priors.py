"""Fit and apply lower-league priors for Tier 1 rookie ratings."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lol_elo.leagues import filter_tier1_games, load_leagues
from lol_elo.player_elo import ROLES, run_player_elo
from lol_elo.team_elo import DEFAULT_MODEL_CONFIG_PATH, load_model_config

DEFAULT_FITTED_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "priors_fitted.yaml"
)


def build_lower_league_history(
    games: pd.DataFrame,
    players: pd.DataFrame,
    config: dict,
    leagues_config: dict | None = None,
) -> pd.DataFrame:
    """Run independent Elo pools for non-Tier-1 leagues and retain snapshots."""
    leagues_config = leagues_config or load_leagues()
    tier1 = filter_tier1_games(games)
    tier1_ids = set(tier1["gameid"].astype(str))
    lower_games = games.loc[~games["gameid"].astype(str).isin(tier1_ids)]
    lower_players = players.loc[
        players["gameid"].astype(str).isin(lower_games["gameid"].astype(str))
    ]
    lower_config = deepcopy(config)
    lower_config.update(
        region_strength_enabled=False,
        offseason_regression_enabled=False,
        use_rookie_priors=False,
        continuity_bonus=0,
    )
    player_rows = lower_players.loc[
        lower_players["position"].astype("string").str.lower().ne("team")
    ].copy()
    player_rows["_side"] = player_rows["side"].astype("string").str.lower()
    player_rows["_role"] = player_rows["position"].astype("string").str.lower()
    roster_stats = player_rows.groupby(["gameid", "_side"], sort=False).agg(
        row_count=("player_key", "size"),
        player_count=("player_key", "nunique"),
        role_count=("_role", "nunique"),
        role_set=("_role", lambda values: frozenset(values)),
    )
    valid_sides = roster_stats.loc[
        roster_stats["row_count"].eq(5)
        & roster_stats["player_count"].eq(5)
        & roster_stats["role_count"].eq(5)
        & roster_stats["role_set"].map(lambda roles: roles == frozenset(ROLES))
    ].reset_index()
    valid_games = valid_sides.groupby("gameid", sort=False)["_side"].nunique()
    valid_ids = set(valid_games.loc[valid_games.eq(2)].index.astype(str))
    has_teams = (games["blue_team_id"].notna() | games["blue_team"].notna()) & (
        games["red_team_id"].notna() | games["red_team"].notna()
    )
    has_winner = games["winner"].notna()
    valid_ids &= set(games.loc[has_teams & has_winner, "gameid"].astype(str))
    lower_games = lower_games.loc[lower_games["gameid"].astype(str).isin(valid_ids)]
    lower_players = lower_players.loc[
        lower_players["gameid"].astype(str).isin(valid_ids)
    ]
    snapshots = []
    for league, league_games in lower_games.groupby("league", sort=True):
        ids = set(league_games["gameid"].astype(str))
        league_players = lower_players.loc[
            lower_players["gameid"].astype(str).isin(ids)
        ]
        result = run_player_elo(
            league_games,
            league_players,
            lower_config,
            record_history=True,
            leagues_config=leagues_config,
        )
        if not result.rating_history.empty:
            snapshots.append(result.rating_history)
    if not snapshots:
        return pd.DataFrame(
            columns=[
                "player_key",
                "date",
                "gameid",
                "league",
                "year",
                "rating",
                "league_mean",
                "games_played",
            ]
        )
    history = pd.concat(snapshots, ignore_index=True)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    return history.sort_values(
        ["date", "gameid", "player_key"], kind="stable"
    ).reset_index(drop=True)


def prior_for_player(
    player_key: str,
    debut_date: object,
    history: pd.DataFrame,
    fitted: dict,
) -> tuple[float, str | None]:
    """Return a rookie starting rating using the latest strictly prior snapshot."""
    base = float(fitted["tier1_rookie_base"])
    if history.empty:
        return base, None
    cutoff = pd.to_datetime(debut_date, errors="coerce")
    rows = history.loc[
        history["player_key"].astype(str).eq(str(player_key))
        & pd.to_datetime(history["date"], errors="coerce").lt(cutoff)
    ]
    if rows.empty:
        return base, None
    row = rows.sort_values(["date", "gameid"], kind="stable").iloc[-1]
    league = str(row["league"])
    shrinks = fitted.get("league_shrink", {})
    shrink = float(shrinks.get(league, fitted.get("global_shrink", 0.0)))
    return base + shrink * (float(row["rating"]) - float(row["league_mean"])), league


def build_initial_rookie_ratings(
    games: pd.DataFrame,
    players: pd.DataFrame,
    lower_history: pd.DataFrame,
    fitted: dict,
) -> dict[str, float]:
    """Create priors for each player's first Tier 1 appearance."""
    ordered = games[["gameid", "date"]].copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
    appearances = players.loc[
        players["position"].astype("string").str.lower().ne("team"),
        ["player_key", "gameid"],
    ].merge(ordered, on="gameid", how="inner")
    debuts = appearances.sort_values(["date", "gameid"], kind="stable").drop_duplicates(
        "player_key", keep="first"
    )
    histories = {
        str(key): group.sort_values(["date", "gameid"], kind="stable")
        for key, group in lower_history.groupby("player_key", sort=False)
    }
    return {
        str(row.player_key): prior_for_player(
            str(row.player_key),
            row.date,
            histories.get(str(row.player_key), lower_history.iloc[0:0]),
            fitted,
        )[0]
        for row in debuts.itertuples(index=False)
    }


def load_or_build_lower_history(
    games: pd.DataFrame,
    players: pd.DataFrame,
    config: dict,
    path: str | Path = "data/processed/lower_league_ratings.parquet",
) -> pd.DataFrame:
    target = Path(path)
    metadata_path = target.with_suffix(target.suffix + ".meta.json")
    input_paths = [
        Path("data/processed/games_unfiltered.parquet"),
        Path("data/processed/players_unfiltered.parquet"),
    ]
    fingerprint = {
        "inputs": [
            {
                "path": str(source),
                "mtime_ns": source.stat().st_mtime_ns,
                "size": source.stat().st_size,
            }
            for source in input_paths
            if source.exists()
        ],
        "elo": {
            key: config.get(key)
            for key in (
                "starting_rating",
                "k_base",
                "k_min_ratio",
                "k_decay",
                "blue_side_advantage",
                "role_weights",
            )
        },
    }
    if target.exists() and metadata_path.exists():
        try:
            cached_fingerprint = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached_fingerprint = None
        if cached_fingerprint == fingerprint:
            return pd.read_parquet(target)
    if target.exists() and not input_paths[0].exists() and not input_paths[1].exists():
        return pd.read_parquet(target)
    history = build_lower_league_history(games, players, config)
    target.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(target, index=False)
    metadata_path.write_text(json.dumps(fingerprint, sort_keys=True), encoding="utf-8")
    return history


def fit_rookie_priors(debuts: pd.DataFrame, cutoff_year: int = 2024) -> dict:
    """Fit a shared base and eligible per-league slopes by least squares."""
    required = {"debut_year", "league", "lower_delta", "rating_after_20"}
    missing = required - set(debuts.columns)
    if missing:
        raise ValueError(f"Debut data missing columns: {', '.join(sorted(missing))}")
    fit = debuts.loc[
        pd.to_numeric(debuts["debut_year"], errors="coerce").lt(cutoff_year)
        & pd.to_numeric(debuts["rating_after_20"], errors="coerce").notna()
    ].copy()
    fit["lower_delta"] = pd.to_numeric(fit["lower_delta"], errors="coerce").fillna(0.0)
    fit["league"] = fit["league"].fillna("").astype(str)
    if fit.empty:
        return {
            "tier1_rookie_base": 1500.0,
            "global_shrink": 0.0,
            "league_shrink": {},
            "fit_cutoff_year": cutoff_year,
            "fit_debuts": 0,
            "league_debuts": {},
        }
    counts = fit.groupby("league").size().to_dict()
    eligible = sorted(
        league for league, count in counts.items() if league and count >= 30
    )
    target = fit["rating_after_20"].to_numpy(dtype=float)
    x = fit["lower_delta"].to_numpy(dtype=float)
    matrix = np.column_stack([np.ones(len(fit)), x])
    global_coef = np.linalg.lstsq(matrix, target, rcond=None)[0]
    global_shrink = float(global_coef[1])
    if eligible:
        columns = [np.ones(len(fit))]
        for league in eligible:
            columns.append(x * fit["league"].eq(league).to_numpy(dtype=float))
        coefficients = np.linalg.lstsq(np.column_stack(columns), target, rcond=None)[0]
        base = float(coefficients[0])
        league_shrink = {
            league: float(value)
            for league, value in zip(eligible, coefficients[1:], strict=True)
        }
    else:
        base = float(global_coef[0])
        league_shrink = {}
    return {
        "tier1_rookie_base": base,
        "global_shrink": global_shrink,
        "league_shrink": league_shrink,
        "fit_cutoff_year": cutoff_year,
        "fit_debuts": int(len(fit)),
        "league_debuts": {str(key): int(value) for key, value in counts.items()},
    }


def build_debut_training_rows(
    tier1_players: pd.DataFrame,
    flat_history: pd.DataFrame,
    lower_history: pd.DataFrame,
    cutoff_year: int = 2024,
) -> pd.DataFrame:
    """Join first Tier 1 debuts to their 20-game flat-Elo outcomes and priors."""
    first = tier1_players.copy()
    first["date"] = pd.to_datetime(first["date"], errors="coerce")
    debut = (
        first.sort_values(["date", "gameid"], kind="stable")
        .drop_duplicates("player_key", keep="first")
        .loc[:, ["player_key", "date", "league", "year"]]
        .rename(
            columns={
                "date": "debut_date",
                "league": "debut_league",
                "year": "debut_year",
            }
        )
    )
    history20 = flat_history.loc[flat_history["games_played"].eq(20)].copy()
    history20["date"] = pd.to_datetime(history20["date"], errors="coerce")
    history20 = history20.sort_values(
        ["date", "gameid"], kind="stable"
    ).drop_duplicates("player_key", keep="first")
    history20 = history20[["player_key", "date", "year", "rating"]].rename(
        columns={
            "date": "game20_date",
            "year": "game20_year",
            "rating": "rating_after_20",
        }
    )
    rows = debut.merge(history20, on="player_key", how="inner", validate="one_to_one")
    rows["debut_year"] = pd.to_numeric(rows["debut_year"], errors="coerce")
    rows["game20_year"] = pd.to_numeric(rows["game20_year"], errors="coerce")
    rows = rows.loc[
        rows["debut_year"].lt(cutoff_year) & rows["game20_year"].lt(cutoff_year)
    ].copy()
    lower_groups = {
        str(key): group.sort_values(["date", "gameid"], kind="stable")
        for key, group in lower_history.groupby("player_key", sort=False)
    }
    prior_rows = []
    for row in rows.itertuples(index=False):
        snapshot = lower_groups.get(str(row.player_key), lower_history.iloc[0:0])
        snapshot = snapshot.loc[
            pd.to_datetime(snapshot["date"], errors="coerce").lt(row.debut_date)
        ]
        if snapshot.empty:
            prior_rows.append({"lower_league": "", "lower_delta": 0.0})
        else:
            item = snapshot.sort_values(["date", "gameid"], kind="stable").iloc[-1]
            prior_rows.append(
                {
                    "lower_league": str(item["league"]),
                    "lower_delta": float(item["rating"] - item["league_mean"]),
                }
            )
    rows = rows.reset_index(drop=True).join(pd.DataFrame(prior_rows))
    return rows.rename(columns={"lower_league": "league"})[
        [
            "player_key",
            "debut_date",
            "debut_year",
            "league",
            "lower_delta",
            "rating_after_20",
            "game20_date",
        ]
    ]


def save_fitted_priors(fitted: dict, path: str | Path = DEFAULT_FITTED_PATH) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(fitted, sort_keys=False), encoding="utf-8")


def load_fitted_priors(path: str | Path = DEFAULT_FITTED_PATH) -> dict:
    with Path(path).open(encoding="utf-8") as fitted_file:
        result = yaml.safe_load(fitted_file) or {}
    if "tier1_rookie_base" not in result:
        raise ValueError(f"Fitted prior file {path} has no tier1_rookie_base")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Tier 1 rookie priors")
    parser.add_argument("--games", default="data/processed/games.parquet")
    parser.add_argument("--players", default="data/processed/players.parquet")
    parser.add_argument(
        "--games-unfiltered", default="data/processed/games_unfiltered.parquet"
    )
    parser.add_argument(
        "--players-unfiltered", default="data/processed/players_unfiltered.parquet"
    )
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG_PATH))
    parser.add_argument("--output", default=str(DEFAULT_FITTED_PATH))
    parser.add_argument("--notes", default="notes/ablations.md")
    parser.add_argument("--start-year", type=int, default=2019)
    args = parser.parse_args()
    from lol_elo.prior_evaluation import evaluate_rookie_priors

    games = pd.read_parquet(args.games)
    players = pd.read_parquet(args.players)
    all_games = pd.read_parquet(args.games_unfiltered)
    all_players = pd.read_parquet(args.players_unfiltered)
    config = load_model_config(args.model_config)
    lower = load_or_build_lower_history(all_games, all_players, config)
    flat = run_player_elo(
        games, players, {**config, "use_rookie_priors": False}, record_history=True
    )
    training = build_debut_training_rows(players, flat.rating_history, lower)
    fitted = fit_rookie_priors(training)
    save_fitted_priors(fitted, args.output)
    results = evaluate_rookie_priors(
        games, players, lower, config, fitted, args.start_year
    )
    print(fitted)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    notes_path = Path(args.notes)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\n## Rookie-prior ablation\n\n",
        "Scored games use walk-forward predictions; continuity bonus was held at its "
        "configured value. Fitted values use Tier 1 debuts and 20-game outcomes "
        "strictly before 2024.\n\n",
    ]
    columns = list(results.columns)
    lines.append("| " + " | ".join(columns) + " |\n")
    lines.append("| " + " | ".join("---" for _ in columns) + " |\n")
    for row in results.itertuples(index=False, name=None):
        values = [
            f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |\n")
    with notes_path.open("a", encoding="utf-8") as notes_file:
        notes_file.writelines(lines)
    print(f"Wrote evaluation to {notes_path}")


if __name__ == "__main__":
    main()
