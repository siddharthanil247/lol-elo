"""Walk-forward player-level Elo ratings aggregated into team strength."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from lol_elo.evaluate import EvaluationResult, evaluate_predictions
from lol_elo.leagues import load_leagues
from lol_elo.team_elo import (
    DEFAULT_MODEL_CONFIG_PATH,
    _probability,
    _team_key,
    load_model_config,
    order_games,
)

ROLES = ("top", "jng", "mid", "bot", "sup")


class PlayerEloResult(NamedTuple):
    predictions: pd.DataFrame
    ratings: pd.DataFrame
    role_ratings: pd.DataFrame
    region_offsets: dict[str, float]
    region_offset_history: pd.DataFrame
    rating_history: pd.DataFrame


def _win_probability(
    blue_rating: float, red_rating: float, blue_advantage: float
) -> float:
    """Return blue's win probability from team ratings and side advantage."""
    return _probability(blue_rating + blue_advantage - red_rating)


def _convert_player_region_rating(
    rating: float,
    old_region: str,
    new_region: str,
    region_offsets: dict[str, float],
) -> float:
    """Convert a player rating while preserving player-plus-region strength."""
    return (
        rating
        + region_offsets.get(old_region, 0.0)
        - region_offsets.get(new_region, 0.0)
    )


def _rosters_by_game(
    players: pd.DataFrame,
) -> dict[str, dict[str, list[tuple[str, str]]]]:
    required = {"gameid", "side", "position", "player_key"}
    missing = required - set(players.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"Players data is missing columns: {missing_names}")

    player_rows = players.loc[
        players["position"].astype("string").str.lower().ne("team")
    ].copy()
    player_rows["_side"] = player_rows["side"].astype("string").str.strip().str.lower()
    player_rows["_position"] = (
        player_rows["position"].astype("string").str.strip().str.lower()
    )
    rosters: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for (gameid, side), side_rows in player_rows.groupby(
        ["gameid", "_side"], sort=False
    ):
        roster = list(
            zip(
                side_rows["player_key"].astype("string"),
                side_rows["_position"].astype("string"),
                strict=True,
            )
        )
        if side not in {"blue", "red"}:
            continue
        if len(roster) != 5 or {role for _, role in roster} != set(ROLES):
            raise ValueError(
                f"Game {gameid} {side} roster must contain exactly one player "
                "for each role"
            )
        if any(pd.isna(player_key) or not player_key for player_key, _ in roster):
            raise ValueError(
                f"Game {gameid} {side} roster has a missing player identity"
            )
        if len({player_key for player_key, _ in roster}) != 5:
            raise ValueError(
                f"Game {gameid} {side} roster has duplicate player identities"
            )
        rosters.setdefault(str(gameid), {})[str(side)] = [
            (str(player_key), str(role)) for player_key, role in roster
        ]
    return rosters


def _weighted_team_rating(
    roster: list[tuple[str, str]],
    player_ratings: dict[str, float],
    role_weights: dict[str, float],
    starting_rating: float,
) -> float:
    weights = [float(role_weights[role]) for _, role in roster]
    weight_total = sum(weights)
    if weight_total <= 0:
        raise ValueError("The five role weights must sum to a positive number")
    return (
        sum(
            player_ratings.get(player_key, starting_rating) * weight
            for (player_key, _), weight in zip(roster, weights, strict=True)
        )
        / weight_total
    )


def _recent_player_keys(players: pd.DataFrame, games: pd.DataFrame) -> set[str]:
    """Return identities active in the 365 days before the latest game date."""
    if games.empty or "date" not in games or "date" not in players:
        return set()
    latest_game = pd.to_datetime(games["date"], errors="coerce").max()
    if pd.isna(latest_game):
        return set()
    cutoff = latest_game - pd.Timedelta(days=365)
    player_dates = pd.to_datetime(players["date"], errors="coerce")
    active = player_dates.between(cutoff, latest_game, inclusive="both")
    return set(players.loc[active, "player_key"].dropna().astype(str))


def _team_key_series(team_ids: pd.Series, team_names: pd.Series) -> pd.Series:
    ids = team_ids.astype("string").str.strip()
    names = team_names.astype("string").str.strip().str.lower()
    has_id = ids.notna() & ids.ne("")
    has_name = names.notna() & names.ne("")
    return ("teamid:" + ids).where(has_id, ("teamname:" + names).where(has_name))


def run_player_elo(
    games: pd.DataFrame,
    players: pd.DataFrame,
    config: dict | None = None,
    start_year: int | None = None,
    leagues_config: dict | None = None,
    initial_ratings: dict[str, float] | None = None,
    record_history: bool = False,
) -> PlayerEloResult:
    """Predict games from participating-player ratings, then update those players."""
    config = config or load_model_config()
    required = {
        "gameid",
        "date",
        "game",
        "year",
        "blue_team",
        "red_team",
        "winner",
    }
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Games data is missing columns: {', '.join(sorted(missing))}")

    role_weights = {
        role: float(config.get("role_weights", {}).get(role, 1.0)) for role in ROLES
    }
    if any(weight < 0 for weight in role_weights.values()):
        raise ValueError("Role weights cannot be negative")
    if sum(role_weights.values()) <= 0:
        raise ValueError("The five role weights must sum to a positive number")
    k_base = float(config.get("k_base", 32))
    k_min_ratio = float(config.get("k_min_ratio", 0.25))
    k_decay = float(config.get("k_decay", 20))
    if k_base < 0 or not 0 <= k_min_ratio <= 1 or k_decay <= 0:
        raise ValueError("Invalid player Elo K configuration")
    starting_rating = float(config.get("starting_rating", 1500))
    blue_advantage = float(config.get("blue_side_advantage", 0))
    continuity_n = int(config.get("continuity_n", 5))
    if continuity_n < 0:
        raise ValueError("continuity_n cannot be negative")
    continuity_bonus = float(config.get("continuity_bonus", 0))
    region_enabled = bool(config.get("region_strength_enabled", False))
    k_region = float(config.get("k_region", 0.1))
    offseason_enabled = bool(config.get("offseason_regression_enabled", False))
    regress_frac = float(config.get("regress_frac", 0.2))
    new_season_m = int(config.get("new_season_m", 10))
    new_season_k_mult = float(config.get("new_season_k_mult", 1.5))
    if k_region < 0 or not 0 <= regress_frac <= 1 or new_season_m < 0:
        raise ValueError("Invalid regional or offseason configuration")
    if new_season_k_mult < 0:
        raise ValueError("new_season_k_mult cannot be negative")
    leagues_config = leagues_config or load_leagues()
    region_leagues = leagues_config.get("regions", {})
    league_regions = {
        league: region
        for region, league_names in region_leagues.items()
        for league in league_names
    }
    region_offsets = {region: 0.0 for region in region_leagues}

    ordered = order_games(games)
    ordered["year"] = pd.to_numeric(ordered["year"], errors="coerce")
    rosters = _rosters_by_game(players)
    if "playername" in players.columns:
        name_rows = players.dropna(subset=["player_key", "playername"]).drop_duplicates(
            "player_key", keep="last"
        )
        player_names = (
            name_rows.set_index("player_key")["playername"].astype(str).to_dict()
        )
    else:
        player_names = {}
    player_ratings: dict[str, float] = {}
    initial_ratings = initial_ratings or {}
    games_played: dict[str, int] = {}
    latest_roles: dict[str, str] = {}
    team_history: dict[str, deque[frozenset[str]]] = {}
    team_regions: dict[tuple[object, str], str] = {}
    player_regions: dict[str, str] = {}
    seen_seasons: dict[str, set[object]] = {}
    season_games_played: dict[tuple[str, object], int] = {}
    predictions = []
    rating_history = []
    offset_snapshots = []
    last_position_by_year = {}
    for position, season in enumerate(ordered["year"]):
        if pd.notna(season):
            last_position_by_year[season] = position

    for position, game in enumerate(ordered.itertuples(index=False)):
        values = game._asdict()
        gameid = str(values["gameid"])
        game_rosters = rosters.get(gameid)
        if game_rosters is None or not {"blue", "red"}.issubset(game_rosters):
            raise ValueError(
                f"Game {gameid} is missing a complete blue/red player roster"
            )
        blue_roster = game_rosters["blue"]
        red_roster = game_rosters["red"]
        blue_team_key = _team_key(values.get("blue_team_id"), values["blue_team"])
        red_team_key = _team_key(values.get("red_team_id"), values["red_team"])
        season = values["year"] if pd.notna(values["year"]) else None
        season_key = int(season) if season is not None else None
        league = str(values.get("league", ""))
        direct_region = league_regions.get(league)
        if direct_region is not None:
            blue_region = red_region = direct_region
        elif season_key is not None:
            blue_region = team_regions.get((season_key, blue_team_key))
            red_region = team_regions.get((season_key, red_team_key))
        else:
            blue_region = red_region = None
        winner_key = _team_key(values.get("winner_team_id"), values["winner"])
        if winner_key == blue_team_key:
            actual_blue_win = 1
        elif winner_key == red_team_key:
            actual_blue_win = 0
        else:
            raise ValueError(f"Winner does not match either team in game {gameid}")

        for roster, region in (
            (blue_roster, blue_region),
            (red_roster, red_region),
        ):
            for player_key, _ in roster:
                if player_key not in player_ratings:
                    player_ratings[player_key] = float(
                        initial_ratings.get(player_key, starting_rating)
                    )
                if offseason_enabled and season_key is not None:
                    player_seasons = seen_seasons.setdefault(player_key, set())
                    if (
                        season_key not in player_seasons
                        and games_played.get(player_key, 0) > 0
                    ):
                        old_rating = player_ratings.get(player_key, starting_rating)
                        player_ratings[player_key] = starting_rating + (
                            old_rating - starting_rating
                        ) * (1 - regress_frac)
                    player_seasons.add(season_key)
                if region_enabled and region is not None:
                    old_region = player_regions.get(player_key)
                    if old_region is not None and old_region != region:
                        player_ratings[player_key] = _convert_player_region_rating(
                            player_ratings.get(player_key, starting_rating),
                            old_region,
                            region,
                            region_offsets,
                        )
                    player_regions[player_key] = region

        blue_continuity = sum(
            any(
                player_key in previous_roster
                for previous_roster in team_history.get(blue_team_key, ())
            )
            for player_key, _ in blue_roster
        )
        red_continuity = sum(
            any(
                player_key in previous_roster
                for previous_roster in team_history.get(red_team_key, ())
            )
            for player_key, _ in red_roster
        )
        blue_pre = _weighted_team_rating(
            blue_roster, player_ratings, role_weights, starting_rating
        )
        red_pre = _weighted_team_rating(
            red_roster, player_ratings, role_weights, starting_rating
        )
        blue_pre += continuity_bonus * (blue_continuity / 5)
        red_pre += continuity_bonus * (red_continuity / 5)
        blue_region_offset_pre = (
            region_offsets.get(blue_region, 0.0)
            if region_enabled and blue_region is not None
            else 0.0
        )
        red_region_offset_pre = (
            region_offsets.get(red_region, 0.0)
            if region_enabled and red_region is not None
            else 0.0
        )
        blue_pre += blue_region_offset_pre
        red_pre += red_region_offset_pre
        probability = _win_probability(blue_pre, red_pre, blue_advantage)
        blue_error = actual_blue_win - probability
        rookie_game = any(
            games_played.get(player_key, 0) < 20
            for player_key, _ in blue_roster + red_roster
        )

        blue_change = 0.0
        red_change = 0.0
        for roster, side_error, team_key in (
            (blue_roster, blue_error, blue_team_key),
            (red_roster, -blue_error, red_team_key),
        ):
            player_keys = frozenset(player_key for player_key, _ in roster)
            team_history.setdefault(team_key, deque(maxlen=continuity_n)).append(
                player_keys
            )
            for player_key, role in roster:
                played = games_played.get(player_key, 0)
                shrink = max(k_min_ratio, 1 / (1 + played / k_decay))
                season_played = (
                    season_games_played.get((player_key, season_key), 0)
                    if season_key is not None
                    else new_season_m
                )
                season_multiplier = (
                    new_season_k_mult
                    if offseason_enabled and season_played < new_season_m
                    else 1.0
                )
                player_delta = (
                    k_base
                    * side_error
                    * role_weights[role]
                    * shrink
                    * season_multiplier
                )
                player_ratings[player_key] = (
                    player_ratings.get(player_key, starting_rating) + player_delta
                )
                games_played[player_key] = played + 1
                if season_key is not None:
                    season_count_key = (player_key, season_key)
                    season_games_played[season_count_key] = (
                        season_games_played.get(season_count_key, 0) + 1
                    )
                latest_roles[player_key] = role
                if team_key == blue_team_key:
                    blue_change += player_delta
                else:
                    red_change += player_delta

        region_delta = 0.0
        if (
            region_enabled
            and blue_region is not None
            and red_region is not None
            and blue_region != red_region
        ):
            region_delta = k_region * k_base * blue_error
            region_offsets[blue_region] = (
                region_offsets.get(blue_region, 0.0) + region_delta
            )
            region_offsets[red_region] = (
                region_offsets.get(red_region, 0.0) - region_delta
            )

        if direct_region is not None and season_key is not None:
            team_regions[(season_key, blue_team_key)] = direct_region
            team_regions[(season_key, red_team_key)] = direct_region

        year = values["year"]
        predictions.append(
            {
                **values,
                "blue_team_key": blue_team_key,
                "red_team_key": red_team_key,
                "winner_team_key": winner_key,
                "actual_blue_win": actual_blue_win,
                "predicted_blue_win_prob": probability,
                "rookie_game": rookie_game,
                "blue_rating_pre": blue_pre,
                "red_rating_pre": red_pre,
                "blue_region": blue_region,
                "red_region": red_region,
                "blue_region_offset_pre": blue_region_offset_pre,
                "red_region_offset_pre": red_region_offset_pre,
                "region_rating_change": region_delta,
                "rating_change_blue": blue_change,
                "rating_change_red": red_change,
                "blue_continuity_starters": blue_continuity,
                "red_continuity_starters": red_continuity,
                "scored": start_year is None or (pd.notna(year) and year >= start_year),
            }
        )
        if record_history:
            league_mean = (
                sum(player_ratings.values()) / len(player_ratings)
                if player_ratings
                else starting_rating
            )
            for roster in (blue_roster, red_roster):
                for player_key, _ in roster:
                    rating_history.append(
                        {
                            "player_key": player_key,
                            "date": values["date"],
                            "gameid": gameid,
                            "league": league,
                            "year": year,
                            "rating": player_ratings[player_key],
                            "league_mean": league_mean,
                            "games_played": games_played[player_key],
                        }
                    )
        if season_key is not None and last_position_by_year[season] == position:
            offset_snapshots.extend(
                {"year": season_key, "region": region, "offset": offset}
                for region, offset in sorted(region_offsets.items())
            )

    rating_rows = [
        {
            "player_key": player_key,
            "playername": player_names.get(player_key),
            "rating": rating,
            "games_played": games_played[player_key],
            "region": player_regions.get(player_key),
        }
        for player_key, rating in player_ratings.items()
    ]
    ratings = (
        pd.DataFrame(
            rating_rows,
            columns=["player_key", "playername", "rating", "games_played", "region"],
        )
        .sort_values("rating", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    role_rows = [
        {
            "player_key": player_key,
            "playername": player_names.get(player_key),
            "role": role,
            "rating": player_ratings[player_key],
            "games_played": games_played[player_key],
        }
        for player_key, role in latest_roles.items()
    ]
    role_ratings = pd.DataFrame(
        role_rows,
        columns=["player_key", "playername", "role", "rating", "games_played"],
    ).sort_values(["role", "rating"], ascending=[True, False], kind="stable")
    return PlayerEloResult(
        pd.DataFrame(predictions),
        ratings,
        role_ratings,
        region_offsets,
        pd.DataFrame(offset_snapshots, columns=["year", "region", "offset"]),
        pd.DataFrame(rating_history),
    )


def _comparison_table(
    player_evaluation: EvaluationResult,
    baseline_evaluation: EvaluationResult,
) -> pd.DataFrame:
    player_games = player_evaluation.metrics.loc[
        player_evaluation.metrics["scope"].isin(["overall", "year"]),
        ["scope", "segment", "elo_n", "elo_log_loss", "elo_brier", "elo_accuracy"],
    ].rename(
        columns={
            "elo_n": "games",
            "elo_log_loss": "player_log_loss",
            "elo_brier": "player_brier",
            "elo_accuracy": "player_accuracy",
        }
    )
    baseline_games = baseline_evaluation.metrics.loc[
        baseline_evaluation.metrics["scope"].isin(["overall", "year"]),
        ["scope", "segment", "elo_n", "elo_log_loss", "elo_brier", "elo_accuracy"],
    ].rename(
        columns={
            "elo_n": "baseline_games",
            "elo_log_loss": "baseline_log_loss",
            "elo_brier": "baseline_brier",
            "elo_accuracy": "baseline_accuracy",
        }
    )
    comparison = player_games.merge(
        baseline_games, on=["scope", "segment"], how="inner", validate="one_to_one"
    )
    player_series = player_evaluation.series_metrics.loc[
        player_evaluation.series_metrics["scope"].isin(["overall", "year"]),
        ["scope", "segment", "series_log_loss"],
    ].rename(columns={"series_log_loss": "player_series_log_loss"})
    baseline_series = baseline_evaluation.series_metrics.loc[
        baseline_evaluation.series_metrics["scope"].isin(["overall", "year"]),
        ["scope", "segment", "series_log_loss"],
    ].rename(columns={"series_log_loss": "baseline_series_log_loss"})
    comparison = comparison.merge(
        player_series, on=["scope", "segment"], how="left", validate="one_to_one"
    ).merge(
        baseline_series,
        on=["scope", "segment"],
        how="left",
        validate="one_to_one",
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the player-level Elo model")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--games", default="data/processed/games.parquet")
    parser.add_argument("--players", default="data/processed/players.parquet")
    parser.add_argument(
        "--baseline-predictions",
        default="data/processed/baseline_preds.parquet",
    )
    parser.add_argument("--predictions", default="data/processed/player_preds.parquet")
    parser.add_argument(
        "--all-time",
        action="store_true",
        help="Include inactive players in the role ranking output",
    )
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG_PATH))
    args = parser.parse_args()

    games = pd.read_parquet(args.games)
    players = pd.read_parquet(args.players)
    baseline_predictions = pd.read_parquet(args.baseline_predictions)
    config = load_model_config(args.model_config)
    initial_ratings = None
    if config.get("use_rookie_priors", False):
        from lol_elo.priors import (
            build_initial_rookie_ratings,
            load_fitted_priors,
            load_or_build_lower_history,
        )

        lower_games = pd.read_parquet("data/processed/games_unfiltered.parquet")
        lower_players = pd.read_parquet("data/processed/players_unfiltered.parquet")
        lower_history = load_or_build_lower_history(lower_games, lower_players, config)
        fitted = load_fitted_priors()
        initial_ratings = build_initial_rookie_ratings(
            games, players, lower_history, fitted
        )
    result = run_player_elo(
        games,
        players,
        config,
        start_year=args.start_year,
        initial_ratings=initial_ratings,
    )
    evaluation = evaluate_predictions(
        result.predictions, config, start_year=args.start_year
    )

    if "scored" in result.predictions:
        player_scored = result.predictions.loc[
            result.predictions["scored"].astype(bool)
        ]
    elif args.start_year is not None:
        player_scored = result.predictions.loc[
            pd.to_numeric(result.predictions["year"], errors="coerce").ge(
                args.start_year
            )
        ]
    else:
        player_scored = result.predictions
    if "scored" in baseline_predictions:
        baseline_scored = baseline_predictions.loc[
            baseline_predictions["scored"].astype(bool)
        ]
    elif args.start_year is not None:
        baseline_scored = baseline_predictions.loc[
            pd.to_numeric(baseline_predictions["year"], errors="coerce").ge(
                args.start_year
            )
        ]
    else:
        baseline_scored = baseline_predictions
    player_ids = set(player_scored["gameid"])
    baseline_ids = set(baseline_scored["gameid"])
    common_ids = player_ids & baseline_ids
    if player_ids != baseline_ids:
        print(
            "Comparison uses the common scored games: "
            f"{len(common_ids)} shared, {len(player_ids - common_ids)} player-only, "
            f"{len(baseline_ids - common_ids)} baseline-only."
        )
    if not common_ids:
        raise ValueError("Player and baseline predictions have no common scored games")

    common_player = result.predictions.loc[
        result.predictions["gameid"].isin(common_ids)
    ].copy()
    common_baseline = baseline_predictions.loc[
        baseline_predictions["gameid"].isin(common_ids)
    ].copy()
    common_player["scored"] = True
    common_baseline["scored"] = True
    comparison_player_eval = evaluate_predictions(common_player, config)
    comparison_baseline_eval = evaluate_predictions(common_baseline, config)
    comparison = _comparison_table(comparison_player_eval, comparison_baseline_eval)

    output_path = Path(args.predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(output_path, index=False)

    print("Player Elo metrics:")
    print(evaluation.metrics.to_string(index=False))
    print("\n10-bin calibration table:")
    print(evaluation.calibration.to_string(index=False))
    print("\nSeries metrics:")
    if evaluation.series_metrics.empty:
        print("No completed series to score.")
    else:
        print(evaluation.series_metrics.to_string(index=False))
    print("\nPlayer Elo vs team Elo on the same scored games:")
    print(comparison.to_string(index=False))
    print("\nRegional offsets at year ends:")
    if result.region_offset_history.empty:
        print("No season snapshots.")
    else:
        for year, offsets in result.region_offset_history.groupby("year", sort=False):
            print(f"\n{year}:")
            print(offsets[["region", "offset"]].to_string(index=False))
    print("\nFinal regional offsets:")
    if result.region_offsets:
        print(
            pd.Series(result.region_offsets, name="offset")
            .rename_axis("region")
            .sort_index()
            .to_string()
        )
    else:
        print("No configured regions.")

    role_ratings = result.role_ratings.merge(
        result.ratings[["player_key", "region"]], on="player_key", how="left"
    )
    role_ratings["effective_rating"] = role_ratings["rating"] + role_ratings[
        "region"
    ].map(result.region_offsets).fillna(0.0)
    role_ratings = role_ratings.sort_values(
        ["role", "effective_rating"], ascending=[True, False], kind="stable"
    )
    if not args.all_time:
        active_keys = _recent_player_keys(players, games)
        role_ratings = role_ratings.loc[role_ratings["player_key"].isin(active_keys)]
        latest_game = pd.to_datetime(games["date"], errors="coerce").max()
        cutoff = latest_game - pd.Timedelta(days=365)
        print(
            "\nTop 10 players per role by effective rating "
            f"(active since {cutoff:%Y-%m-%d}):"
        )
    else:
        print("\nTop 10 players per role by effective rating (all time):")
    for role in ROLES:
        top = role_ratings.loc[role_ratings["role"].eq(role)].head(10)
        print(f"\n{role}:")
        print(
            top[
                [
                    "playername",
                    "player_key",
                    "rating",
                    "region",
                    "effective_rating",
                    "games_played",
                ]
            ].to_string(index=False)
        )
    print(f"\nSaved predictions to {output_path}")
    if args.start_year is not None:
        print(
            f"Games before {args.start_year} updated ratings as burn-in "
            "and were not scored."
        )
