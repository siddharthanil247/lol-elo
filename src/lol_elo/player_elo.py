"""Walk-forward player-level Elo ratings aggregated into team strength."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from lol_elo.evaluate import EvaluationResult, evaluate_predictions
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


def _win_probability(
    blue_rating: float, red_rating: float, blue_advantage: float
) -> float:
    """Return blue's win probability from team ratings and side advantage."""
    return _probability(blue_rating + blue_advantage - red_rating)


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
    return sum(
        player_ratings.get(player_key, starting_rating) * weight
        for (player_key, _), weight in zip(roster, weights, strict=True)
    ) / weight_total


def run_player_elo(
    games: pd.DataFrame,
    players: pd.DataFrame,
    config: dict | None = None,
    start_year: int | None = None,
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
    games_played: dict[str, int] = {}
    latest_roles: dict[str, str] = {}
    team_history: dict[str, deque[frozenset[str]]] = {}
    predictions = []

    for game in ordered.itertuples(index=False):
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
        winner_key = _team_key(values.get("winner_team_id"), values["winner"])
        if winner_key == blue_team_key:
            actual_blue_win = 1
        elif winner_key == red_team_key:
            actual_blue_win = 0
        else:
            raise ValueError(f"Winner does not match either team in game {gameid}")

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
        probability = _win_probability(blue_pre, red_pre, blue_advantage)
        blue_error = actual_blue_win - probability

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
                player_delta = (
                    k_base * side_error
                    * role_weights[role]
                    * shrink
                )
                player_ratings[player_key] = (
                    player_ratings.get(player_key, starting_rating) + player_delta
                )
                games_played[player_key] = played + 1
                latest_roles[player_key] = role
                if team_key == blue_team_key:
                    blue_change += player_delta
                else:
                    red_change += player_delta

        year = values["year"]
        predictions.append(
            {
                **values,
                "blue_team_key": blue_team_key,
                "red_team_key": red_team_key,
                "winner_team_key": winner_key,
                "actual_blue_win": actual_blue_win,
                "predicted_blue_win_prob": probability,
                "blue_rating_pre": blue_pre,
                "red_rating_pre": red_pre,
                    "rating_change_blue": blue_change,
                    "rating_change_red": red_change,
                "blue_continuity_starters": blue_continuity,
                "red_continuity_starters": red_continuity,
                "scored": start_year is None
                or (pd.notna(year) and year >= start_year),
            }
        )

    rating_rows = [
        {
            "player_key": player_key,
            "playername": player_names.get(player_key),
            "rating": rating,
            "games_played": games_played[player_key],
        }
        for player_key, rating in player_ratings.items()
    ]
    ratings = pd.DataFrame(
        rating_rows, columns=["player_key", "playername", "rating", "games_played"]
    ).sort_values("rating", ascending=False, kind="stable").reset_index(drop=True)
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
    return PlayerEloResult(pd.DataFrame(predictions), ratings, role_ratings)


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
    parser.add_argument(
        "--predictions", default="data/processed/player_preds.parquet"
    )
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG_PATH))
    args = parser.parse_args()

    games = pd.read_parquet(args.games)
    players = pd.read_parquet(args.players)
    baseline_predictions = pd.read_parquet(args.baseline_predictions)
    config = load_model_config(args.model_config)
    result = run_player_elo(games, players, config, start_year=args.start_year)
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
    print("\nTop 10 players per role by current rating:")
    for role in ROLES:
        top = result.role_ratings.loc[result.role_ratings["role"].eq(role)].head(10)
        print(f"\n{role}:")
        print(
            top[["playername", "player_key", "rating", "games_played"]].to_string(
                index=False
            )
        )
    print(f"\nSaved predictions to {output_path}")
    if args.start_year is not None:
        print(
            f"Games before {args.start_year} updated ratings as burn-in "
            "and were not scored."
        )
