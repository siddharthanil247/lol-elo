"""Walk-forward metrics and series-level evaluation for game predictions."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from lol_elo.team_elo import load_model_config

EPSILON = 1e-15


class EvaluationResult(NamedTuple):
    metrics: pd.DataFrame
    calibration: pd.DataFrame
    series_predictions: pd.DataFrame
    series_metrics: pd.DataFrame


def _binary_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    y = actual.to_numpy(dtype=float)
    p = np.clip(predicted.to_numpy(dtype=float), EPSILON, 1 - EPSILON)
    if not len(y):
        return {"n": 0, "log_loss": np.nan, "brier": np.nan, "accuracy": np.nan}
    return {
        "n": len(y),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
    }


def _metric_row(
    scope: str,
    segment: object,
    frame: pd.DataFrame,
) -> dict[str, object]:
    actual = frame["actual_blue_win"].astype(float)
    metrics = {"scope": scope, "segment": segment}
    for label, column in (
        ("elo", "predicted_blue_win_prob"),
        ("coinflip", "_coinflip"),
        ("historical_blue_rate", "_historical_blue_rate"),
    ):
        for name, value in _binary_metrics(actual, frame[column]).items():
            metrics[f"{label}_{name}"] = value
    return metrics


def _series_best_of(series: pd.DataFrame, config: dict) -> int:
    formats = config.get("series_formats", {})
    league = str(series["league"].iloc[0])
    playoffs = series["playoffs"].iloc[0]
    playoff_value = str(playoffs).strip().lower() in {"1", "true", "yes", "y"}
    stage = "playoffs" if playoff_value else "regular"
    league_format = formats.get("leagues", {}).get(league, {})
    default_format = formats.get("default", {})
    best_of = int(league_format.get(stage, default_format.get(stage, 1)))
    observed_max = int(series["_game_number"].max())
    if observed_max > best_of:
        best_of = observed_max if observed_max % 2 else observed_max + 1
    return max(best_of, 1)


def _elo_probability(rating_difference: float) -> float:
    return 1 / (1 + 10 ** (-rating_difference / 400))


def _series_win_probability(game_probabilities: list[float], best_of: int) -> float:
    wins_needed = best_of // 2 + 1
    state = np.zeros(wins_needed + 1, dtype=float)
    state[0] = 1.0
    for probability in game_probabilities:
        updated = np.zeros_like(state)
        updated[wins_needed] = state[wins_needed]
        for wins in range(wins_needed):
            updated[wins] += state[wins] * (1 - probability)
            updated[wins + 1] += state[wins] * probability
        state = updated
    return float(state[wins_needed])


def _evaluate_series(
    scored: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scored.empty:
        return pd.DataFrame(), pd.DataFrame()

    series_games = scored.copy()
    series_games["date"] = pd.to_datetime(series_games["date"], errors="coerce")
    series_games["_date"] = series_games["date"].dt.normalize()
    series_games["_game_number"] = pd.to_numeric(
        series_games["game"], errors="coerce"
    ).fillna(1)
    blue_key = series_games["blue_team_key"]
    red_key = series_games["red_team_key"]
    series_games["_team_a"] = blue_key.where(blue_key.le(red_key), red_key)
    series_games["_team_b"] = red_key.where(blue_key.le(red_key), blue_key)
    partitions = ["league", "_date", "_team_a", "_team_b"]
    series_games = series_games.sort_values(
        [*partitions, "_game_number", "gameid"], kind="stable"
    )
    grouped = series_games.groupby(partitions, sort=False, dropna=False)
    previous_number = grouped["_game_number"].shift()
    series_games["_series_start"] = (
        previous_number.isna()
        | series_games["_game_number"].ne(previous_number + 1)
    )
    series_games["_series_segment"] = series_games.groupby(
        partitions, sort=False, dropna=False
    )["_series_start"].cumsum()
    series_games["_series_id"] = (
        series_games.groupby(partitions, sort=False, dropna=False)
        .ngroup()
        .astype(str)
        + "-"
        + series_games["_series_segment"].astype(str)
    )

    model = config
    blue_advantage = float(model.get("blue_side_advantage", 0))
    records = []
    for series_id, series in series_games.groupby("_series_id", sort=False):
        series = series.sort_values(["_game_number", "gameid"], kind="stable")
        best_of = _series_best_of(series, config)
        team_a = series["_team_a"].iloc[0]
        team_b = series["_team_b"].iloc[0]
        first = series.iloc[0]
        if first["blue_team_key"] == team_a:
            rating_a = float(first["blue_rating_pre"])
            rating_b = float(first["red_rating_pre"])
            team_a_blue_first = True
        else:
            rating_a = float(first["red_rating_pre"])
            rating_b = float(first["blue_rating_pre"])
            team_a_blue_first = False

        first_game_number = int(first["_game_number"])
        per_game_probabilities = []
        # Future games use a frozen pre-series rating snapshot and alternating sides.
        for game_number in range(1, best_of + 1):
            team_a_blue = (
                team_a_blue_first
                if (game_number - first_game_number) % 2 == 0
                else not team_a_blue_first
            )
            difference = rating_a - rating_b + (
                blue_advantage if team_a_blue else -blue_advantage
            )
            per_game_probabilities.append(_elo_probability(difference))

        probability_a = _series_win_probability(per_game_probabilities, best_of)
        team_a_game_wins = (
            series["actual_blue_win"].astype(bool)
            .where(
                series["blue_team_key"].eq(team_a),
                ~series["actual_blue_win"].astype(bool),
            )
            .sum()
        )
        team_b_game_wins = len(series) - int(team_a_game_wins)
        wins_needed = best_of // 2 + 1
        if team_a_game_wins < wins_needed and team_b_game_wins < wins_needed:
            continue
        actual_a_win = int(team_a_game_wins >= wins_needed)
        records.append(
            {
                "series_id": series_id,
                "league": series["league"].iloc[0],
                "year": series["year"].iloc[0],
                "is_international": bool(series["is_international"].iloc[0]),
                "date": series["date"].iloc[0],
                "team_a_key": team_a,
                "team_b_key": team_b,
                "best_of": best_of,
                "games_played": len(series),
                "team_a_win_probability": probability_a,
                "actual_team_a_win": actual_a_win,
            }
        )

    series_predictions = pd.DataFrame(records)
    if series_predictions.empty:
        return series_predictions, pd.DataFrame()

    rows = []
    for scope, group_columns in (
        ("overall", []),
        ("year", ["year"]),
        ("competition", ["is_international"]),
    ):
        if not group_columns:
            groups = [("all", series_predictions)]
        else:
            grouper: str | list[str] = (
                group_columns[0] if len(group_columns) == 1 else group_columns
            )
            groups = series_predictions.groupby(grouper, sort=True, dropna=False)
        for segment, frame in groups:
            if scope == "competition":
                segment = "international" if bool(segment) else "domestic"
            p = frame["team_a_win_probability"].clip(EPSILON, 1 - EPSILON)
            y = frame["actual_team_a_win"].astype(float)
            rows.append(
                {
                    "scope": scope,
                    "segment": segment,
                    "n": len(frame),
                    "series_log_loss": float(
                        -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                    ),
                    "series_accuracy": float(
                        np.mean((p >= 0.5) == (y == 1))
                    ),
                }
            )
    return series_predictions, pd.DataFrame(rows)


def evaluate_predictions(
    predictions: pd.DataFrame,
    config: dict | None = None,
    start_year: int | None = None,
) -> EvaluationResult:
    """Compute game, calibration, benchmark, and series walk-forward metrics."""
    config = config or load_model_config()
    required = {
        "date",
        "year",
        "is_international",
        "actual_blue_win",
        "predicted_blue_win_prob",
    }
    missing = required - set(predictions.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"Predictions are missing columns: {missing_names}")

    ordered = predictions.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
    if "game" in ordered:
        ordered["_game_number"] = pd.to_numeric(ordered["game"], errors="coerce")
    sort_columns = ["date"]
    if "gameid" in ordered:
        sort_columns.append("gameid")
    if "_game_number" in ordered:
        sort_columns.append("_game_number")
    ordered = ordered.sort_values(sort_columns, kind="stable", na_position="last")
    actual_all = ordered["actual_blue_win"].astype(float)
    ordered["_coinflip"] = 0.5
    ordered["_historical_blue_rate"] = actual_all.expanding().mean().shift().fillna(0.5)
    if "scored" in ordered:
        scored = ordered.loc[ordered["scored"].astype(bool)].copy()
    elif start_year is not None:
        years = pd.to_numeric(ordered["year"], errors="coerce")
        scored = ordered.loc[years.ge(start_year)].copy()
    else:
        scored = ordered.copy()

    metric_rows = [_metric_row("overall", "all", scored)]
    for scope, column in (("year", "year"), ("competition", "is_international")):
        for segment, frame in scored.groupby(column, sort=True, dropna=False):
            if scope == "competition":
                segment = "international" if bool(segment) else "domestic"
            metric_rows.append(_metric_row(scope, segment, frame))
    metrics = pd.DataFrame(metric_rows)

    calibration = pd.DataFrame(columns=["bin", "n", "mean_predicted", "observed_rate"])
    if not scored.empty:
        bins = pd.cut(
            scored["predicted_blue_win_prob"].clip(0, 1),
            bins=np.linspace(0, 1, 11),
            include_lowest=True,
            labels=False,
        )
        calibration = (
            scored.assign(_calibration_bin=bins)
            .groupby("_calibration_bin", observed=False)
            .agg(
                n=("actual_blue_win", "size"),
                mean_predicted=("predicted_blue_win_prob", "mean"),
                observed_rate=("actual_blue_win", "mean"),
            )
            .reindex(range(10))
            .rename_axis("bin")
            .reset_index()
        )

    series_predictions, series_metrics = _evaluate_series(scored, config)
    return EvaluationResult(metrics, calibration, series_predictions, series_metrics)
