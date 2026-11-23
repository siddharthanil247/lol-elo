"""Walk-forward evaluation for rookie-prior ablations."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from lol_elo.evaluate import evaluate_predictions
from lol_elo.player_elo import run_player_elo
from lol_elo.priors import build_initial_rookie_ratings


def _log_loss(predictions: pd.DataFrame) -> float:
    if predictions.empty:
        return float("nan")
    p = np.clip(
        predictions["predicted_blue_win_prob"].to_numpy(float), 1e-15, 1 - 1e-15
    )
    y = predictions["actual_blue_win"].to_numpy(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluate_rookie_priors(
    games: pd.DataFrame,
    players: pd.DataFrame,
    lower_history: pd.DataFrame,
    config: dict,
    fitted: dict,
    start_year: int = 2019,
) -> pd.DataFrame:
    """Compare flat and fitted priors on identical games and slices."""
    initial = build_initial_rookie_ratings(games, players, lower_history, fitted)
    rows = []
    for enabled in (False, True):
        run_config = deepcopy(config)
        run_config["use_rookie_priors"] = enabled
        result = run_player_elo(
            games,
            players,
            run_config,
            start_year=start_year,
            initial_ratings=initial if enabled else None,
        )
        predictions = result.predictions.loc[
            result.predictions["scored"].astype(bool)
        ].copy()
        year = pd.to_numeric(predictions["year"], errors="coerce")
        for scope, mask in (
            ("overall", pd.Series(True, index=predictions.index)),
            ("2024 onward", year.ge(2024)),
        ):
            subset = predictions.loc[mask].copy()
            evaluation = evaluate_predictions(subset.assign(scored=True), run_config)
            overall = evaluation.metrics.loc[evaluation.metrics["scope"].eq("overall")]
            series = evaluation.series_metrics.loc[
                evaluation.series_metrics["scope"].eq("overall")
            ]
            rookie = subset.loc[subset["rookie_game"].astype(bool)]
            rows.append(
                {
                    "configuration": "rookie priors on"
                    if enabled
                    else "rookie priors off",
                    "scope": scope,
                    "games": len(subset),
                    "game_log_loss": float(overall.iloc[0]["elo_log_loss"])
                    if not overall.empty
                    else float("nan"),
                    "series_log_loss": float(series.iloc[0]["series_log_loss"])
                    if not series.empty
                    else float("nan"),
                    "rookie_games": len(rookie),
                    "rookie_game_log_loss": _log_loss(rookie),
                }
            )
    return pd.DataFrame(rows)
