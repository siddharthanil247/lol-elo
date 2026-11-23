"""Command-line runner for the team-level Elo baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lol_elo.evaluate import evaluate_predictions
from lol_elo.team_elo import (
    DEFAULT_MODEL_CONFIG_PATH,
    load_model_config,
    run_team_elo,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the team Elo baseline")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--games", default="data/processed/games.parquet")
    parser.add_argument(
        "--predictions", default="data/processed/baseline_preds.parquet"
    )
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG_PATH))
    args = parser.parse_args()

    games = pd.read_parquet(args.games)
    config = load_model_config(args.model_config)
    result = run_team_elo(games, config, start_year=args.start_year)
    evaluation = evaluate_predictions(
        result.predictions, config, start_year=args.start_year
    )

    predictions_path = Path(args.predictions)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(predictions_path, index=False)

    print("Game metrics (walk-forward):")
    print(evaluation.metrics.to_string(index=False))
    print("\n10-bin calibration table:")
    print(evaluation.calibration.to_string(index=False))
    print("\nSeries metrics:")
    if evaluation.series_metrics.empty:
        print("No completed series to score.")
    else:
        print(evaluation.series_metrics.to_string(index=False))
    print(f"\nSaved predictions to {predictions_path}")
    if args.start_year is not None:
        print(
            f"Games before {args.start_year} updated ratings as burn-in "
            "and were not scored."
        )
