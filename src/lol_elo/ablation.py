"""Compare all feature-flag combinations for the player Elo model."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from lol_elo.evaluate import evaluate_predictions
from lol_elo.player_elo import run_player_elo
from lol_elo.team_elo import DEFAULT_MODEL_CONFIG_PATH, load_model_config


def run_ablation_study(
    games: pd.DataFrame,
    players: pd.DataFrame,
    config: dict,
    start_year: int = 2019,
) -> pd.DataFrame:
    """Evaluate the four regional-strength/offseason-regression combinations."""
    cases = (
        (False, False, "neither"),
        (True, False, "regional only"),
        (False, True, "offseason only"),
        (True, True, "both"),
    )
    rows = []
    for region_enabled, offseason_enabled, name in cases:
        run_config = deepcopy(config)
        run_config["region_strength_enabled"] = region_enabled
        run_config["offseason_regression_enabled"] = offseason_enabled
        result = run_player_elo(games, players, run_config, start_year=start_year)
        evaluation = evaluate_predictions(
            result.predictions, run_config, start_year=start_year
        )
        overall = evaluation.metrics.loc[
            evaluation.metrics["scope"].eq("overall")
        ].iloc[0]
        international = evaluation.metrics.loc[
            evaluation.metrics["scope"].eq("competition")
            & evaluation.metrics["segment"].eq("international")
        ]
        series = evaluation.series_metrics.loc[
            evaluation.series_metrics["scope"].eq("overall")
        ]
        rows.append(
            {
                "configuration": name,
                "region_strength": region_enabled,
                "offseason_regression": offseason_enabled,
                "continuity_bonus": run_config.get("continuity_bonus", 0),
                "games": int(overall["elo_n"]),
                "game_log_loss": overall["elo_log_loss"],
                "brier": overall["elo_brier"],
                "accuracy": overall["elo_accuracy"],
                "series_log_loss": (
                    series.iloc[0]["series_log_loss"]
                    if not series.empty
                    else float("nan")
                ),
                "international_log_loss": (
                    international.iloc[0]["elo_log_loss"]
                    if not international.empty
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(results: pd.DataFrame) -> str:
    columns = list(results.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in results.itertuples(index=False, name=None):
        values = [
            f"{value:.4f}" if isinstance(value, float) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run player Elo feature ablations")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--games", default="data/processed/games.parquet")
    parser.add_argument("--players", default="data/processed/players.parquet")
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG_PATH))
    parser.add_argument("--notes", default="notes/ablations.md")
    args = parser.parse_args()

    games = pd.read_parquet(args.games)
    players = pd.read_parquet(args.players)
    config = load_model_config(args.model_config)
    results = run_ablation_study(games, players, config, args.start_year)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    notes_path = Path(args.notes)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(
        "# Player Elo feature ablations\n\n"
        f"Scored games from {args.start_year} onward; earlier games are burn-in. "
        f"Continuity bonus was held at {config.get('continuity_bonus', 0)} "
        "for every run.\n\n"
        + _markdown_table(results)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {notes_path}")
