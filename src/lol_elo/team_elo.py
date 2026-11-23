"""Walk-forward team-level Elo ratings for Tier 1 games."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd
import yaml

DEFAULT_MODEL_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "model.yaml"
)


class EloResult(NamedTuple):
    predictions: pd.DataFrame
    ratings: pd.DataFrame


def load_model_config(config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH) -> dict:
    """Read Elo and series settings from YAML."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Model config must be a YAML mapping")
    return config


def _team_key(team_id: object, team_name: object) -> str:
    if pd.notna(team_id) and str(team_id).strip():
        return f"teamid:{str(team_id).strip()}"
    if pd.notna(team_name) and str(team_name).strip():
        return f"teamname:{str(team_name).strip().lower()}"
    raise ValueError("A game has a team without a teamid or teamname")


def _probability(rating_difference: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-rating_difference / 400.0))


def run_team_elo(
    games: pd.DataFrame,
    config: dict | None = None,
    start_year: int | None = None,
) -> EloResult:
    """Predict each game before updating ratings and return final team ratings."""
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

    k_factor = float(config.get("k_factor", 32))
    starting_rating = float(config.get("starting_rating", 1500))
    blue_advantage = float(config.get("blue_side_advantage", 0))
    ordered = games.copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
    ordered["_game_number"] = pd.to_numeric(ordered["game"], errors="coerce")
    ordered["year"] = pd.to_numeric(ordered["year"], errors="coerce")
    ordered = ordered.sort_values(
        ["date", "gameid", "_game_number"],
        kind="stable",
        na_position="last",
    )
    ordered = ordered.drop(columns=["_game_number"])

    ratings_by_key: dict[str, float] = {}
    labels_by_key: dict[str, tuple[object, object]] = {}
    prediction_rows = []
    for row in ordered.itertuples(index=False):
        values = row._asdict()
        blue_key = _team_key(values.get("blue_team_id"), values["blue_team"])
        red_key = _team_key(values.get("red_team_id"), values["red_team"])
        winner_key = _team_key(values.get("winner_team_id"), values["winner"])
        if winner_key == blue_key:
            actual_blue_win = 1
        elif winner_key == red_key:
            actual_blue_win = 0
        else:
            raise ValueError(
                f"Winner does not match either team in game {values['gameid']}"
            )

        blue_pre = ratings_by_key.get(blue_key, starting_rating)
        red_pre = ratings_by_key.get(red_key, starting_rating)
        probability = _probability(blue_pre + blue_advantage - red_pre)
        blue_change = k_factor * (actual_blue_win - probability)
        red_change = -blue_change
        blue_post = blue_pre + blue_change
        red_post = red_pre + red_change
        ratings_by_key[blue_key] = blue_post
        ratings_by_key[red_key] = red_post
        labels_by_key[blue_key] = (values.get("blue_team_id"), values["blue_team"])
        labels_by_key[red_key] = (values.get("red_team_id"), values["red_team"])

        year = values["year"]
        prediction_rows.append(
            {
                **values,
                "blue_team_key": blue_key,
                "red_team_key": red_key,
                "winner_team_key": winner_key,
                "actual_blue_win": actual_blue_win,
                "predicted_blue_win_prob": probability,
                "blue_rating_pre": blue_pre,
                "red_rating_pre": red_pre,
                "blue_rating_post": blue_post,
                "red_rating_post": red_post,
                "rating_change_blue": blue_change,
                "rating_change_red": red_change,
                "scored": start_year is None or (pd.notna(year) and year >= start_year),
            }
        )

    predictions = pd.DataFrame(prediction_rows)
    if not predictions.empty:
        predictions = predictions.drop(columns=["_game_number"], errors="ignore")
    rating_rows = [
        {
            "team_key": key,
            "teamid": labels_by_key[key][0],
            "teamname": labels_by_key[key][1],
            "rating": rating,
        }
        for key, rating in ratings_by_key.items()
    ]
    final_ratings = pd.DataFrame(
        rating_rows, columns=["team_key", "teamid", "teamname", "rating"]
    ).sort_values("rating", ascending=False, kind="stable").reset_index(drop=True)
    return EloResult(predictions, final_ratings)
