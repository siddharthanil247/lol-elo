"""League configuration and Tier 1 match filtering."""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "leagues.yaml"


def load_leagues(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load league names and optional year ranges from a YAML configuration."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict) or not isinstance(
        config.get("tier1_domestic"), list
    ):
        raise ValueError("League config must contain a tier1_domestic list")
    if not isinstance(config.get("international"), list):
        raise ValueError("League config must contain an international list")
    return config


def filter_tier1_games(
    games: pd.DataFrame, config_path: str | Path = DEFAULT_CONFIG_PATH
) -> pd.DataFrame:
    """Return rows whose league name and year qualify as Tier 1."""
    required_columns = {"league", "year"}
    missing_columns = required_columns - set(games.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Games data is missing required columns: {missing}")

    config = load_leagues(config_path)
    eligible = pd.Series(False, index=games.index)
    for entry in config["tier1_domestic"] + config["international"]:
        if isinstance(entry, str):
            league_name, start_year, end_year = entry, None, None
        elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
            league_name = entry["name"]
            start_year = entry.get("start_year")
            end_year = entry.get("end_year")
        else:
            raise ValueError(f"Invalid league entry: {entry!r}")

        matches = games["league"].eq(league_name)
        if start_year is not None:
            matches &= games["year"].ge(start_year)
        if end_year is not None:
            matches &= games["year"].le(end_year)
        eligible |= matches.fillna(False)

    return games.loc[eligible].copy()
