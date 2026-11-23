"""Load Oracle's Elixir CSVs and build player and game tables."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from lol_elo.leagues import DEFAULT_CONFIG_PATH, filter_tier1_games, load_leagues

LOGGER = logging.getLogger(__name__)
CORE_COLUMNS = [
    "gameid",
    "league",
    "year",
    "split",
    "playoffs",
    "date",
    "game",
    "patch",
    "side",
    "position",
    "playername",
    "playerid",
    "teamname",
    "teamid",
    "result",
    "gamelength",
]
GAME_COLUMNS = [
    "gameid",
    "blue_team",
    "red_team",
    "winner",
    "date",
    "league",
    "year",
    "is_international",
]
ALIASES_FILENAME = "player_aliases.csv"


class IngestResult(NamedTuple):
    players: pd.DataFrame
    games: pd.DataFrame
    players_unfiltered: pd.DataFrame
    games_unfiltered: pd.DataFrame
    dropped_missing_results: int
    dropped_insufficient_players: int


def _load_raw_csvs(raw_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(raw_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, low_memory=False)
        for column in CORE_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
        frames.append(frame[CORE_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def _player_key(row: pd.Series) -> str | None:
    player_id = row["playerid"]
    if pd.notna(player_id) and str(player_id).strip():
        return f"playerid:{str(player_id).strip()}"
    player_name = row["playername"]
    if pd.notna(player_name) and str(player_name).strip():
        return f"name:{str(player_name).strip().lower()}"
    return None


def _load_aliases(processed_dir: Path) -> dict[str, str]:
    alias_path = processed_dir / ALIASES_FILENAME
    if not alias_path.exists():
        pd.DataFrame(columns=["player_alias", "canonical_player_key"]).to_csv(
            alias_path, index=False
        )
        return {}

    aliases = pd.read_csv(alias_path, dtype="string").dropna(
        subset=["player_alias", "canonical_player_key"]
    )
    return dict(zip(aliases["player_alias"], aliases["canonical_player_key"], strict=True))


def _build_games(
    raw: pd.DataFrame, international_leagues: set[str]
) -> tuple[pd.DataFrame, int, int]:
    team_rows = raw.loc[raw["position"].astype("string").str.lower().eq("team")]
    player_rows = raw.loc[~raw.index.isin(team_rows.index)]
    records: list[dict[str, object]] = []
    dropped_missing_results = 0
    dropped_insufficient_players = 0

    for gameid, rows in raw.groupby("gameid", sort=False, dropna=False):
        game_team_rows = team_rows.loc[team_rows["gameid"].eq(gameid)]
        game_player_rows = player_rows.loc[player_rows["gameid"].eq(gameid)]
        results = pd.to_numeric(game_team_rows["result"], errors="coerce")
        if results.isna().any() or not results.eq(1).sum() == 1:
            dropped_missing_results += 1
            continue

        side_counts = game_player_rows.groupby(
            game_player_rows["side"].astype("string").str.lower(), dropna=False
        ).size()
        if side_counts.get("blue", 0) < 5 or side_counts.get("red", 0) < 5:
            dropped_insufficient_players += 1
            continue

        team_rows_with_side = game_team_rows.assign(
            _side=game_team_rows["side"].astype("string").str.lower()
        )
        blue_rows = team_rows_with_side.loc[team_rows_with_side["_side"].eq("blue")]
        red_rows = team_rows_with_side.loc[team_rows_with_side["_side"].eq("red")]
        if blue_rows.empty or red_rows.empty:
            dropped_missing_results += 1
            continue

        blue = blue_rows.iloc[0]
        red = red_rows.iloc[0]
        winner_row = game_team_rows.loc[results.eq(1)].iloc[0]
        records.append(
            {
                "gameid": gameid,
                "blue_team": blue["teamname"],
                "red_team": red["teamname"],
                "winner": winner_row["teamname"],
                "date": rows["date"].iloc[0],
                "league": rows["league"].iloc[0],
                "year": rows["year"].iloc[0],
                "is_international": rows["league"].iloc[0] in international_leagues,
            }
        )

    games = pd.DataFrame(records, columns=GAME_COLUMNS)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    return games, dropped_missing_results, dropped_insufficient_players


def ingest(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> IngestResult:
    """Ingest raw match files, save tables, and return filtered and full tables."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_raw_csvs(raw_dir)
    config = load_leagues(config_path)
    international_leagues = {
        entry if isinstance(entry, str) else entry["name"]
        for entry in config["international"]
    }
    games_unfiltered, dropped_missing, dropped_players = _build_games(
        raw, international_leagues
    )
    valid_gameids = set(games_unfiltered["gameid"])
    players_unfiltered = raw.loc[
        (raw["position"].astype("string").str.lower() != "team")
        & raw["gameid"].isin(valid_gameids)
    ].copy()
    players_unfiltered["date"] = pd.to_datetime(
        players_unfiltered["date"], errors="coerce"
    )
    players_unfiltered["player_key"] = players_unfiltered.apply(_player_key, axis=1)

    aliases = _load_aliases(processed_dir)
    players_unfiltered["player_key"] = players_unfiltered["player_key"].map(
        lambda key: aliases.get(key, key) if pd.notna(key) else key
    )
    games = filter_tier1_games(games_unfiltered, config_path)
    tier1_gameids = set(games["gameid"])
    players = players_unfiltered.loc[
        players_unfiltered["gameid"].isin(tier1_gameids)
    ].copy()

    games_unfiltered = games_unfiltered.sort_values(
        "date", kind="stable", na_position="last"
    ).reset_index(drop=True)
    games = games.sort_values("date", kind="stable", na_position="last").reset_index(
        drop=True
    )
    players_unfiltered = players_unfiltered.sort_values(
        "date", kind="stable", na_position="last"
    ).reset_index(drop=True)
    players = players.sort_values("date", kind="stable", na_position="last").reset_index(
        drop=True
    )

    games.to_parquet(processed_dir / "games.parquet", index=False)
    players.to_parquet(processed_dir / "players.parquet", index=False)
    games_unfiltered.to_parquet(processed_dir / "games_unfiltered.parquet", index=False)
    players_unfiltered.to_parquet(
        processed_dir / "players_unfiltered.parquet", index=False
    )

    LOGGER.info(
        "Dropped %d games with missing/invalid results and %d games with fewer than "
        "five players on a side",
        dropped_missing,
        dropped_players,
    )
    return IngestResult(
        players,
        games,
        players_unfiltered,
        games_unfiltered,
        dropped_missing,
        dropped_players,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Oracle's Elixir match CSVs")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = ingest(args.raw_dir, args.processed_dir, args.config)
    counts = (
        result.games.groupby(["league", "year"], dropna=False)
        .size()
        .rename("games")
        .reset_index()
        .sort_values(["league", "year"], kind="stable")
    )
    print("Games per league per year after Tier 1 filtering:")
    print(counts.to_string(index=False))
    print(
        "Dropped games: "
        f"{result.dropped_missing_results} missing/invalid results, "
        f"{result.dropped_insufficient_players} fewer than five players per side"
    )


if __name__ == "__main__":
    main()
