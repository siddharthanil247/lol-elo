"""Load Oracle's Elixir CSVs and build player and game tables."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter
from typing import Callable, NamedTuple

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
    "game",
    "playoffs",
    "blue_team",
    "blue_team_id",
    "red_team",
    "red_team_id",
    "winner",
    "winner_team_id",
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
    international_drops: pd.DataFrame


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
    return dict(
        zip(aliases["player_alias"], aliases["canonical_player_key"], strict=True)
    )


def _build_games(
    raw: pd.DataFrame, international_leagues: set[str]
) -> tuple[pd.DataFrame, int, int]:
    is_team = raw["position"].astype("string").str.lower().eq("team")
    team_rows = raw.loc[is_team].copy()
    team_rows["_side"] = team_rows["side"].astype("string").str.lower()
    team_rows["_result"] = pd.to_numeric(team_rows["result"], errors="coerce")
    team_rows["_winner"] = team_rows["_result"].eq(1)

    result_stats = team_rows.groupby("gameid", sort=False).agg(
        team_row_count=("result", "size"),
        result_count=("_result", "count"),
        winner_count=("_winner", "sum"),
    )
    team_side_counts = team_rows.groupby(["gameid", "_side"], sort=False).size()
    team_side_counts = team_side_counts.unstack("_side", fill_value=0).reindex(
        columns=["blue", "red"], fill_value=0
    )
    result_stats = result_stats.join(
        team_side_counts.rename(
            columns={"blue": "blue_row_count", "red": "red_row_count"}
        ),
        how="outer",
    )
    all_gameids = raw["gameid"].dropna().drop_duplicates()
    result_stats = result_stats.reindex(all_gameids, fill_value=0)
    valid_results = (
        result_stats["team_row_count"].eq(2)
        & result_stats["result_count"].eq(2)
        & result_stats["winner_count"].eq(1)
        & result_stats["blue_row_count"].eq(1)
        & result_stats["red_row_count"].eq(1)
    )
    dropped_missing_results = int((~valid_results).sum())

    player_rows = raw.loc[~is_team].assign(
        _side=raw.loc[~is_team, "side"].astype("string").str.lower()
    )
    player_counts = player_rows.groupby(["gameid", "_side"], sort=False).size()
    player_counts = player_counts.unstack("_side", fill_value=0).reindex(
        columns=["blue", "red"], fill_value=0
    )
    has_five_players = player_counts.ge(5).all(axis=1)
    eligible_for_player_count = valid_results.reindex(
        player_counts.index, fill_value=False
    )
    dropped_insufficient_players = int(
        (eligible_for_player_count & ~has_five_players).sum()
    )
    eligible_gameids = valid_results.index[valid_results].intersection(
        player_counts.index[has_five_players]
    )

    team_columns = ["gameid", "teamname", "teamid", "_result"]
    blue = team_rows.loc[team_rows["_side"].eq("blue"), team_columns].rename(
        columns={"teamname": "blue_team", "teamid": "blue_team_id"}
    )
    red = team_rows.loc[team_rows["_side"].eq("red"), team_columns].rename(
        columns={"teamname": "red_team", "teamid": "red_team_id"}
    )[["gameid", "red_team", "red_team_id"]]
    winners = team_rows.loc[
        team_rows["_winner"], ["gameid", "teamname", "teamid"]
    ].rename(columns={"teamname": "winner", "teamid": "winner_team_id"})
    metadata = raw.drop_duplicates("gameid", keep="first")[
        ["gameid", "game", "playoffs", "date", "league", "year"]
    ]
    games = (
        metadata.merge(blue, on="gameid", how="inner", sort=False)
        .merge(red, on="gameid", how="inner", sort=False)
        .merge(winners, on="gameid", how="inner", sort=False)
    )
    games = games.loc[games["gameid"].isin(eligible_gameids)].copy()
    games["is_international"] = games["league"].isin(international_leagues)
    games = games[GAME_COLUMNS]
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    return games, dropped_missing_results, dropped_insufficient_players


def _team_key(team_ids: pd.Series, team_names: pd.Series) -> pd.Series:
    ids = team_ids.astype("string").str.strip()
    names = team_names.astype("string").str.strip().str.lower()
    has_id = ids.notna() & ids.ne("")
    has_name = names.notna() & names.ne("")
    return ("teamid:" + ids).where(has_id, ("teamname:" + names).where(has_name))


def _filter_international_teams(
    tier1_games: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    domestic_names = {
        entry if isinstance(entry, str) else entry["name"]
        for entry in config["tier1_domestic"]
    }
    international_names = {
        entry if isinstance(entry, str) else entry["name"]
        for entry in config["international"]
    }
    domestic = tier1_games.loc[tier1_games["league"].isin(domestic_names)]
    international = tier1_games.loc[tier1_games["league"].isin(international_names)]
    dropped = international[["league", "year"]].copy()
    dropped["dropped_games"] = 0

    if (
        config.get("international_require_tier1_teams", False)
        and not international.empty
    ):
        domestic_blue = domestic[["year", "blue_team_id", "blue_team"]].rename(
            columns={"blue_team_id": "team_id", "blue_team": "team_name"}
        )
        domestic_red = domestic[["year", "red_team_id", "red_team"]].rename(
            columns={"red_team_id": "team_id", "red_team": "team_name"}
        )
        domestic_teams = pd.concat([domestic_blue, domestic_red], ignore_index=True)
        domestic_teams["team_key"] = _team_key(
            domestic_teams["team_id"], domestic_teams["team_name"]
        )
        domestic_teams = domestic_teams.dropna(subset=["team_key"]).drop_duplicates(
            ["year", "team_key"]
        )[["year", "team_key"]]

        blue = international[["gameid", "year", "blue_team_id", "blue_team"]].rename(
            columns={"blue_team_id": "team_id", "blue_team": "team_name"}
        )
        red = international[["gameid", "year", "red_team_id", "red_team"]].rename(
            columns={"red_team_id": "team_id", "red_team": "team_name"}
        )
        international_teams = pd.concat([blue, red], ignore_index=True)
        international_teams["team_key"] = _team_key(
            international_teams["team_id"], international_teams["team_name"]
        )
        checks = international_teams.merge(
            domestic_teams.assign(_played_tier1=True),
            on=["year", "team_key"],
            how="left",
            sort=False,
        )
        checks["_played_tier1"] = checks["_played_tier1"].fillna(False).astype(bool)
        game_eligibility = checks.groupby("gameid", sort=False)["_played_tier1"].all()
        kept_international_ids = game_eligibility.index[game_eligibility]
        dropped_ids = game_eligibility.index[~game_eligibility]
        dropped_mask = international["gameid"].isin(dropped_ids)
        dropped = international.loc[dropped_mask, ["league", "year"]].copy()
        dropped["dropped_games"] = 1
        drop_counts = dropped.groupby(["league", "year"], as_index=False)[
            "dropped_games"
        ].sum()
        all_event_years = international[["league", "year"]].drop_duplicates()
        dropped = all_event_years.merge(
            drop_counts, on=["league", "year"], how="left", sort=False
        )
        dropped["dropped_games"] = dropped["dropped_games"].fillna(0).astype(int)
        filtered = tier1_games.loc[
            ~tier1_games["gameid"].isin(international["gameid"])
            | tier1_games["gameid"].isin(kept_international_ids)
        ].copy()
        return filtered, dropped

    if not international.empty:
        dropped = (
            international[["league", "year"]]
            .drop_duplicates()
            .assign(dropped_games=0)
        )
    return tier1_games.copy(), dropped


def ingest(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    progress_callback: Callable[[str, float], None] | None = None,
) -> IngestResult:
    """Ingest raw match files, save tables, and return filtered and full tables."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    stage_start = perf_counter()
    raw = _load_raw_csvs(raw_dir)
    config = load_leagues(config_path)
    international_leagues = {
        entry if isinstance(entry, str) else entry["name"]
        for entry in config["international"]
    }
    if progress_callback:
        progress_callback("loading", perf_counter() - stage_start)

    stage_start = perf_counter()
    games_unfiltered, dropped_missing, dropped_players = _build_games(
        raw, international_leagues
    )
    valid_gameids = games_unfiltered["gameid"]
    players_unfiltered = raw.loc[
        (raw["position"].astype("string").str.lower() != "team")
        & raw["gameid"].isin(valid_gameids)
    ].copy()
    players_unfiltered["date"] = pd.to_datetime(
        players_unfiltered["date"], errors="coerce"
    )
    player_ids = players_unfiltered["playerid"].astype("string").str.strip()
    player_names = (
        players_unfiltered["playername"].astype("string").str.strip().str.lower()
    )
    id_keys = "playerid:" + player_ids
    name_keys = "name:" + player_names
    id_keys = id_keys.mask(player_ids.isna() | player_ids.eq(""))
    name_keys = name_keys.mask(player_names.isna() | player_names.eq(""))
    players_unfiltered["player_key"] = id_keys.fillna(name_keys)
    if progress_callback:
        progress_callback("building games", perf_counter() - stage_start)

    stage_start = perf_counter()
    aliases = _load_aliases(processed_dir)
    if aliases:
        source_keys = players_unfiltered["player_key"].astype(object)
        aliased_keys = source_keys.map(aliases)
        players_unfiltered["player_key"] = source_keys.where(
            aliased_keys.isna(), aliased_keys
        )
    games = filter_tier1_games(games_unfiltered, config_path)
    games, international_drops = _filter_international_teams(games, config)
    tier1_gameids = set(games["gameid"])
    players = players_unfiltered.loc[
        players_unfiltered["gameid"].isin(tier1_gameids)
    ].copy()
    if progress_callback:
        progress_callback("filtering", perf_counter() - stage_start)

    stage_start = perf_counter()
    games_unfiltered = games_unfiltered.sort_values(
        "date", kind="stable", na_position="last"
    ).reset_index(drop=True)
    games = games.sort_values("date", kind="stable", na_position="last").reset_index(
        drop=True
    )
    players_unfiltered = players_unfiltered.sort_values(
        "date", kind="stable", na_position="last"
    ).reset_index(drop=True)
    players = players.sort_values(
        "date", kind="stable", na_position="last"
    ).reset_index(drop=True)

    games.to_parquet(processed_dir / "games.parquet", index=False)
    players.to_parquet(processed_dir / "players.parquet", index=False)
    games_unfiltered.to_parquet(processed_dir / "games_unfiltered.parquet", index=False)
    players_unfiltered.to_parquet(
        processed_dir / "players_unfiltered.parquet", index=False
    )
    if progress_callback:
        progress_callback("saving", perf_counter() - stage_start)

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
        international_drops,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Oracle's Elixir match CSVs")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    def report_stage(stage: str, elapsed_seconds: float) -> None:
        print(f"{stage}: {elapsed_seconds:.2f}s", flush=True)

    result = ingest(
        args.raw_dir,
        args.processed_dir,
        args.config,
        progress_callback=report_stage,
    )
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
    print("International games dropped by event and year:")
    if result.international_drops.empty:
        print("(none)")
    else:
        print(result.international_drops.to_string(index=False))


if __name__ == "__main__":
    main()
