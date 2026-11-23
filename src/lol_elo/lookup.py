"""Look up player identities and their current Tier 1 Elo history."""

from __future__ import annotations

import argparse
import re

import pandas as pd

from lol_elo.player_elo import run_player_elo
from lol_elo.team_elo import DEFAULT_MODEL_CONFIG_PATH, load_model_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Look up player identities")
    parser.add_argument("names", nargs="+", metavar="NAME")
    parser.add_argument("--games", default="data/processed/games.parquet")
    parser.add_argument("--players", default="data/processed/players.parquet")
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG_PATH))
    args = parser.parse_args()

    games = pd.read_parquet(args.games)
    players = pd.read_parquet(args.players)
    if "playername" not in players or "player_key" not in players:
        raise ValueError("Ingested players data must include playername and player_key")

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
        initial_ratings = build_initial_rookie_ratings(
            games, players, lower_history, load_fitted_priors()
        )
    result = run_player_elo(games, players, config, initial_ratings=initial_ratings)
    ratings = result.ratings.copy()
    ratings["effective_rating"] = ratings["rating"] + ratings["region"].map(
        result.region_offsets
    ).fillna(0.0)
    ratings = ratings.set_index("player_key")
    players = players.copy()
    players["_search_name"] = players["playername"].astype("string")
    players["_position"] = players["position"].astype("string").str.strip().str.lower()
    players["_date"] = pd.to_datetime(players["date"], errors="coerce")

    for query in args.names:
        query = query.strip()
        if not query:
            parser.error("NAME cannot be empty")
        matching_rows = players.loc[
            players["_search_name"].str.contains(
                re.escape(query), case=False, regex=True, na=False
            )
        ]
        print(f'\nMatches for "{query}":')
        if matching_rows.empty:
            print("No players matched.")
            continue

        player_keys = matching_rows["player_key"].dropna().astype(str).unique()
        for player_key in sorted(player_keys):
            history = players.loc[players["player_key"].astype("string").eq(player_key)]
            names = sorted(history["_search_name"].dropna().astype(str).unique())
            role_counts = (
                history.loc[history["_position"].ne("team")]
                .groupby("_position", dropna=True)
                .size()
                .sort_index()
            )
            role_summary = ", ".join(
                f"{role}: {count}" for role, count in role_counts.items()
            )
            teams = history[["teamname", "teamid"]].drop_duplicates().dropna(how="all")
            team_summary = ", ".join(
                f"{row.teamname} ({row.teamid})"
                if pd.notna(row.teamid)
                else str(row.teamname)
                for row in teams.itertuples(index=False)
                if pd.notna(row.teamname)
            )
            if player_key not in ratings.index:
                rating = float("nan")
                games_played = 0
                region = None
                effective_rating = float("nan")
            else:
                rating = float(ratings.loc[player_key, "rating"])
                games_played = int(ratings.loc[player_key, "games_played"])
                region = ratings.loc[player_key, "region"]
                effective_rating = float(ratings.loc[player_key, "effective_rating"])
            first_game = history["_date"].min()
            last_game = history["_date"].max()
            first_text = (
                first_game.strftime("%Y-%m-%d") if pd.notna(first_game) else "unknown"
            )
            last_text = (
                last_game.strftime("%Y-%m-%d") if pd.notna(last_game) else "unknown"
            )

            print(f"  player_key: {player_key}")
            print(f"  playernames: {', '.join(names)}")
            print(f"  rating: {rating:.2f}")
            print(f"  region: {region if pd.notna(region) else 'unknown'}")
            print(f"  effective_rating: {effective_rating:.2f}")
            print(f"  games_played: {games_played}")
            print(f"  roles: {role_summary or 'none'}")
            print(f"  teams: {team_summary or 'none'}")
            print(f"  first game: {first_text}")
            print(f"  last game: {last_text}")
