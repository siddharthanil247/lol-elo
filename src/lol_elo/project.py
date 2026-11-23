"""Roster-based season projections and walk-forward backtests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from lol_elo.evaluate import _series_win_probability
from lol_elo.leagues import load_leagues
from lol_elo.player_elo import (
    ROLES,
    _convert_player_region_rating,
    run_player_elo,
)
from lol_elo.priors import (
    build_debut_training_rows,
    build_initial_rookie_ratings,
    fit_rookie_priors,
    load_fitted_priors,
    load_or_build_lower_history,
    prior_for_player,
)
from lol_elo.team_elo import load_model_config, run_team_elo


def season_start(games: pd.DataFrame, season: int) -> pd.Timestamp:
    """Find the first game date labeled with this season; dates define ordering."""
    dates = pd.to_datetime(games["date"], errors="coerce")
    years = pd.to_numeric(games["year"], errors="coerce")
    season_dates = dates.loc[years.eq(season) & dates.notna()]
    if season_dates.empty:
        return pd.Timestamp(year=season, month=1, day=1)
    return season_dates.min()


def before_season(
    games: pd.DataFrame, players: pd.DataFrame, season: int
) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    """Return only games strictly before the season's first game date."""
    start = season_start(games, season)
    game_dates = pd.to_datetime(games["date"], errors="coerce")
    prior_games = games.loc[game_dates.lt(start)].copy()
    prior_ids = set(prior_games["gameid"].astype(str))
    prior_players = players.loc[players["gameid"].astype(str).isin(prior_ids)].copy()
    return start, prior_games, prior_players


def _norm(value: object) -> str:
    return str(value).strip().casefold()


def _org_key(team_id: object, team_name: object) -> str:
    if pd.notna(team_id) and str(team_id).strip():
        return f"teamid:{str(team_id).strip()}"
    return f"teamname:{_norm(team_name)}"


def _playoffs_mask(values: pd.Series) -> pd.Series:
    return (
        values.astype("string").str.strip().str.lower().isin({"1", "true", "yes", "y"})
    )


def _league_region(league: str, leagues: dict) -> str | None:
    for region, league_names in leagues.get("regions", {}).items():
        if league in league_names:
            return region
    return None


def _name_catalog(
    players: pd.DataFrame,
    aliases_path: str | Path = "data/processed/player_aliases.csv",
) -> tuple[dict[str, set[str]], dict[str, str]]:
    by_name: dict[str, set[str]] = {}
    for row in players[["playername", "player_key"]].dropna().itertuples(index=False):
        by_name.setdefault(_norm(row.playername), set()).add(str(row.player_key))
    direct_aliases: dict[str, str] = {}
    alias_file = Path(aliases_path)
    if alias_file.exists():
        alias_rows = pd.read_csv(alias_file, dtype="string").dropna(
            subset=["player_alias", "canonical_player_key"]
        )
        for row in alias_rows.itertuples(index=False):
            alias = _norm(row.player_alias)
            key = str(row.canonical_player_key)
            direct_aliases[alias] = key
            by_name.setdefault(alias, set()).add(key)
    return by_name, direct_aliases


def _project_name_catalog(
    tier1_players: pd.DataFrame, all_players: pd.DataFrame
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, set[str]]]:
    tier1_names, aliases = _name_catalog(tier1_players)
    lower_names, lower_aliases = _name_catalog(all_players)
    # Match the lookup command against Tier 1 identities first. Lower-league
    # names are a fallback for rookies and cannot make a Tier 1 name ambiguous.
    return tier1_names, {**lower_aliases, **aliases}, lower_names


def resolve_player_name(
    name: str,
    by_name: dict[str, set[str]],
    direct_aliases: dict[str, str] | None = None,
) -> tuple[str | None, str]:
    """Resolve with lookup's case-insensitive partial matching semantics."""
    query = _norm(name)
    direct_aliases = direct_aliases or {}
    if query in direct_aliases:
        return direct_aliases[query], "alias"
    keys = {
        key
        for candidate, values in by_name.items()
        if query in candidate
        for key in values
    }
    if len(keys) == 1:
        return next(iter(keys)), "matched"
    if len(keys) > 1:
        return None, "ambiguous"
    return f"name:{query}", "unknown"


def _lineup_for_team_rows(
    rows: pd.DataFrame, by_name: dict[str, set[str]], aliases: dict[str, str]
) -> dict[str, Any]:
    starters = {}
    for role in ROLES:
        role_rows = rows.loc[
            rows["position"].astype("string").str.strip().str.lower().eq(role)
        ].sort_values(["date", "gameid"], kind="stable")
        if role_rows.empty:
            continue
        counts = role_rows["player_key"].astype(str).value_counts()
        top_count = int(counts.max())
        top_keys = set(counts.index[counts.eq(top_count)])
        selected = role_rows.loc[
            role_rows["player_key"].astype(str).isin(top_keys)
        ].iloc[-1]
        starters[role] = {
            "name": str(selected["playername"]),
            "player_key": str(selected["player_key"]),
        }
    return starters


def generate_roster_template(
    games: pd.DataFrame,
    players: pd.DataFrame,
    season: int,
    leagues_config: dict | None = None,
) -> dict:
    """Build a YAML-ready roster from each team's last five Tier 1 games."""
    leagues_config = leagues_config or load_leagues()
    domestic = {
        entry if isinstance(entry, str) else entry["name"]
        for entry in leagues_config["tier1_domestic"]
    }
    domestic_games = games.loc[games["league"].isin(domestic)].copy()
    domestic_games["date"] = pd.to_datetime(domestic_games["date"], errors="coerce")
    team_sides = []
    for side in ("blue", "red"):
        team_sides.append(
            domestic_games[
                ["gameid", "date", "league", "year", f"{side}_team", f"{side}_team_id"]
            ].rename(columns={f"{side}_team": "teamname", f"{side}_team_id": "teamid"})
        )
    appearances = pd.concat(team_sides, ignore_index=True).drop_duplicates(
        ["gameid", "teamid", "teamname"]
    )
    appearances["org_key"] = appearances.apply(
        lambda row: _org_key(row["teamid"], row["teamname"]), axis=1
    )
    latest_date = pd.to_datetime(games["date"], errors="coerce").max()
    appearances = appearances.loc[
        appearances["date"].ge(latest_date - pd.Timedelta(days=365))
    ]
    player_rows = players.loc[
        players["position"].astype("string").str.lower().ne("team")
    ].copy()
    player_rows["date"] = pd.to_datetime(player_rows["date"], errors="coerce")
    teams = []
    for org_key, org_games in appearances.groupby("org_key", sort=False):
        org_games = org_games.sort_values(["date", "gameid"], kind="stable")
        last_five_ids = set(org_games.tail(5)["gameid"].astype(str))
        rows = player_rows.loc[
            player_rows["gameid"].astype(str).isin(last_five_ids)
        ].copy()
        rows["org_key"] = rows.apply(
            lambda row: _org_key(row.get("teamid"), row.get("teamname")), axis=1
        )
        rows = rows.loc[rows["org_key"].eq(org_key)]
        latest = org_games.iloc[-1]
        starters = _lineup_for_team_rows(rows, {}, {})
        if set(starters) != set(ROLES):
            continue
        teams.append(
            {
                "name": str(latest["teamname"]),
                "teamid": str(latest["teamid"]) if pd.notna(latest["teamid"]) else None,
                "league": str(latest["league"]),
                "starters": starters,
            }
        )
    teams.sort(key=lambda team: (team["league"], team["name"].casefold()))
    return {"season": int(season), "teams": teams}


def _last_org_lineups(
    games: pd.DataFrame,
    players: pd.DataFrame,
    before: pd.Timestamp,
) -> dict[str, frozenset[str]]:
    dates = pd.to_datetime(games["date"], errors="coerce")
    prior_games = games.loc[dates.lt(before)]
    ids = set(prior_games["gameid"].astype(str))
    player_rows = players.loc[
        players["gameid"].astype(str).isin(ids)
        & players["position"].astype("string").str.lower().ne("team")
    ].copy()
    player_rows["date"] = pd.to_datetime(player_rows["date"], errors="coerce")
    player_rows["org_key"] = player_rows.apply(
        lambda row: _org_key(row.get("teamid"), row.get("teamname")), axis=1
    )
    lineups = {}
    for org_key, rows in player_rows.groupby("org_key", sort=False):
        ordered = rows.sort_values(["date", "gameid"], kind="stable")
        last_game_id = ordered.iloc[-1]["gameid"]
        final_game = ordered.loc[ordered["gameid"].eq(last_game_id)]
        lineup = frozenset(final_game["player_key"].dropna().astype(str))
        if len(lineup) == 5:
            lineups[str(org_key)] = lineup
    return lineups


def _fill_team_ids(rosters: list[dict], games: pd.DataFrame) -> list[dict]:
    """Resolve omitted roster team IDs from the latest exact league/name match."""
    sides = []
    for side in ("blue", "red"):
        sides.append(
            games[["date", "league", f"{side}_team", f"{side}_team_id"]].rename(
                columns={f"{side}_team": "teamname", f"{side}_team_id": "teamid"}
            )
        )
    known = pd.concat(sides, ignore_index=True)
    known["date"] = pd.to_datetime(known["date"], errors="coerce")
    known["_team_norm"] = known["teamname"].map(_norm)
    resolved = []
    for roster in rosters:
        item = dict(roster)
        if not item.get("teamid"):
            matches = known.loc[
                known["league"].eq(item.get("league"))
                & known["_team_norm"].eq(_norm(item.get("name", "")))
                & known["teamid"].notna()
            ].sort_values("date", kind="stable")
            if not matches.empty:
                item["teamid"] = str(matches.iloc[-1]["teamid"])
        resolved.append(item)
    return resolved


def project_team_ratings(
    rosters: list[dict],
    ratings: pd.DataFrame,
    region_offsets: dict[str, float],
    org_lineups: dict[str, frozenset[str]],
    model_config: dict,
    leagues_config: dict,
    lower_history: pd.DataFrame,
    fitted_priors: dict,
    as_of: object,
    name_catalog: tuple[dict[str, set[str]], dict[str, str]]
    | tuple[dict[str, set[str]], dict[str, str], dict[str, set[str]]],
) -> tuple[pd.DataFrame, list[str]]:
    """Calculate frozen projected strength and report name-resolution warnings."""
    by_name, aliases = name_catalog[:2]
    lower_names = name_catalog[2] if len(name_catalog) > 2 else {}
    ratings_by_key = (
        ratings.set_index("player_key").to_dict("index") if not ratings.empty else {}
    )
    role_weights = {
        role: float(model_config.get("role_weights", {}).get(role, 1.0))
        for role in ROLES
    }
    weight_total = sum(role_weights.values())
    if weight_total <= 0:
        raise ValueError("The five role weights must sum to a positive number")
    warnings = []
    output = []
    for roster in rosters:
        team_name = str(roster["name"])
        league = str(roster["league"])
        org_key = _org_key(roster.get("teamid"), team_name)
        region = roster.get("region") or _league_region(league, leagues_config)
        if region is None:
            warnings.append(f"{team_name}: no configured region for {league}")
        players_by_role = roster.get("starters", {})
        if set(players_by_role) != set(ROLES):
            warnings.append(
                f"{team_name}: starters must specify exactly {', '.join(ROLES)}"
            )
            continue
        rating_sum = 0.0
        resolved_keys = {}
        for role in ROLES:
            starter = players_by_role[role]
            if isinstance(starter, dict):
                name = str(starter.get("name", ""))
                supplied_key = str(starter.get("player_key", ""))
            else:
                name = str(starter)
                supplied_key = ""
            exact_candidates = by_name.get(_norm(name), set())
            if supplied_key and supplied_key in exact_candidates:
                key, status = supplied_key, "matched"
            else:
                key, status = resolve_player_name(name, by_name, aliases)
                if status == "unknown":
                    key, lower_status = resolve_player_name(name, lower_names)
                    if lower_status != "unknown":
                        status = (
                            "lower_matched"
                            if lower_status == "matched"
                            else lower_status
                        )
            if status == "ambiguous":
                warnings.append(
                    f"{team_name} {role} {name!r}: ambiguous name; using rookie base"
                )
                key = f"name:{_norm(name)}"
            elif status == "unknown":
                warnings.append(
                    f"{team_name} {role} {name!r}: unknown player; "
                    "checking lower leagues"
                )
            resolved_keys[role] = key
            if key in ratings_by_key:
                player_rating = float(ratings_by_key[key]["rating"])
                old_region = ratings_by_key[key].get("region")
                if region and old_region and old_region != region:
                    player_rating = _convert_player_region_rating(
                        player_rating, str(old_region), str(region), region_offsets
                    )
            else:
                player_rating, lower_league = prior_for_player(
                    key, as_of, lower_history, fitted_priors
                )
                if lower_league:
                    warnings.append(
                        f"{team_name} {role} {name!r}: no Tier 1 history; "
                        f"using {lower_league} prior"
                    )
                elif status in {"matched", "lower_matched"}:
                    warnings.append(
                        f"{team_name} {role} {name!r}: no Tier 1 or lower-league "
                        "history; using rookie base"
                    )
            rating_sum += role_weights[role] * player_rating
        player_average = rating_sum / weight_total
        continuity = len(
            set(resolved_keys.values()) & set(org_lineups.get(org_key, ()))
        )
        continuity_term = (
            float(model_config.get("continuity_bonus", 0)) * continuity / 5
        )
        region_offset = float(region_offsets.get(region, 0.0)) if region else 0.0
        output.append(
            {
                "team": team_name,
                "teamid": roster.get("teamid"),
                "league": league,
                "region": region,
                "player_rating": player_average,
                "region_offset": region_offset,
                "continuity_starters": continuity,
                "continuity_term": continuity_term,
                "projected_rating": player_average + region_offset + continuity_term,
                "starter_keys": resolved_keys,
            }
        )
    return pd.DataFrame(output), warnings


def _regular_format(league: str, leagues_config: dict) -> tuple[int, int]:
    root = leagues_config.get("regular_season", {})
    default = root.get("default", {})
    setting = root.get("leagues", {}).get(league, {})
    round_robin = str(setting.get("round_robin", default.get("round_robin", "double")))
    repeats = {"single": 1, "double": 2, "triple": 3}.get(round_robin, 2)
    best_of = max(1, int(setting.get("best_of", default.get("best_of", 1))))
    return repeats, best_of


def _game_probability(rating_a: float, rating_b: float, blue_advantage: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(rating_a + blue_advantage - rating_b) / 400.0))


def _simulate_league(
    frame: pd.DataFrame,
    leagues_config: dict,
    model_config: dict,
    runs: int = 10_000,
    seed: int = 2027,
) -> pd.DataFrame:
    teams = frame.reset_index(drop=True)
    count = len(teams)
    if count < 2:
        return teams.assign(
            expected_game_win_rate=0.5,
            expected_series_win_rate=0.5,
            expected_win_rate=0.5,
            average_standing=1.0,
            probability_first=1.0,
            probability_top2=1.0,
            probability_top4=1.0,
        )
    repeats, best_of = _regular_format(str(teams.iloc[0]["league"]), leagues_config)
    rng = np.random.default_rng(seed)
    match_wins = np.zeros((runs, count), dtype=np.int16)
    game_wins = np.zeros((runs, count), dtype=np.int32)
    game_counts = np.zeros((runs, count), dtype=np.int32)
    expected_game = np.zeros(count, dtype=float)
    expected_series = np.zeros(count, dtype=float)
    needed = best_of // 2 + 1
    blue_advantage = float(model_config.get("blue_side_advantage", 0))
    ratings = teams["projected_rating"].to_numpy(float)
    for team_a in range(count):
        for team_b in range(team_a + 1, count):
            p_blue = _game_probability(ratings[team_a], ratings[team_b], blue_advantage)
            p_red = _game_probability(ratings[team_a], ratings[team_b], -blue_advantage)
            expected_game[team_a] += (p_blue + p_red) * 0.5 * repeats
            expected_game[team_b] += (2 - p_blue - p_red) * 0.5 * repeats
            series_p = _series_win_probability(
                [p_blue if game_idx % 2 == 0 else p_red for game_idx in range(best_of)],
                best_of,
            )
            reverse_series_p = _series_win_probability(
                [p_red if game_idx % 2 == 0 else p_blue for game_idx in range(best_of)],
                best_of,
            )
            for repeat in range(repeats):
                first_team_a_blue = repeat % 2 == 0
                team_a_series_probability = (
                    series_p if first_team_a_blue else reverse_series_p
                )
                expected_series[team_a] += team_a_series_probability
                expected_series[team_b] += 1 - team_a_series_probability
                probs = np.array(
                    [
                        p_blue if (game_idx % 2 == 0) == first_team_a_blue else p_red
                        for game_idx in range(best_of)
                    ]
                )
                outcomes = rng.random((runs, best_of)) < probs[None, :]
                cumulative = np.cumsum(outcomes, axis=1)
                a_wins_series = cumulative[:, -1] >= needed
                a_reaches = cumulative >= needed
                b_reaches = (np.cumsum(~outcomes, axis=1)) >= needed
                completed_at = np.full(runs, best_of, dtype=np.int16)
                any_done = a_reaches.any(axis=1) | b_reaches.any(axis=1)
                completed_at[any_done] = (
                    np.argmax(a_reaches[any_done] | b_reaches[any_done], axis=1) + 1
                )
                game_counts[:, team_a] += completed_at
                game_counts[:, team_b] += completed_at
                wins_a = np.minimum(
                    cumulative[np.arange(runs), completed_at - 1], needed
                )
                wins_b = completed_at - wins_a
                game_wins[:, team_a] += wins_a
                game_wins[:, team_b] += wins_b
                match_wins[:, team_a] += a_wins_series
                match_wins[:, team_b] += ~a_wins_series
    expected_series /= (count - 1) * repeats
    expected_game = game_wins.mean(axis=0) / game_counts.mean(axis=0)
    game_diff = game_wins - (game_counts - game_wins)
    tie_noise = rng.random((runs, count))
    order = np.lexsort((tie_noise, game_diff, match_wins), axis=1)[:, ::-1]
    ranks = np.empty_like(order)
    ranks[np.arange(runs)[:, None], order] = np.arange(1, count + 1)[None, :]
    result = teams.copy()
    result["expected_game_win_rate"] = expected_game
    result["expected_series_win_rate"] = expected_series
    result["expected_win_rate"] = expected_series if best_of > 1 else expected_game
    result["average_standing"] = ranks.mean(axis=0)
    result["probability_first"] = (ranks == 1).mean(axis=0)
    result["probability_top2"] = (ranks <= 2).mean(axis=0)
    result["probability_top4"] = (ranks <= 4).mean(axis=0)
    return result


def project_season(
    season: int,
    rosters_path: str | Path,
    games: pd.DataFrame,
    players: pd.DataFrame,
    lower_history: pd.DataFrame,
    fitted_priors: dict,
    model_config: dict,
    leagues_config: dict,
    runs: int = 10_000,
    seed: int = 2027,
) -> tuple[pd.DataFrame, pd.Timestamp, list[str]]:
    """Freeze pre-season state, project supplied rosters, simulate standings."""
    roster_data = yaml.safe_load(Path(rosters_path).read_text(encoding="utf-8")) or {}
    roster_list = roster_data.get("teams", [])
    start, prior_games, prior_players = before_season(games, players, season)
    roster_list = _fill_team_ids(roster_list, prior_games)
    initial = build_initial_rookie_ratings(
        prior_games, prior_players, lower_history, fitted_priors
    )
    frozen = run_player_elo(
        prior_games,
        prior_players,
        model_config,
        initial_ratings=initial,
        leagues_config=leagues_config,
    )
    lineups = _last_org_lineups(prior_games, prior_players, start)
    catalog = _project_name_catalog(
        players, pd.read_parquet("data/processed/players_unfiltered.parquet")
    )
    projections, warnings = project_team_ratings(
        roster_list,
        frozen.ratings,
        frozen.region_offsets,
        lineups,
        model_config,
        leagues_config,
        lower_history,
        fitted_priors,
        start,
        catalog,
    )
    simulations = []
    for league, frame in projections.groupby("league", sort=True):
        simulations.append(
            _simulate_league(
                frame,
                leagues_config,
                model_config,
                runs=runs,
                seed=seed + sum(ord(char) for char in str(league)),
            )
        )
    return (
        pd.concat(simulations, ignore_index=True) if simulations else projections,
        start,
        warnings,
    )


def _most_common_roster(
    team_rows: pd.DataFrame,
    n_games: int = 3,
    most_recent: bool = False,
) -> dict[str, str]:
    ordered_games = team_rows.sort_values(
        ["date", "gameid"], kind="stable"
    ).drop_duplicates("gameid", keep="first")
    chosen = ordered_games.tail(n_games) if most_recent else ordered_games.head(n_games)
    selected_games = set(chosen["gameid"].astype(str))
    rows = team_rows.loc[team_rows["gameid"].astype(str).isin(selected_games)]
    return _lineup_for_team_rows(rows, {}, {})


def _season_rosters_from_games(
    games: pd.DataFrame,
    players: pd.DataFrame,
    season: int,
    domestic: set[str],
    most_recent: bool = False,
) -> list[dict]:
    season_games = games.loc[
        pd.to_numeric(games["year"], errors="coerce").eq(season)
        & games["league"].isin(domestic)
        & ~_playoffs_mask(games["playoffs"])
    ]
    ids = set(season_games["gameid"].astype(str))
    rows = players.loc[
        players["gameid"].astype(str).isin(ids)
        & players["position"].astype("string").str.lower().ne("team")
    ].copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows["org_key"] = rows.apply(
        lambda row: _org_key(row.get("teamid"), row.get("teamname")), axis=1
    )
    result = []
    for (league, org_key), group in rows.groupby(
        ["league", "org_key"], dropna=False, sort=False
    ):
        group = group.sort_values(["date", "gameid"], kind="stable")
        starters = _most_common_roster(group, 5 if most_recent else 3, most_recent)
        if set(starters) != set(ROLES):
            continue
        latest = group.iloc[-1]
        result.append(
            {
                "name": str(latest["teamname"]),
                "teamid": None if pd.isna(latest["teamid"]) else str(latest["teamid"]),
                "league": str(league),
                "starters": starters,
            }
        )
    return result


def _starter_identity(starter: object) -> str:
    if isinstance(starter, dict):
        return str(starter.get("player_key", starter.get("name", "")))
    return str(starter)


def _roster_change_count(previous: dict | None, current: dict) -> int:
    """Count replaced players, independent of role assignments."""
    if previous is None:
        return 5
    old_players = {_starter_identity(value) for value in previous.values()}
    new_players = {_starter_identity(value) for value in current.values()}
    return 5 - len(old_players & new_players)


def _format_lineup(lineup: dict | None) -> str:
    if not lineup:
        return "<no prior lineup>"
    formatted = []
    for role, value in lineup.items():
        name = (
            value.get("name", value.get("player_key", "?"))
            if isinstance(value, dict)
            else value
        )
        formatted.append(f"{role}={name}")
    return ", ".join(formatted)


def _first_three_substitutions(
    games: pd.DataFrame,
    players: pd.DataFrame,
    season: int,
    team: dict,
) -> str:
    selected_games = games.loc[
        pd.to_numeric(games["year"], errors="coerce").eq(season)
        & games["league"].eq(team["league"])
        & ~_playoffs_mask(games["playoffs"])
    ].copy()
    selected_games["date"] = pd.to_datetime(selected_games["date"], errors="coerce")
    if team.get("teamid"):
        matches = selected_games["blue_team_id"].astype("string").eq(
            str(team["teamid"])
        ) | selected_games["red_team_id"].astype("string").eq(str(team["teamid"]))
    else:
        matches = selected_games["blue_team"].map(_norm).eq(
            _norm(team["name"])
        ) | selected_games["red_team"].map(_norm).eq(_norm(team["name"]))
    first_ids = set(
        selected_games.loc[matches]
        .sort_values(["date", "gameid"], kind="stable")
        .head(3)["gameid"]
        .astype(str)
    )
    if not first_ids:
        return "none"
    team_rows = players.loc[
        players["gameid"].astype(str).isin(first_ids)
        & players["position"].astype("string").str.lower().ne("team")
    ]
    if team.get("teamid"):
        team_rows = team_rows.loc[
            team_rows["teamid"].astype("string").eq(str(team["teamid"]))
        ]
    else:
        team_rows = team_rows.loc[
            team_rows["teamname"].map(_norm).eq(_norm(team["name"]))
        ]
    changes = []
    for role, role_rows in team_rows.groupby("position", sort=False):
        counts = role_rows["player_key"].astype(str).value_counts()
        if len(counts) > 1:
            names = (
                role_rows.drop_duplicates("player_key")
                .set_index("player_key")["playername"]
                .to_dict()
            )
            changes.append(
                f"{role}: "
                + ", ".join(
                    f"{names.get(key, key)} ({count})" for key, count in counts.items()
                )
            )
    return "; ".join(changes) if changes else "none"


def print_roster_change_sample(
    games: pd.DataFrame,
    players: pd.DataFrame,
    seasons: range = range(2020, 2027),
    sample_size: int = 15,
    seed: int = 2027,
) -> None:
    """Print random roster pairs and substitutions visible in the first three games."""
    domestic = {
        entry if isinstance(entry, str) else entry["name"]
        for entry in load_leagues()["tier1_domestic"]
    }
    candidates = []
    for season in seasons:
        previous = {
            _org_key(team.get("teamid"), team["name"]): team
            for team in _season_rosters_from_games(
                games, players, season - 1, domestic, most_recent=True
            )
        }
        current = _season_rosters_from_games(games, players, season, domestic)
        for team in current:
            old = previous.get(_org_key(team.get("teamid"), team["name"]))
            candidates.append(
                {
                    "season": season,
                    "league": team["league"],
                    "team": team,
                    "previous": old["starters"] if old else None,
                    "changes": _roster_change_count(
                        old["starters"] if old else None, team["starters"]
                    ),
                }
            )
    sample = pd.DataFrame(candidates).sample(
        min(sample_size, len(candidates)), random_state=seed
    )
    print(f"Random roster-change sanity sample (n={len(sample)}, seed={seed}):")
    for row in sample.itertuples(index=False):
        print(
            f"{row.season} {row.league} {row.team['name']}: "
            f"{row.changes} changed players; first-three substitutions: "
            f"{_first_three_substitutions(games, players, row.season, row.team)}\n"
            f"  previous: {_format_lineup(row.previous)}\n"
            f"  new:      {_format_lineup(row.team['starters'])}"
        )


def _team_actual_rates(games: pd.DataFrame, season: int, league: str) -> pd.DataFrame:
    regular = games.loc[
        pd.to_numeric(games["year"], errors="coerce").eq(season)
        & games["league"].eq(league)
        & ~_playoffs_mask(games["playoffs"])
    ]
    blue = regular[["blue_team_id", "blue_team", "winner_team_id", "winner"]].copy()
    blue["teamkey"] = blue.apply(
        lambda row: _org_key(row["blue_team_id"], row["blue_team"]), axis=1
    )
    blue["win"] = (
        (
            blue["winner_team_id"]
            .astype("string")
            .eq(blue["blue_team_id"].astype("string"))
            | blue["winner"].astype("string").eq(blue["blue_team"].astype("string"))
        )
        .fillna(False)
        .astype(int)
    )
    red = regular[["red_team_id", "red_team", "winner_team_id", "winner"]].copy()
    red["teamkey"] = red.apply(
        lambda row: _org_key(row["red_team_id"], row["red_team"]), axis=1
    )
    red["win"] = (
        (
            red["winner_team_id"]
            .astype("string")
            .eq(red["red_team_id"].astype("string"))
            | red["winner"].astype("string").eq(red["red_team"].astype("string"))
        )
        .fillna(False)
        .astype(int)
    )
    combined = pd.concat(
        [
            blue.rename(columns={"blue_team_id": "teamid", "blue_team": "teamname"}),
            red.rename(columns={"red_team_id": "teamid", "red_team": "teamname"}),
        ],
        ignore_index=True,
    )
    return combined.groupby("teamkey", as_index=False).agg(
        actual_win_rate=("win", "mean"),
        games=("win", "size"),
        teamid=("teamid", "last"),
        teamname=("teamname", "last"),
    )


def _baseline_rates(
    teams: pd.DataFrame,
    prior_games: pd.DataFrame,
    games: pd.DataFrame,
    season: int,
    league: str,
    model_config: dict,
) -> tuple[dict[str, float], dict[str, float]]:
    previous = _team_actual_rates(games, season - 1, league).set_index("teamkey")
    last_rates = previous["actual_win_rate"].astype(float).to_dict()
    elo = run_team_elo(prior_games, model_config)
    elo_by_key = elo.ratings.set_index("team_key")["rating"].to_dict()
    blue_advantage = float(model_config.get("blue_side_advantage", 0))
    team_elo_rates = {}
    for row in teams.itertuples(index=False):
        key = _org_key(row.teamid, row.team)
        rating = float(elo_by_key.get(key, model_config.get("starting_rating", 1500)))
        opponents = teams.loc[teams["team"].ne(row.team)]
        probs = []
        for opponent in opponents.itertuples(index=False):
            opponent_key = _org_key(opponent.teamid, opponent.team)
            opponent_rating = float(
                elo_by_key.get(opponent_key, model_config.get("starting_rating", 1500))
            )
            blue_p = _game_probability(rating, opponent_rating, blue_advantage)
            red_p = _game_probability(rating, opponent_rating, -blue_advantage)
            probs.extend([blue_p, red_p])
        team_elo_rates[key] = float(np.mean(probs)) if probs else 0.5
    return last_rates, team_elo_rates


def _walk_forward_fitted_priors(
    games: pd.DataFrame,
    players: pd.DataFrame,
    lower_history: pd.DataFrame,
    model_config: dict,
    cutoff: pd.Timestamp,
) -> dict:
    dates = pd.to_datetime(games["date"], errors="coerce")
    prior_games = games.loc[dates.lt(cutoff)].copy()
    prior_ids = set(prior_games["gameid"].astype(str))
    prior_players = players.loc[players["gameid"].astype(str).isin(prior_ids)].copy()
    flat = run_player_elo(
        prior_games,
        prior_players,
        {**model_config, "use_rookie_priors": False},
        record_history=True,
    )
    training = build_debut_training_rows(
        prior_players, flat.rating_history, lower_history
    )
    return fit_rookie_priors(training, cutoff_year=9999)


def _spearman(actual: pd.Series, predicted: pd.Series) -> float:
    if len(actual) < 2 or actual.nunique() < 2 or predicted.nunique() < 2:
        return float("nan")
    return float(actual.rank(method="average").corr(predicted.rank(method="average")))


def run_backtest(
    games: pd.DataFrame,
    players: pd.DataFrame,
    all_games: pd.DataFrame,
    all_players: pd.DataFrame,
    model_config: dict,
    leagues_config: dict,
    fitted_priors: dict,
    seasons: range = range(2020, 2027),
    lower_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Backtest roster strength against actual regular-season game win rates."""
    if lower_history is None:
        lower_history = load_or_build_lower_history(
            all_games, all_players, model_config
        )
    domestic = {
        entry if isinstance(entry, str) else entry["name"]
        for entry in leagues_config["tier1_domestic"]
    }
    records = []
    for season in seasons:
        start, prior_games, prior_players = before_season(games, players, season)
        season_rosters = _season_rosters_from_games(games, players, season, domestic)
        walkforward_fitted = _walk_forward_fitted_priors(
            games, players, lower_history, model_config, start
        )
        initial = build_initial_rookie_ratings(
            prior_games, prior_players, lower_history, walkforward_fitted
        )
        frozen = run_player_elo(
            prior_games,
            prior_players,
            model_config,
            initial_ratings=initial,
            leagues_config=leagues_config,
        )
        lineups = _last_org_lineups(prior_games, prior_players, start)
        catalog = _project_name_catalog(players, all_players)
        projections, _ = project_team_ratings(
            season_rosters,
            frozen.ratings,
            frozen.region_offsets,
            lineups,
            model_config,
            leagues_config,
            lower_history,
            walkforward_fitted,
            start,
            catalog,
        )
        for league, projected in projections.groupby("league", sort=True):
            actual = _team_actual_rates(games, season, league)
            if actual.empty:
                continue
            projected = projected.copy()
            projected["teamkey"] = projected.apply(
                lambda row: _org_key(row["teamid"], row["team"]), axis=1
            )
            cohort = projected.merge(
                actual, on="teamkey", how="inner", suffixes=("", "_actual")
            )
            if cohort.empty:
                continue
            prior_rate, team_elo_rate = _baseline_rates(
                projected.rename(columns={"name": "team"})
                if "name" in projected
                else projected,
                prior_games,
                games,
                season,
                str(league),
                model_config,
            )
            cohort["player_projection"] = (
                cohort["expected_game_win_rate"]
                if "expected_game_win_rate" in cohort
                else np.nan
            )
            # Backtests score projected ratings as a balanced-side game-win rate.
            cohort["player_projection"] = cohort["projected_rating"].map(
                lambda _: np.nan
            )
            projected_ratings = projected.set_index("teamkey")["projected_rating"]
            for idx, row in cohort.iterrows():
                team_rating = float(projected_ratings.loc[row["teamkey"]])
                opponent_ratings = projected_ratings.drop(row["teamkey"])
                cohort.loc[idx, "player_projection"] = np.mean(
                    [
                        (
                            _game_probability(
                                team_rating,
                                float(opp),
                                float(model_config.get("blue_side_advantage", 0)),
                            )
                            + _game_probability(
                                team_rating,
                                float(opp),
                                -float(model_config.get("blue_side_advantage", 0)),
                            )
                        )
                        / 2
                        for opp in opponent_ratings
                    ]
                )
            cohort["last_season_baseline"] = (
                cohort["teamkey"].map(prior_rate).fillna(0.5)
            )
            cohort["team_elo_baseline"] = (
                cohort["teamkey"].map(team_elo_rate).fillna(0.5)
            )
            previous_rosters = _season_rosters_from_games(
                games, players, season - 1, domestic, most_recent=True
            )
            previous_by_key = {
                _org_key(item.get("teamid"), item["name"]): item["starters"]
                for item in previous_rosters
            }
            cohort["roster_changes"] = cohort.apply(
                lambda row: _roster_change_count(
                    previous_by_key.get(row["teamkey"]),
                    row["starter_keys"],
                ),
                axis=1,
            )
            for row in cohort.itertuples(index=False):
                records.append(
                    {
                        "season": season,
                        "league": league,
                        "team": row.team,
                        "actual_win_rate": row.actual_win_rate,
                        "projected_win_rate": row.player_projection,
                        "last_season_win_rate": row.last_season_baseline,
                        "team_elo_win_rate": row.team_elo_baseline,
                        "roster_changes": row.roster_changes,
                    }
                )
    predictions = pd.DataFrame(records)
    summaries = []
    methods = {
        "player projection": "projected_win_rate",
        "last season": "last_season_win_rate",
        "team Elo": "team_elo_win_rate",
    }
    subsets = [("all teams", predictions)]
    subsets.append(
        ("3+ roster changes", predictions.loc[predictions["roster_changes"].ge(3)])
    )
    for label, subset in subsets:
        for league in ["overall", *sorted(subset["league"].dropna().unique())]:
            frame = (
                subset
                if league == "overall"
                else subset.loc[subset["league"].eq(league)]
            )
            for method, column in methods.items():
                summaries.append(
                    {
                        "cohort": label,
                        "league": league,
                        "method": method,
                        "teams": len(frame),
                        "spearman": _spearman(frame["actual_win_rate"], frame[column])
                        if not frame.empty
                        else np.nan,
                        "mae": float(
                            (frame["actual_win_rate"] - frame[column]).abs().mean()
                        )
                        if not frame.empty
                        else np.nan,
                    }
                )
    return pd.DataFrame(summaries)


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet("data/processed/games.parquet"),
        pd.read_parquet("data/processed/players.parquet"),
        pd.read_parquet("data/processed/games_unfiltered.parquet"),
        pd.read_parquet("data/processed/players_unfiltered.parquet"),
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [
            f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_template(season: int, output: Path) -> None:
    games, players, _, _ = _load_inputs()
    data = generate_roster_template(games, players, season)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"Wrote {len(data['teams'])} teams to {output}")


def main_project(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Project a League of Legends season")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--rosters", required=True)
    parser.add_argument("--games", default="data/processed/games.parquet")
    parser.add_argument("--players", default="data/processed/players.parquet")
    parser.add_argument("--runs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args(argv)
    games = pd.read_parquet(args.games)
    players = pd.read_parquet(args.players)
    all_games = pd.read_parquet("data/processed/games_unfiltered.parquet")
    all_players = pd.read_parquet("data/processed/players_unfiltered.parquet")
    config = load_model_config()
    leagues = load_leagues()
    lower = load_or_build_lower_history(all_games, all_players, config)
    results, start, warnings = project_season(
        args.season,
        args.rosters,
        games,
        players,
        lower,
        load_fitted_priors(),
        config,
        leagues,
        runs=args.runs,
        seed=args.seed,
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Frozen ratings before {start:%Y-%m-%d}")
    display = results.drop(columns=["starter_keys"], errors="ignore")
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    output = Path(f"data/processed/roster_projection_{args.season}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    display.to_csv(output, index=False)
    print(f"Saved projection to {output}")


def main_backtest(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backtest roster projections")
    parser.add_argument("--notes", default="notes/backtest.md")
    parser.add_argument("--sanity-sample", type=int, default=15)
    args = parser.parse_args(argv)
    games, players, all_games, all_players = _load_inputs()
    if args.sanity_sample > 0:
        print_roster_change_sample(games, players, sample_size=args.sanity_sample)
    model = load_model_config()
    leagues = load_leagues()
    fitted = load_fitted_priors()
    results = run_backtest(
        games, players, all_games, all_players, model, leagues, fitted
    )
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    summary_path = Path("data/processed/backtest_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(summary_path, index=False)
    output = Path(args.notes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Roster projection backtest\n\n"
        "For each season 2020–2026, ratings and rookie-prior fits use only games "
        "and debut outcomes dated before that season's first game. Rosters are "
        "the most common starters across each team's first three regular-season games. "
        "Win rate uses balanced-side game probabilities.\n\n"
        + _markdown_table(results)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")


def main_template(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a roster YAML template")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    output = Path(args.output or f"rosters/{args.season}.yaml")
    _write_template(args.season, output)


def main() -> None:
    main_project()
