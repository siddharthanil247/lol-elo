import pandas as pd

from lol_elo.player_elo import ROLES
from lol_elo.project import before_season, project_team_ratings


def _roster() -> dict:
    return {
        "name": "Alpha",
        "teamid": "alpha-id",
        "league": "LCK",
        "starters": {role: role.title() for role in ROLES},
    }


def _ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_key": f"playerid:{role}",
                "rating": 1600.0,
                "games_played": 100,
                "region": "KR",
            }
            for role in ROLES
        ]
    )


def _catalog() -> tuple[dict[str, set[str]], dict[str, str]]:
    return ({role: {f"playerid:{role}"} for role in ROLES}, {})


def test_unchanged_roster_gets_frozen_rating_and_full_continuity() -> None:
    starters = {f"playerid:{role}" for role in ROLES}
    projected, warnings = project_team_ratings(
        [_roster()],
        _ratings(),
        {"KR": 25.0},
        {"teamid:alpha-id": frozenset(starters)},
        {"role_weights": {role: 1 for role in ROLES}, "continuity_bonus": 50},
        {"regions": {"KR": ["LCK"]}},
        pd.DataFrame(),
        {"tier1_rookie_base": 1500, "global_shrink": 0, "league_shrink": {}},
        "2027-01-01",
        _catalog(),
    )

    assert not warnings
    assert projected.loc[0, "continuity_starters"] == 5
    assert projected.loc[0, "projected_rating"] == 1675


def test_higher_rated_starter_increases_projected_rating() -> None:
    roster = _roster()
    starters = {f"playerid:{role}" for role in ROLES}
    org_lineups = {"teamid:alpha-id": frozenset(starters)}
    config = {"role_weights": {role: 1 for role in ROLES}, "continuity_bonus": 0}
    leagues = {"regions": {"KR": ["LCK"]}}
    priors = {"tier1_rookie_base": 1500, "global_shrink": 0, "league_shrink": {}}
    ratings = _ratings()
    normal, _ = project_team_ratings(
        [roster],
        ratings,
        {"KR": 0},
        org_lineups,
        config,
        leagues,
        pd.DataFrame(),
        priors,
        "2027-01-01",
        _catalog(),
    )

    roster["starters"] = {**roster["starters"], "top": "Rookie"}
    ratings = pd.concat(
        [
            ratings,
            pd.DataFrame(
                [{"player_key": "playerid:rookie", "rating": 1800, "region": "KR"}]
            ),
        ],
        ignore_index=True,
    )
    catalog = _catalog()
    catalog[0]["rookie"] = {"playerid:rookie"}
    changed, _ = project_team_ratings(
        [roster],
        ratings,
        {"KR": 0},
        org_lineups,
        config,
        leagues,
        pd.DataFrame(),
        priors,
        "2027-01-01",
        catalog,
    )

    assert changed.loc[0, "projected_rating"] > normal.loc[0, "projected_rating"]


def test_preseason_snapshot_excludes_games_on_or_after_first_game() -> None:
    games = pd.DataFrame(
        [
            {"gameid": "old", "date": "2019-12-20", "year": 2019},
            {"gameid": "first", "date": "2020-01-10", "year": 2020},
            {"gameid": "later", "date": "2020-01-11", "year": 2020},
        ]
    )
    players = pd.DataFrame(
        [{"gameid": gameid, "player_key": "playerid:p"} for gameid in games.gameid]
    )

    start, prior_games, prior_players = before_season(games, players, 2020)

    assert start == pd.Timestamp("2020-01-10")
    assert prior_games.gameid.tolist() == ["old"]
    assert prior_players.gameid.tolist() == ["old"]
