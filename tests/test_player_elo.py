import pandas as pd

from lol_elo.player_elo import (
    _convert_player_region_rating,
    _win_probability,
    run_player_elo,
)

CONFIG = {
    "starting_rating": 1500,
    "blue_side_advantage": 0,
    "k_base": 32,
    "k_min_ratio": 0.25,
    "k_decay": 20,
    "role_weights": {role: 1 for role in ("top", "jng", "mid", "bot", "sup")},
    "continuity_n": 5,
    "continuity_bonus": 0,
}
ROLES = ("top", "jng", "mid", "bot", "sup")


def _roster_rows(gameid: str, side: str, prefix: str) -> list[dict[str, str]]:
    return [
        {
            "gameid": gameid,
            "side": side,
            "position": role,
            "player_key": f"playerid:{prefix}-{role}",
            "playername": f"{prefix}-{role}",
        }
        for role in ROLES
    ]


def _game(
    gameid: str,
    game: int,
    date: str,
    blue: str,
    red: str,
    winner: str,
) -> dict[str, object]:
    return {
        "gameid": gameid,
        "game": game,
        "date": date,
        "year": 2024,
        "league": "LCK",
        "playoffs": 0,
        "is_international": False,
        "blue_team": blue,
        "blue_team_id": blue.lower(),
        "red_team": red,
        "red_team_id": red.lower(),
        "winner": winner,
        "winner_team_id": winner.lower(),
    }


def test_player_elo_is_zero_sum_with_equal_weights_and_k() -> None:
    games = pd.DataFrame([_game("g1", 1, "2024-01-01", "Alpha", "Beta", "Alpha")])
    players = pd.DataFrame(
        _roster_rows("g1", "Blue", "alpha") + _roster_rows("g1", "Red", "beta")
    )

    result = run_player_elo(games, players, CONFIG)

    assert abs(
        result.predictions.loc[0, "rating_change_blue"]
        + result.predictions.loc[0, "rating_change_red"]
    ) < 1e-10
    assert abs(result.ratings["rating"].sum() - 10 * 1500) < 1e-10


def test_swapping_sides_with_no_advantage_gives_symmetric_probabilities() -> None:
    games = pd.DataFrame(
        [
            _game("g1", 1, "2024-01-01", "Alpha", "Beta", "Alpha"),
            _game("g2", 2, "2024-01-01", "Beta", "Alpha", "Beta"),
        ]
    )
    players = pd.DataFrame(
        _roster_rows("g1", "Blue", "alpha")
        + _roster_rows("g1", "Red", "beta")
        + _roster_rows("g2", "Blue", "beta")
        + _roster_rows("g2", "Red", "alpha")
    )

    result = run_player_elo(games, players, {**CONFIG, "k_base": 0})

    probabilities = result.predictions["predicted_blue_win_prob"]
    assert abs(probabilities.iloc[0] + probabilities.iloc[1] - 1) < 1e-12
    assert abs(
        _win_probability(1650, 1450, 0) + _win_probability(1450, 1650, 0) - 1
    ) < 1e-12


def test_substitute_updates_but_absent_regular_starter_does_not() -> None:
    games = pd.DataFrame(
        [
            _game("g1", 1, "2024-01-01", "Alpha", "Beta", "Alpha"),
            _game("g2", 1, "2024-01-02", "Alpha", "Beta", "Alpha"),
        ]
    )
    alpha_game_one = _roster_rows("g1", "Blue", "alpha")
    alpha_game_two = _roster_rows("g2", "Blue", "alpha")
    alpha_game_two[0] = {
        **alpha_game_two[0],
        "player_key": "playerid:alpha-sub-top",
        "playername": "alpha-sub-top",
    }
    players = pd.DataFrame(
        alpha_game_one
        + _roster_rows("g1", "Red", "beta")
        + alpha_game_two
        + _roster_rows("g2", "Red", "beta")
    )

    result = run_player_elo(games, players, CONFIG)
    ratings = result.ratings.set_index("player_key")

    assert ratings.loc["playerid:alpha-top", "games_played"] == 1
    assert ratings.loc["playerid:alpha-sub-top", "games_played"] == 1
    assert ratings.loc["playerid:alpha-top", "rating"] != 1500
    assert ratings.loc["playerid:alpha-sub-top", "rating"] != 1500


def test_international_win_updates_both_region_offsets() -> None:
    games = pd.DataFrame(
        [
            {
                **_game("g1", 1, "2024-01-01", "Alpha", "KR Other", "Alpha"),
                "league": "LCK",
            },
            {
                **_game("g2", 1, "2024-01-02", "Beta", "CN Other", "Beta"),
                "league": "LPL",
            },
            {
                **_game("g3", 1, "2024-05-01", "Alpha", "Beta", "Alpha"),
                "league": "WLDs",
            },
        ]
    )
    players = pd.DataFrame(
        _roster_rows("g1", "Blue", "alpha")
        + _roster_rows("g1", "Red", "kr-other")
        + _roster_rows("g2", "Blue", "beta")
        + _roster_rows("g2", "Red", "cn-other")
        + _roster_rows("g3", "Blue", "alpha")
        + _roster_rows("g3", "Red", "beta")
    )
    config = {
        **CONFIG,
        "region_strength_enabled": True,
        "offseason_regression_enabled": False,
        "continuity_bonus": 0,
    }

    result = run_player_elo(games, players, config)

    assert result.region_offsets["KR"] > 0
    assert result.region_offsets["CN"] < 0


def test_same_region_game_does_not_change_offsets() -> None:
    games = pd.DataFrame(
        [{**_game("g1", 1, "2024-01-01", "Alpha", "Beta", "Alpha"), "league": "LCK"}]
    )
    players = pd.DataFrame(
        _roster_rows("g1", "Blue", "alpha") + _roster_rows("g1", "Red", "beta")
    )

    result = run_player_elo(
        games,
        players,
        {**CONFIG, "region_strength_enabled": True, "continuity_bonus": 0},
    )

    assert set(result.region_offsets.values()) == {0.0}


def test_region_move_preserves_effective_player_rating() -> None:
    offsets = {"KR": 24.0, "CN": -11.0}
    old_rating = 1640.0

    new_rating = _convert_player_region_rating(old_rating, "KR", "CN", offsets)

    assert new_rating + offsets["CN"] == old_rating + offsets["KR"]
