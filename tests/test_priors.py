import pandas as pd
import pytest

from lol_elo.priors import fit_rookie_priors, prior_for_player


def test_player_without_lower_league_history_starts_at_base() -> None:
    fitted = {"tier1_rookie_base": 1523.5, "global_shrink": 0.4, "league_shrink": {}}

    rating, league = prior_for_player(
        "playerid:new", "2024-01-01", pd.DataFrame(), fitted
    )

    assert rating == 1523.5
    assert league is None


def test_dominant_lower_league_player_gets_higher_prior() -> None:
    history = pd.DataFrame(
        [
            {
                "player_key": "playerid:strong",
                "date": "2023-12-01",
                "gameid": "a",
                "league": "LDL",
                "rating": 1600,
                "league_mean": 1500,
            },
            {
                "player_key": "playerid:weak",
                "date": "2023-12-01",
                "gameid": "b",
                "league": "LDL",
                "rating": 1400,
                "league_mean": 1500,
            },
        ]
    )
    fitted = {"tier1_rookie_base": 1500, "global_shrink": 0.5, "league_shrink": {}}

    strong, _ = prior_for_player("playerid:strong", "2024-01-01", history, fitted)
    weak, _ = prior_for_player("playerid:weak", "2024-01-01", history, fitted)

    assert strong == 1550
    assert weak == 1450
    assert strong > weak


def test_prior_fit_excludes_debuts_in_2024_and_later() -> None:
    debuts = pd.DataFrame(
        [
            {
                "debut_year": 2022,
                "league": "LDL",
                "lower_delta": 100,
                "rating_after_20": 1600,
            },
            {
                "debut_year": 2023,
                "league": "LFL",
                "lower_delta": 0,
                "rating_after_20": 1500,
            },
            {
                "debut_year": 2024,
                "league": "LDL",
                "lower_delta": 100,
                "rating_after_20": 5000,
            },
        ]
    )

    fitted = fit_rookie_priors(debuts, cutoff_year=2024)

    assert fitted["fit_debuts"] == 2
    assert fitted["tier1_rookie_base"] == 1500
    assert fitted["global_shrink"] == pytest.approx(1)
