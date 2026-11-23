import pandas as pd

from lol_elo.evaluate import evaluate_predictions
from lol_elo.team_elo import run_team_elo

MODEL_CONFIG = {
    "k_factor": 32,
    "starting_rating": 1500,
    "blue_side_advantage": 20,
    "series_formats": {"default": {"regular": 3, "playoffs": 5}},
}


def test_team_elo_probabilities_updates_and_winner_rating() -> None:
    games = pd.DataFrame(
        [
            {
                "gameid": "g1",
                "game": 1,
                "date": "2024-01-01",
                "year": 2024,
                "league": "LCK",
                "playoffs": 0,
                "is_international": False,
                "blue_team": "Alpha",
                "blue_team_id": "alpha",
                "red_team": "Beta",
                "red_team_id": "beta",
                "winner": "Alpha",
                "winner_team_id": "alpha",
            },
            {
                "gameid": "g2",
                "game": 1,
                "date": "2024-01-02",
                "year": 2024,
                "league": "LCK",
                "playoffs": 0,
                "is_international": False,
                "blue_team": "Gamma",
                "blue_team_id": "gamma",
                "red_team": "Alpha",
                "red_team_id": "alpha",
                "winner": "Alpha",
                "winner_team_id": "alpha",
            },
            {
                "gameid": "g3",
                "game": 1,
                "date": "2024-01-03",
                "year": 2024,
                "league": "LCK",
                "playoffs": 0,
                "is_international": False,
                "blue_team": "Beta",
                "blue_team_id": "beta",
                "red_team": "Alpha",
                "red_team_id": "alpha",
                "winner": "Alpha",
                "winner_team_id": "alpha",
            },
        ]
    )

    result = run_team_elo(games, MODEL_CONFIG)

    assert result.predictions["predicted_blue_win_prob"].between(
        0, 1, inclusive="neither"
    ).all()
    total_changes = (
        result.predictions["rating_change_blue"]
        + result.predictions["rating_change_red"]
    )
    assert total_changes.abs().lt(1e-10).all()
    assert result.ratings.iloc[0]["team_key"] == "teamid:alpha"


def test_series_evaluation_uses_best_of_and_reports_metrics() -> None:
    games = pd.DataFrame(
        [
            ("g1", 1, "Alpha", "alpha", "Beta", "beta", "Alpha", "alpha"),
            ("g2", 2, "Beta", "beta", "Alpha", "alpha", "Beta", "beta"),
            ("g3", 3, "Alpha", "alpha", "Beta", "beta", "Alpha", "alpha"),
        ],
        columns=[
            "gameid",
            "game",
            "blue_team",
            "blue_team_id",
            "red_team",
            "red_team_id",
            "winner",
            "winner_team_id",
        ],
    )
    games["date"] = "2024-01-01"
    games["year"] = 2024
    games["league"] = "LCK"
    games["playoffs"] = 0
    games["is_international"] = False

    result = run_team_elo(games, MODEL_CONFIG)
    evaluation = evaluate_predictions(result.predictions, MODEL_CONFIG)

    assert len(evaluation.series_predictions) == 1
    assert evaluation.series_predictions.iloc[0]["best_of"] == 3
    assert evaluation.series_predictions.iloc[0]["actual_team_a_win"] == 1
    assert set(evaluation.metrics["scope"]) == {"overall", "year", "competition"}
    assert len(evaluation.calibration) == 10
