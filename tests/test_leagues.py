import pandas as pd

from lol_elo.leagues import filter_tier1_games


def test_pcs_year_range_excludes_2025_and_includes_2022() -> None:
    games = pd.DataFrame(
        {
            "league": ["PCS", "PCS"],
            "year": [2025, 2022],
            "gameid": ["pcs-2025", "pcs-2022"],
        }
    )

    filtered = filter_tier1_games(games)

    assert filtered["gameid"].tolist() == ["pcs-2022"]


def test_lcs_and_lec_start_in_2019() -> None:
    games = pd.DataFrame(
        {
            "league": ["LCS", "LCS", "LEC", "LEC"],
            "year": [2018, 2019, 2018, 2019],
            "gameid": ["lcs-2018", "lcs-2019", "lec-2018", "lec-2019"],
        }
    )

    filtered = filter_tier1_games(games)

    assert filtered["gameid"].tolist() == ["lcs-2019", "lec-2019"]
