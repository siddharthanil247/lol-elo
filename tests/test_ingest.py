import shutil
from pathlib import Path

from lol_elo.ingest import ingest

FIXTURE = Path(__file__).parent / "fixtures" / "ingest_matches.csv"
WORLDS_FIXTURE = Path(__file__).parent / "fixtures" / "worlds_matches.csv"


def test_ingest_splits_games_and_sorts_stably(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    shutil.copy(FIXTURE, raw_dir / FIXTURE.name)

    result = ingest(raw_dir=raw_dir, processed_dir=processed_dir)

    assert result.games["gameid"].tolist() == ["g-a", "g-c", "g-b"]
    assert result.games["winner"].tolist() == ["Red A", "Blue C", "Blue B"]
    assert len(result.games) == 3
    assert len(result.players) == 30
    assert result.players.groupby("gameid").size().eq(10).all()
    assert result.games["gameid"].isin(result.players["gameid"]).all()
    assert result.games["date"].is_monotonic_increasing
    assert result.games_unfiltered["gameid"].tolist() == ["g-a", "g-c", "g-b"]
    assert (processed_dir / "games.parquet").exists()
    assert (processed_dir / "players.parquet").exists()
    assert (processed_dir / "games_unfiltered.parquet").exists()
    assert (processed_dir / "players_unfiltered.parquet").exists()
    assert (processed_dir / "player_aliases.csv").exists()


def test_international_games_require_two_tier1_teams(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    shutil.copy(FIXTURE, raw_dir / FIXTURE.name)
    shutil.copy(WORLDS_FIXTURE, raw_dir / WORLDS_FIXTURE.name)

    result = ingest(raw_dir=raw_dir, processed_dir=processed_dir)

    assert "wld-out" not in result.games["gameid"].tolist()
    assert "wld-in" in result.games["gameid"].tolist()
    assert result.games.loc[
        result.games["gameid"].eq("wld-in"), "is_international"
    ].all()
    assert result.international_drops.to_dict("records") == [
        {"league": "WLDs", "year": 2024, "dropped_games": 1}
    ]
