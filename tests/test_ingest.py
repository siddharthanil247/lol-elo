from pathlib import Path
import shutil

from lol_elo.ingest import ingest


FIXTURE = Path(__file__).parent / "fixtures" / "ingest_matches.csv"


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
