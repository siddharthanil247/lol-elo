"""Command line entry point for lol_elo."""

import argparse
import sys

from lol_elo.baseline import main as run_baseline
from lol_elo.ingest import main as run_ingest
from lol_elo.player_elo import main as run_player


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        run_ingest()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "baseline":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        run_baseline()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "player":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        run_player()
        return

    parser = argparse.ArgumentParser(prog="python -m lol_elo")
    parser.add_argument("command", choices=["ingest", "baseline", "player"])
    parser.parse_args()


if __name__ == "__main__":
    main()
