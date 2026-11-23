"""Command line entry point for lol_elo."""

import argparse
import sys

from lol_elo.ingest import main as run_ingest


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        run_ingest()
        return

    parser = argparse.ArgumentParser(prog="python -m lol_elo")
    parser.add_argument("command", choices=["ingest"])
    parser.parse_args()


if __name__ == "__main__":
    main()
