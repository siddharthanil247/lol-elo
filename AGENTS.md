# Project context

- Goal: build a player-level Elo system whose team strength is derived from its five players, so teams with new rosters can be rated before they play.
- Data source: Oracle's Elixir match CSVs are placed manually in `data/raw/`. Each game has one row per player plus two team rows.
- Tier 1 leagues and international events are configured in `config/leagues.yaml`; do not hardcode them in model or data-processing code.
- Evaluate with walk-forward validation only: predictions for a game may use data from before that game, never from the game itself or later games. Prevent data leakage.
- Always order games by the `date` column, never by `year`, because some offseason games are labeled with the next season's year.
- Main metrics: log loss and Brier score for game outcomes, plus series win accuracy.
- Keep Python package code under `src/lol_elo/`, tests under `tests/`, and notebooks under `notebooks/`.
- Keep generated/derived data under `data/processed/`; this directory is gitignored. Preserve manually supplied source files in `data/raw/`.
- The project uses pandas, numpy, pyyaml, pytest, and Ruff. Do not add model code until requested.
