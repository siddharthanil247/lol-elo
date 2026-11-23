# lol-elo

Player-level Elo ratings for Tier 1 League of Legends teams, with team strength
built from the five players on each roster.

Oracle's Elixir match CSVs go in `data/raw/`. League and event scope is configured
in `config/leagues.yaml`. See [AGENTS.md](AGENTS.md) for project conventions.

## Ingestion

Run `python -m lol_elo ingest` to create filtered Tier 1 tables and unfiltered
tables in `data/processed/`. Edit `data/processed/player_aliases.csv` to merge
identities: map a source key such as `playerid:old-id` or `name:oldname` in
`player_alias` to the desired canonical key in `canonical_player_key`.

## Team Elo baseline

Run `python -m lol_elo baseline --start-year 2019` after ingestion. Earlier
games update ratings as burn-in and are excluded from the evaluation metrics.
Elo parameters and editable series formats are in `config/model.yaml`; game
probabilities are written to `data/processed/baseline_preds.parquet`.
