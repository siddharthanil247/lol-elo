# lol-elo

Player-level Elo ratings for Tier 1 League of Legends teams, with team strength
built from the five players on each roster.

Oracle's Elixir match CSVs go in `data/raw/`. League and event scope is configured
in `config/leagues.yaml`. See [AGENTS.md](AGENTS.md) for project conventions.

This repository currently contains project scaffolding only; model code has not
been added.

## Ingestion

Run `python -m lol_elo ingest` to create filtered Tier 1 tables and unfiltered
tables in `data/processed/`. Edit `data/processed/player_aliases.csv` to merge
identities: map a source key such as `playerid:old-id` or `name:oldname` in
`player_alias` to the desired canonical key in `canonical_player_key`.
