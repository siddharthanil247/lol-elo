# lol-elo

Player-level Elo ratings for Tier 1 League of Legends teams. Team strength is
built from the five players on a roster, allowing new lineups to be rated before
they play. Evaluation is walk-forward: a prediction only uses games dated before
the predicted game. Game processing is ordered by `date`, never by `year`.

## Data setup

Download Oracle's Elixir match CSVs and place them manually in `data/raw/`.
Each CSV should contain player rows and the two team rows for each game. Then
run:

```sh
uv run python -m lol_elo ingest
```

Ingestion writes filtered Tier 1 and unfiltered player/game Parquet files under
`data/processed/`. The unfiltered tables support lower-league rookie priors.
Edit `data/processed/player_aliases.csv` to merge renamed or inconsistently
identified players. The `player_alias` column accepts a source player key or a
normalized `name:` key; `canonical_player_key` is the identity to use.

Tier 1 league names, year ranges, regional mappings, international eligibility,
and regular-season formats live in `config/leagues.yaml`. League names are
matched against all historical data, so historical names are listed explicitly.

## Commands

Run the team Elo benchmark after ingestion:

```sh
uv run python -m lol_elo baseline --start-year 2019
```

Earlier games are burn-in. The player-level evaluation uses the same scored
games and reports comparison metrics:

```sh
uv run python -m lol_elo player --start-year 2019
uv run python -m lol_elo player --start-year 2019 --all-time
```

Inspect identity matches and their ratings:

```sh
uv run python -m lol_elo lookup Keria Faker
```

Create a season roster template from each Tier 1 team's most common starters in
its latest five games, then edit it:

```sh
uv run python -m lol_elo roster-template --season 2027
# edit rosters/2027.yaml
```

Project a season and simulate each league's regular season 10,000 times:

```sh
uv run python -m lol_elo project --season 2027 --rosters rosters/2027.yaml
```

Run historical roster projections, audit a reproducible sample of roster
changes, and update `notes/backtest.md`:

```sh
uv run python -m lol_elo backtest
```

Search model settings using 2019–2023 for tuning and 2024 onward only for
holdout evaluation. The number of random trials and seed can be changed:

```sh
uv run python -m lol_elo tune --trials 8 --seed 42
```

Build a self-contained HTML report. `make report` also opens it on macOS:

```sh
uv run python -m lol_elo report --season 2027
make report SEASON=2027
```

## Configuration

`config/model.yaml` controls rating behavior:

- `k_base`, `k_min_ratio`, and `k_decay` set the experience-shrinking player
  update size: `k_base * max(k_min_ratio, 1 / (1 + games_played / k_decay))`.
- `starting_rating` is the rating before any history; `blue_side_advantage`
  adds rating points to the blue team's game probability.
- `role_weights` weights top, jungle, mid, bot, and support ratings when forming
  a team rating. Weights are normalized by the player model.
- `continuity_n` sets the recent team-game window used for shared starters;
  `continuity_bonus` is the rating bonus at five shared starters.
- `region_strength_enabled` and `k_region` control cross-region rating offsets.
- `use_rookie_priors` enables fitted lower-league starting ratings from
  `config/priors_fitted.yaml`.
- `offseason_regression_enabled`, `regress_frac`, `new_season_m`, and
  `new_season_k_mult` control rating regression and increased update size for a
  player's first games in a season.
- `tuning_objective_weights` sets the game log-loss and backtest MAE objective
  weights (default 0.7 and 0.3).
- `series_formats` sets default and league-specific regular-season and playoff
  best-of lengths.

`config/model_tuned.yaml` and `config/priors_tuned.yaml` preserve the best
random-search candidate. Tuning copies them into the active config only when
the candidate beats the current settings on the weighted 2024+ holdout.
`config/leagues.yaml` has editable league year bounds, region memberships,
international-event rules, and schedule formats.

## Results

The corrected 2020–2026 backtest covers 486 team-seasons. Player projections
achieved Spearman correlation **0.6071** and win-rate MAE **0.1121**, versus
**0.4759 / 0.1335** for last-season win rate and **0.4990 / 0.1375** for carried
team Elo. In the 309 seasons with at least three player replacements, player
projections reached **0.4438 / 0.1223**, versus **0.2568 / 0.1494** and
**0.2726 / 0.1491**, respectively. See [notes/backtest.md](notes/backtest.md)
for league-level results and the roster-change audit, and
[notes/tuning.md](notes/tuning.md) for the holdout decision and parameter search.

## Development

```sh
make test
make lint
```
