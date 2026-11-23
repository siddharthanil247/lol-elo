"""Build a standalone HTML report from roster projections and model outputs."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

from lol_elo.leagues import load_leagues
from lol_elo.player_elo import _recent_player_keys, run_player_elo
from lol_elo.priors import (
    build_initial_rookie_ratings,
    load_fitted_priors,
    load_or_build_lower_history,
)
from lol_elo.project import _load_inputs, project_season
from lol_elo.team_elo import load_model_config


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return html.escape(str(value))


def _rating_table(frame: pd.DataFrame) -> str:
    columns = [
        "playername",
        "rating",
        "region",
        "effective_rating",
        "games_played",
    ]
    headers = "".join(
        f"<th>{html.escape(column.replace('_', ' ').title())}</th>"
        for column in columns
    )
    body = []
    for row in frame[columns].itertuples(index=False, name=None):
        cells = []
        for index, value in enumerate(row):
            shown = f"{value:.1f}" if index in {1, 3} else _text(value)
            cells.append(f"<td>{shown}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _backtest_table(frame: pd.DataFrame) -> str:
    if frame.empty or not {"league", "cohort"}.issubset(frame.columns):
        return (
            "<p>No saved backtest summary. Run "
            "<code>python -m lol_elo backtest</code> first.</p>"
        )
    display = frame.loc[
        frame["league"].eq("overall")
        & frame["cohort"].isin(["all teams", "3+ roster changes"])
    ].copy()
    if display.empty:
        return (
            "<p>No saved backtest summary. Run "
            "<code>python -m lol_elo backtest</code> first.</p>"
        )
    columns = ["cohort", "method", "teams", "spearman", "mae"]
    header = "".join(
        f"<th>{html.escape(column.replace('_', ' ').title())}</th>"
        for column in columns
    )
    body = []
    for row in display[columns].itertuples(index=False, name=None):
        cells = []
        for index, value in enumerate(row):
            if index in {3, 4} and pd.notna(value):
                shown = f"{value:.4f}"
            else:
                shown = _text(value)
            cells.append(f"<td>{shown}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _standings_table(frame: pd.DataFrame) -> str:
    columns = [
        "team",
        "projected_rating",
        "expected_win_rate",
        "average_standing",
        "probability_first",
        "probability_top2",
        "probability_top4",
    ]
    headers = [
        "Team",
        "Projected rating",
        "Expected win rate",
        "Average standing",
        "1st",
        "Top 2",
        "Top 4",
    ]
    body = []
    ordered = frame.sort_values(
        ["probability_first", "projected_rating"], ascending=[False, False]
    )
    for row in ordered[columns].itertuples(index=False, name=None):
        team, rating, rate, standing, first, top2, top4 = row

        def bar(value: float) -> str:
            percent = max(0.0, min(100.0, float(value) * 100))
            return (
                '<div class="bar"><span '
                f'style="width:{percent:.2f}%"></span></div>'
                f"<small>{percent:.1f}%</small>"
            )

        body.append(
            "<tr>"
            f"<td>{_text(team)}</td><td>{rating:.1f}</td>"
            f"<td>{rate:.1%}</td><td>{standing:.2f}</td>"
            f"<td>{bar(first)}</td><td>{bar(top2)}</td><td>{bar(top4)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{heading}</th>" for heading in headers)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _region_table(history: pd.DataFrame) -> str:
    if history.empty:
        return "<p>No regional offsets are available.</p>"
    matrix = history.pivot_table(
        index="year", columns="region", values="offset", aggfunc="last"
    ).sort_index()
    columns = list(matrix.columns)
    header = "<th>Year</th>" + "".join(
        f"<th>{_text(region)}</th>" for region in columns
    )
    body = []
    for year, row in matrix.iterrows():
        cells = "".join(
            f"<td>{float(row[region]):.1f}</td>"
            if pd.notna(row[region])
            else "<td>—</td>"
            for region in columns
        )
        body.append(f"<tr><td>{int(year)}</td>{cells}</tr>")
    return (
        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def build_report(season: int, roster_path: str | Path | None = None) -> Path:
    games, players, all_games, all_players = _load_inputs()
    model = load_model_config()
    leagues = load_leagues()
    fitted = load_fitted_priors()
    lower = load_or_build_lower_history(all_games, all_players, model)
    rosters_path = Path(roster_path or f"rosters/{season}.yaml")
    projection, start_date, warnings = project_season(
        season,
        rosters_path,
        games,
        players,
        lower,
        fitted,
        model,
        leagues,
        runs=10_000,
    )
    projection_path = Path(f"data/processed/roster_projection_{season}.csv")
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    projection.drop(columns=["starter_keys"], errors="ignore").to_csv(
        projection_path, index=False
    )

    initial = build_initial_rookie_ratings(games, players, lower, fitted)
    current = run_player_elo(
        games, players, model, initial_ratings=initial, leagues_config=leagues
    )
    active = _recent_player_keys(players, games)
    top = current.role_ratings.merge(
        current.ratings[["player_key", "region"]], on="player_key", how="left"
    )
    top["effective_rating"] = top["rating"] + top["region"].map(
        current.region_offsets
    ).fillna(0.0)
    top = top.loc[top["player_key"].isin(active)].sort_values(
        ["role", "effective_rating"], ascending=[True, False], kind="stable"
    )
    backtest_path = Path("data/processed/backtest_summary.csv")
    backtest = pd.read_csv(backtest_path) if backtest_path.exists() else pd.DataFrame()

    sections = []
    for league, frame in projection.groupby("league", sort=True):
        sections.append(
            f"<section><h3>{_text(league)}</h3>{_standings_table(frame)}</section>"
        )
    role_sections = []
    for role in ("top", "jng", "mid", "bot", "sup"):
        role_sections.append(
            f"<section><h3>{_text(role.title())}</h3>"
            f"{_rating_table(top.loc[top['role'].eq(role)].head(10))}</section>"
        )
    warning_html = (
        "<ul>" + "".join(f"<li>{_text(item)}</li>" for item in warnings) + "</ul>"
        if warnings
        else "<p>No roster-name warnings.</p>"
    )
    latest_offset = current.region_offset_history
    html_text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LoL Elo {season} roster projection</title>
<style>
body{{font:15px/1.5 system-ui,-apple-system,sans-serif;
margin:2rem auto;padding:0 1rem;max-width:1400px;
color:#17212b;background:#f7f9fb}}
h1,h2,h3{{line-height:1.2}}
h2{{margin-top:2.2rem;border-bottom:1px solid #ccd5df;padding-bottom:.4rem}}
section{{margin:1rem 0 2rem;background:white;padding:1rem;
border:1px solid #dbe2e8;border-radius:8px;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}
th,td{{padding:.45rem .55rem;border-bottom:1px solid #e3e8ed;
text-align:left;white-space:nowrap}}
th{{background:#edf2f6;position:sticky;top:0}}
.bar{{width:90px;height:9px;background:#e6ebf0;border-radius:6px;
display:inline-block;vertical-align:middle;margin-right:5px}}
.bar span{{display:block;height:100%;background:#3478b9;border-radius:6px}}
small{{font-variant-numeric:tabular-nums}}
.meta{{color:#52616e}}
</style></head><body>
<h1>League of Legends {season} roster projections</h1>
<p class="meta">Ratings frozen before {start_date:%Y-%m-%d}; 10,000 Monte Carlo
regular-season simulations per league.</p>
<h2>Projected standings</h2>{"".join(sections)}
<h2>Active player top 10 by role</h2>
<p class="meta">Effective rating includes the current region offset. Active means
a Tier 1 game within 365 days of the latest game.</p>{"".join(role_sections)}
<h2>Regional offsets over time</h2>{_region_table(latest_offset)}
<h2>Backtest summary</h2>{_backtest_table(backtest)}
<h2>Roster warnings</h2>{warning_html}
<footer><p class="meta">Generated from the project data and configuration.
All report data is embedded in this file.</p></footer>
</body></html>"""
    output = Path(f"reports/{season}.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standalone HTML report")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--rosters", default=None)
    args = parser.parse_args()
    output = build_report(args.season, args.rosters)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
