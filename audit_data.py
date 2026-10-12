import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from project_paths import data_path, output_path

COLUMNS = ["season", "date", "away_team", "away_pts", "home_team", "home_pts"]
TEAMS = {
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
    "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
    "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns",
    "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors",
    "Utah Jazz", "Washington Wizards",
}


def audit(path: Path) -> dict:
    content = path.read_bytes()
    header = next(csv.reader(content.decode("utf-8-sig").splitlines()), [])
    report = {
        "input_file": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "row_count": 0,
        "errors": [],
        "warnings": ["Source provenance and game types are not independently verified."],
        "seasons": {},
        "matchups_to_review": [],
    }
    if len(header) != len(set(header)):
        report["errors"].append("Duplicate column names.")
    missing = sorted(set(COLUMNS) - set(header))
    if missing:
        report["errors"].append(f"Missing columns: {missing}")
        return report
    df = pd.read_csv(path, dtype=str)
    report["row_count"] = len(df)
    if df.empty:
        report["errors"].append("No game rows.")
        return report

    def check(message, mask):
        bad = mask.fillna(True)
        if bad.any():
            rows = [int(i) + 2 for i in df.index[bad][:10]]
            report["errors"].append(f"{message}: {int(bad.sum())} rows; examples {rows}")

    check("Missing required values", df[COLUMNS].isna().any(axis=1))
    for col in ["home_team", "away_team"]:
        check(f"Unknown or incorrectly formatted {col}", ~df[col].isin(TEAMS))
    check("A team plays itself", df["home_team"].eq(df["away_team"]))
    scores = {}
    for col in ["home_pts", "away_pts"]:
        scores[col] = pd.to_numeric(df[col], errors="coerce")
        score = scores[col]
        check(f"Invalid {col}; expected a positive integer",
              ~np.isfinite(score) | score.le(0) | score.mod(1).ne(0))
    check("Tied final score", scores["home_pts"].eq(scores["away_pts"]))
    dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    check("Invalid date; expected YYYY-MM-DD",
          ~df["date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False) | dates.isna())
    season_format = df["season"].str.fullmatch(r"20\d{2}-\d{2}", na=False)
    years = pd.to_numeric(df["season"].str[:4], errors="coerce").where(season_format)
    endings = pd.to_numeric(df["season"].str[-2:], errors="coerce").where(season_format)
    valid_season = season_format & endings.eq((years + 1).mod(100))
    check("Invalid season; expected consecutive years such as 2025-26", ~valid_season)
    # NBA seasons in this snapshot run from October through the following June.
    # This checks the season label, not whether a game is regular-season or playoff.
    starts = pd.to_datetime(years.astype("Int64").astype(str) + "-10-01", errors="coerce")
    ends = pd.to_datetime((years + 1).astype("Int64").astype(str) + "-06-30", errors="coerce")
    check("Date outside its labeled season",
          valid_season & dates.notna() & (dates.lt(starts) | dates.gt(ends)))
    check("Duplicate game", df.duplicated(["date", "home_team", "away_team"], keep=False))
    appearances = pd.concat([
        df[["date", "home_team"]].rename(columns={"home_team": "team"}),
        df[["date", "away_team"]].rename(columns={"away_team": "team"}),
    ])
    repeated = appearances.duplicated(["date", "team"], keep=False)
    if repeated.any():
        report["errors"].append("A team has multiple game appearances on the same date.")

    clean = df.loc[valid_season & dates.notna()].copy()
    clean["parsed_date"] = dates.loc[clean.index]
    for season, games in clean.groupby("season"):
        year = int(season[:4])
        counts = games["parsed_date"].dt.to_period("M").value_counts().sort_index()
        expected = pd.period_range(f"{year}-10", f"{year + 1}-04", freq="M")
        absent = [str(month) for month in expected if month not in counts.index]
        report["seasons"][season] = {
            "rows": len(games),
            "first_date": str(games["parsed_date"].min().date()),
            "last_date": str(games["parsed_date"].max().date()),
            "games_by_month": {str(month): int(n) for month, n in counts.items()},
            "months_without_rows": absent,
        }
        if absent:
            report["warnings"].append(f"{season}: months without rows: {', '.join(absent)}")
    pairs = df.loc[df["home_team"].isin(TEAMS) & df["away_team"].isin(TEAMS)].copy()
    pairs["pair"] = [
        " | ".join(sorted(pair))
        for pair in pairs[["home_team", "away_team"]].itertuples(index=False, name=None)
    ]
    for (season, pair), count in pairs.groupby(["season", "pair"]).size().items():
        if count > 4:
            report["matchups_to_review"].append({
                "season": season, "teams": pair, "games": int(count)
            })
    if report["matchups_to_review"]:
        report["warnings"].append(
            "Some pairs have more than four meetings; verify game types against a trusted schedule."
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=data_path("nba_games_raw.csv"))
    parser.add_argument("--report", type=Path, default=output_path("data_audit.json"))
    args = parser.parse_args()
    if args.input.resolve() == args.report.resolve():
        parser.error("The report must not overwrite the input CSV.")
    try:
        report = audit(args.input)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        parser.exit(1, f"Cannot audit input: {error}\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Rows: {report['row_count']}; errors: {len(report['errors'])}; "
        f"warnings: {len(report['warnings'])}"
    )
    for message in report["errors"] + report["warnings"]:
        print(message)
    print(f"Report: {args.report}")
    # Zero means structural checks passed, not that coverage or provenance is verified.
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())