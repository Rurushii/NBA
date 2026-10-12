# Bundled dataset audit

This audit describes the bundled `nba_games_raw.csv` snapshot identified
by the SHA-256 checksum below.

- File: `nba_games_raw.csv`
- SHA-256: `c9a9e8618e9fb1bd0d82ef600945628c9236ac3f2e15d03f059ecb8df22884a5`
- Raw game-result rows: 2,317
- Observed dates: 2023-10-24 through 2026-04-30
- Teams: all 30 current NBA team names
- Structural checks found no missing required values, invalid scores or dates,
  duplicate games, self-matchups, or multiple appearances per team/date.

## Coverage

| Season label | Rows | October-April months without rows |
| --- | ---: | --- |
| 2023-24 | 840 | November 2023 |
| 2024-25 | 621 | October 2024, November 2024, February 2025 |
| 2025-26 | 856 | January 2026 |

These counts describe recorded rows, not a verified complete schedule.
Missing histories can distort records, rolling form, streaks, rest days,
and head-to-head totals. Do not fill missing games with guessed results.

## Provenance and game types

The README attributes the data to Basketball Reference. The repository does
not contain the scraper, source URLs per game, provider game IDs, collection
timestamps, or a game-type field. Source accuracy is not independently verified.

There are 25 season/matchup pairs with more than four recorded meetings.
Boston-Philadelphia and Denver-Minnesota each have 10 meetings in 2025-26.
Late-April rows and repeated matchups suggest additional game types may be
included, but these signals do not establish which rows are postseason games.
Do not describe the snapshot as verified regular-season-only data.

Before separating regular-season, play-in, playoff, or other games, reconcile
rows against a trusted schedule containing stable game IDs and game types.
Document its provider, source URL, retrieval time, and permitted usage.

## Model sample sizes

The feature pipeline produces 2,268 eligible examples from the raw CSV.
Its chronological split uses 1,811 training examples and 457 holdout examples.
Existing metrics describe this incomplete snapshot, not verified prospective
performance on a complete regular-season dataset.

## Validation

Run `audit_data.py` with the project's virtual environment and the same
directory settings as the pipeline. It writes `data_audit.json` to
`NBA_OUTPUT_DIR`.

Exit status 0 means structural checks passed; coverage and provenance warnings
still require review. Exit status 1 indicates invalid data or an unreadable input.
The script does not clean data or label game types.