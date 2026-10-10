"""
Precompute predictions for every possible team matchup using each team's
most recent rolling stats, so the HTML dashboard can be a static lookup
(no python runtime needed in the browser).

Head-to-head is the one feature that's pair-specific rather than
team-specific, so it's computed directly from this season's actual
completed meetings between each pair, rather than pulled from either
team's "latest game" row (which reflects their most recent *actual*
opponent, not the hypothetical one picked in the dashboard).
"""
import pandas as pd
import numpy as np
import json
import joblib

from project_paths import output_path

LONG = output_path("nba_team_game_features.csv")
long_df = pd.read_csv(LONG, parse_dates=["date"])

TEAM_FEATURES = [
    "season_avg_pts_for", "season_avg_pts_against", "season_win_pct", "season_point_diff",
    "last10_avg_pts_for", "last10_avg_pts_against", "last10_win_pct", "last10_point_diff",
    "win_streak", "rest_days", "back_to_back", "split_win_pct", "strength_of_schedule",
]

# latest row per team = their most recent game (use it as "current form")
latest = (long_df.sort_values("date")
          .groupby("team", as_index=False)
          .tail(1)
          .set_index("team"))

teams = sorted(latest.index.tolist())
current_season = latest["season"].mode()[0]
print(f"{len(teams)} teams, current season: {current_season}")

# Actual head-to-head tally for every pair, using all their meetings played
# so far in the current season (this is the real, complete record entering
# a hypothetical next matchup between them).
season_games = long_df[long_df["season"] == current_season]
h2h_pairs = {}
for (team, opp), g in season_games.groupby(["team", "opponent"]):
    wins = int(g["win"].sum())
    n = len(g)
    h2h_pairs[(team, opp)] = {"h2h_net_record": 2 * wins - n, "h2h_meetings_so_far": n}

def get_h2h(team, opp):
    return h2h_pairs.get((team, opp), {"h2h_net_record": 0, "h2h_meetings_so_far": 0})

scaler = joblib.load(output_path("scaler.joblib"))
rf_cls = joblib.load(output_path("model_rf_cls.joblib"))
logreg = joblib.load(output_path("model_logreg.joblib"))
rf_home_pts = joblib.load(output_path("model_rf_home_pts.joblib"))
rf_away_pts = joblib.load(output_path("model_rf_away_pts.joblib"))

MODEL_FEATURE_ORDER = [
    "home_season_avg_pts_for", "home_season_avg_pts_against", "home_season_win_pct",
    "home_season_point_diff", "home_last10_avg_pts_for", "home_last10_avg_pts_against",
    "home_last10_win_pct", "home_last10_point_diff", "home_win_streak", "home_rest_days",
    "home_back_to_back", "home_split_win_pct", "home_strength_of_schedule",
    "home_h2h_net_record", "home_h2h_meetings_so_far",
    "away_season_avg_pts_for", "away_season_avg_pts_against", "away_season_win_pct",
    "away_season_point_diff", "away_last10_avg_pts_for", "away_last10_avg_pts_against",
    "away_last10_win_pct", "away_last10_point_diff", "away_win_streak", "away_rest_days",
    "away_back_to_back", "away_split_win_pct", "away_strength_of_schedule",
    "away_h2h_net_record", "away_h2h_meetings_so_far",
]

def build_row(home_team, away_team):
    h = latest.loc[home_team]
    a = latest.loc[away_team]
    h2h_home = get_h2h(home_team, away_team)
    h2h_away = get_h2h(away_team, home_team)
    row = {f"home_{c}": h[c] for c in TEAM_FEATURES}
    row.update({f"away_{c}": a[c] for c in TEAM_FEATURES})
    row["home_h2h_net_record"] = h2h_home["h2h_net_record"]
    row["home_h2h_meetings_so_far"] = h2h_home["h2h_meetings_so_far"]
    row["away_h2h_net_record"] = h2h_away["h2h_net_record"]
    row["away_h2h_meetings_so_far"] = h2h_away["h2h_meetings_so_far"]
    return pd.DataFrame([row])[MODEL_FEATURE_ORDER]

matchups = {}
for home_team in teams:
    for away_team in teams:
        if home_team == away_team:
            continue
        X = build_row(home_team, away_team)
        X_scaled = scaler.transform(X)
        rf_prob = rf_cls.predict_proba(X)[0, 1]
        logreg_prob = logreg.predict_proba(X_scaled)[0, 1]
        home_pts_pred = float(rf_home_pts.predict(X)[0])
        away_pts_pred = float(rf_away_pts.predict(X)[0])
        h2h = get_h2h(home_team, away_team)
        key = f"{home_team}|{away_team}"
        matchups[key] = {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": round(float(rf_prob), 4),
            "home_win_prob_logreg": round(float(logreg_prob), 4),
            "predicted_home_pts": round(home_pts_pred, 1),
            "predicted_away_pts": round(away_pts_pred, 1),
            "predicted_winner": home_team if rf_prob >= 0.5 else away_team,
            "h2h_meetings_this_season": h2h["h2h_meetings_so_far"],
            "h2h_net_record_home_perspective": h2h["h2h_net_record"],
        }

with output_path("matchups.json").open("w") as f:
    json.dump(matchups, f, indent=1)

# Team summary cards for dashboard (current form as of most recent game)
team_summary = {}
for t in teams:
    row = latest.loc[t]
    team_summary[t] = {
        "season": row["season"],
        "as_of_date": str(row["date"].date()),
        "games_played": int(row["games_played_so_far"]) + 1,
        "season_win_pct": round(float(row["season_win_pct"]), 3),
        "season_avg_pts_for": round(float(row["season_avg_pts_for"]), 1),
        "season_avg_pts_against": round(float(row["season_avg_pts_against"]), 1),
        "season_point_diff": round(float(row["season_point_diff"]), 1),
        "last10_win_pct": round(float(row["last10_win_pct"]), 3),
        "split_win_pct": round(float(row["split_win_pct"]), 3),
        "strength_of_schedule": round(float(row["strength_of_schedule"]), 3),
        "win_streak": int(row["win_streak"]),
    }

with output_path("team_summary.json").open("w") as f:
    json.dump(team_summary, f, indent=1)

print(f"Wrote {len(matchups)} matchup predictions and {len(team_summary)} team summaries")
print(json.dumps(list(matchups.items())[:2], indent=2))
