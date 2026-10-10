"""
Feature engineering for NBA game outcome / score prediction.

Turns the raw game-result CSV (one row per game: away/home team + final score)
into a per-team-per-game "long" table with rolling, pre-game-only features
(no data leakage: every feature for a given game is computed strictly from
that team's games BEFORE that date), then re-merges into a wide table
(one row per game, home_* and away_* feature columns) ready for modeling.

Feature set:
  - season-to-date win pct, avg pts for/against, point differential
  - last-10-game win pct, avg pts for/against, point differential (recent form)
  - home/away-specific win pct (split by game location, not overall record)
  - current win/loss streak
  - rest days since last game, and a back-to-back flag
  - strength of schedule (avg win pct of opponents faced so far this season)
  - head-to-head net record vs this exact opponent, this season, prior to this game
"""
import pandas as pd
import numpy as np

from project_paths import data_path, output_path

RAW = data_path("nba_games_raw.csv")
OUT_LONG = output_path("nba_team_game_features.csv")
OUT_WIDE = output_path("nba_games_features.csv")

df = pd.read_csv(RAW)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df["game_id"] = df.index

# ---- Long format: one row per team per game ----
home = df.rename(columns={
    "home_team": "team", "away_team": "opponent",
    "home_pts": "team_pts", "away_pts": "opp_pts",
})[["game_id", "season", "date", "team", "opponent", "team_pts", "opp_pts"]]
home["is_home"] = 1

away = df.rename(columns={
    "away_team": "team", "home_team": "opponent",
    "away_pts": "team_pts", "home_pts": "opp_pts",
})[["game_id", "season", "date", "team", "opponent", "team_pts", "opp_pts"]]
away["is_home"] = 0

long_df = pd.concat([home, away], ignore_index=True)
long_df["win"] = (long_df["team_pts"] > long_df["opp_pts"]).astype(int)
long_df = long_df.sort_values(["team", "season", "date"]).reset_index(drop=True)

# ---- Core rolling pre-game features (shifted so current game's result is excluded) ----
def add_rolling(g):
    g = g.sort_values("date")
    shifted_pts = g["team_pts"].shift(1)
    shifted_opp_pts = g["opp_pts"].shift(1)
    shifted_win = g["win"].shift(1)

    g["games_played_so_far"] = np.arange(len(g))
    g["season_avg_pts_for"] = shifted_pts.expanding().mean()
    g["season_avg_pts_against"] = shifted_opp_pts.expanding().mean()
    g["season_win_pct"] = shifted_win.expanding().mean()

    g["last10_avg_pts_for"] = shifted_pts.rolling(10, min_periods=1).mean()
    g["last10_avg_pts_against"] = shifted_opp_pts.rolling(10, min_periods=1).mean()
    g["last10_win_pct"] = shifted_win.rolling(10, min_periods=1).mean()

    # current win/loss streak entering this game (positive = win streak, negative = loss streak)
    streak = []
    cur = 0
    for w in shifted_win:
        if pd.isna(w):
            streak.append(np.nan)
        else:
            if w == 1:
                cur = cur + 1 if cur >= 0 else 1
            else:
                cur = cur - 1 if cur <= 0 else -1
            streak.append(cur)
    g["win_streak"] = streak

    # rest days since previous game
    g["rest_days"] = g["date"].diff().dt.days

    return g

long_df = long_df.groupby(["team", "season"], group_keys=False).apply(add_rolling, include_groups=False).join(
    long_df[["team", "season"]]
)

# ---- New feature 1: point differential (derived, no extra data needed) ----
long_df["season_point_diff"] = long_df["season_avg_pts_for"] - long_df["season_avg_pts_against"]
long_df["last10_point_diff"] = long_df["last10_avg_pts_for"] - long_df["last10_avg_pts_against"]

# ---- New feature 2: back-to-back flag ----
long_df["back_to_back"] = (long_df["rest_days"] <= 1).astype("Int64")

# ---- New feature 3: home/away-specific win pct (split by game location) ----
def add_split_win_pct(g):
    g = g.sort_values("date")
    g["split_win_pct"] = g["win"].shift(1).expanding().mean()
    return g

long_df = long_df.groupby(["team", "season", "is_home"], group_keys=False).apply(
    add_split_win_pct, include_groups=False
).join(long_df[["team", "season", "is_home"]])

# ---- New feature 4: strength of schedule (avg win pct of opponents faced so far) ----
# For each past game, look up what the OPPONENT's season_win_pct was entering that same game
# (already a leakage-free, pre-game value on the opponent's own row), then average those
# over this team's games prior to the current one.
opp_lookup = long_df[["game_id", "team", "season_win_pct"]].rename(
    columns={"team": "opponent", "season_win_pct": "opp_win_pct_at_time"}
)
long_df = long_df.merge(opp_lookup, on=["game_id", "opponent"], how="left")

def add_sos(g):
    g = g.sort_values("date")
    g["strength_of_schedule"] = g["opp_win_pct_at_time"].shift(1).expanding().mean()
    return g

long_df = long_df.groupby(["team", "season"], group_keys=False).apply(add_sos, include_groups=False).join(
    long_df[["team", "season"]]
)
# Cold-start fallback: a team's first couple of games can have no valid opponent
# win-pct history yet (e.g. two season-openers played against each other). Fill
# with a neutral .500 opponent as the prior rather than leaving it undefined.
long_df["strength_of_schedule"] = long_df["strength_of_schedule"].fillna(0.5)
long_df["split_win_pct"] = long_df["split_win_pct"].fillna(0.5)

# ---- New feature 5: head-to-head net record vs this exact opponent, this season ----
def add_h2h(g):
    g = g.sort_values("date")
    shifted_win = g["win"].shift(1)
    wins_so_far = shifted_win.expanding().sum()
    games_so_far = shifted_win.expanding().count()
    g["h2h_net_record"] = (2 * wins_so_far - games_so_far).fillna(0)
    g["h2h_meetings_so_far"] = games_so_far.fillna(0)
    return g

long_df = long_df.groupby(["team", "season", "opponent"], group_keys=False).apply(
    add_h2h, include_groups=False
).join(long_df[["team", "season", "opponent"]])

# Drop each team's first game of the season (no history yet -> can't predict from nothing)
long_df = long_df[long_df["games_played_so_far"] > 0].reset_index(drop=True)

long_df.to_csv(OUT_LONG, index=False)

# ---- Wide format: one row per game, home_*/away_* features + targets ----
feature_cols = [
    "season_avg_pts_for", "season_avg_pts_against", "season_win_pct",
    "season_point_diff", "last10_avg_pts_for", "last10_avg_pts_against",
    "last10_win_pct", "last10_point_diff", "win_streak", "rest_days",
    "back_to_back", "split_win_pct", "strength_of_schedule",
    "h2h_net_record", "h2h_meetings_so_far", "games_played_so_far",
]

home_feat = long_df[long_df["is_home"] == 1][
    ["game_id", "season", "date", "team", "opponent", "team_pts", "opp_pts"] + feature_cols
].rename(columns={"team": "home_team", "opponent": "away_team",
                   "team_pts": "home_pts", "opp_pts": "away_pts",
                   **{c: f"home_{c}" for c in feature_cols}})

away_feat = long_df[long_df["is_home"] == 0][
    ["game_id"] + feature_cols
].rename(columns={c: f"away_{c}" for c in feature_cols})

wide = home_feat.merge(away_feat, on="game_id", how="inner")
wide["home_win"] = (wide["home_pts"] > wide["away_pts"]).astype(int)

wide = wide.sort_values("date").reset_index(drop=True)
wide.to_csv(OUT_WIDE, index=False)

print("long shape:", long_df.shape)
print("wide shape:", wide.shape)
print(wide.head(3).to_string())
print("\nnull counts in wide feature cols:")
check_cols = [f"home_{c}" for c in feature_cols] + [f"away_{c}" for c in feature_cols]
print(wide[check_cols].isna().sum())
