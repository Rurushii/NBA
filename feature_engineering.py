import numpy as np
import pandas as pd

FEATURE_COLS = [
    "season_avg_pts_for", "season_avg_pts_against", "season_win_pct",
    "season_point_diff", "last10_avg_pts_for", "last10_avg_pts_against",
    "last10_win_pct", "last10_point_diff", "win_streak", "rest_days",
    "back_to_back", "split_win_pct", "strength_of_schedule",
    "h2h_net_record", "h2h_meetings_so_far", "games_played_so_far",
]


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

    # Streak entering the game: positive for wins, negative for losses.
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

    g["rest_days"] = g["date"].diff().dt.days
    return g


def add_split_win_pct(g):
    g = g.sort_values("date")
    g["split_win_pct"] = g["win"].shift(1).expanding().mean()
    return g


def add_sos(g):
    g = g.sort_values("date")
    g["strength_of_schedule"] = g["opp_win_pct_at_time"].shift(1).expanding().mean()
    return g


def add_h2h(g):
    g = g.sort_values("date")
    shifted_win = g["win"].shift(1)
    wins_so_far = shifted_win.expanding().sum()
    games_so_far = shifted_win.expanding().count()
    g["h2h_net_record"] = (2 * wins_so_far - games_so_far).fillna(0)
    g["h2h_meetings_so_far"] = games_so_far.fillna(0)
    return g


def build_game_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return team-game and game feature tables without changing the input.

    The caller supplies raw game results using the existing CSV schema.
    Feature definitions and ordering preserve the original pipeline.
    """
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["game_id"] = df.index

    # One row per team per game.
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

    long_df = long_df.groupby(["team", "season"], group_keys=False).apply(
        add_rolling, include_groups=False
    ).join(long_df[["team", "season"]])

    long_df["season_point_diff"] = (
        long_df["season_avg_pts_for"] - long_df["season_avg_pts_against"]
    )
    long_df["last10_point_diff"] = (
        long_df["last10_avg_pts_for"] - long_df["last10_avg_pts_against"]
    )
    long_df["back_to_back"] = (long_df["rest_days"] <= 1).astype("Int64")

    # Win percentage at the same game location.
    long_df = long_df.groupby(
        ["team", "season", "is_home"], group_keys=False
    ).apply(
        add_split_win_pct, include_groups=False
    ).join(long_df[["team", "season", "is_home"]])

    # Opponents' pregame records, averaged over previously played games.
    opp_lookup = long_df[["game_id", "team", "season_win_pct"]].rename(
        columns={"team": "opponent", "season_win_pct": "opp_win_pct_at_time"}
    )
    long_df = long_df.merge(opp_lookup, on=["game_id", "opponent"], how="left")

    long_df = long_df.groupby(["team", "season"], group_keys=False).apply(
        add_sos, include_groups=False
    ).join(long_df[["team", "season"]])

    # Neutral priors when the relevant history is unavailable.
    long_df["strength_of_schedule"] = long_df["strength_of_schedule"].fillna(0.5)
    long_df["split_win_pct"] = long_df["split_win_pct"].fillna(0.5)

    long_df = long_df.groupby(
        ["team", "season", "opponent"], group_keys=False
    ).apply(
        add_h2h, include_groups=False
    ).join(long_df[["team", "season", "opponent"]])

    # Preserve the existing exclusion of each team's season opener.
    long_df = long_df[long_df["games_played_so_far"] > 0].reset_index(drop=True)

    # One row per game, with home/away features and targets.
    home_feat = long_df[long_df["is_home"] == 1][
        ["game_id", "season", "date", "team", "opponent", "team_pts", "opp_pts"]
        + FEATURE_COLS
    ].rename(columns={
        "team": "home_team", "opponent": "away_team",
        "team_pts": "home_pts", "opp_pts": "away_pts",
        **{c: f"home_{c}" for c in FEATURE_COLS},
    })

    away_feat = long_df[long_df["is_home"] == 0][
        ["game_id"] + FEATURE_COLS
    ].rename(columns={c: f"away_{c}" for c in FEATURE_COLS})

    wide = home_feat.merge(away_feat, on="game_id", how="inner")
    wide["home_win"] = (wide["home_pts"] > wide["away_pts"]).astype(int)
    wide = wide.sort_values("date").reset_index(drop=True)

    return long_df, wide