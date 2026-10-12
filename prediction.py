import pandas as pd

from feature_engineering import FEATURE_COLS
from model_training import FEATURES

TEAM_FEATURES = [
    name for name in FEATURE_COLS
    if name not in {"h2h_net_record", "h2h_meetings_so_far", "games_played_so_far"}
]


class MatchupPredictor:
    def __init__(self, long_df, artifacts):
        if long_df.empty:
            raise ValueError("Team history must not be empty.")
        history = long_df.copy()
        history["date"] = pd.to_datetime(history["date"])
        self.artifacts = dict(artifacts)
        self.latest = (
            history.sort_values("date").groupby("team", as_index=False)
            .tail(1).set_index("team")
        )
        self.teams = sorted(self.latest.index.tolist())
        self.current_season = self.latest["season"].mode()[0]
        self.h2h_pairs = {}
        season_games = history[history["season"] == self.current_season]
        for (team, opponent), games in season_games.groupby(["team", "opponent"]):
            count = len(games)
            self.h2h_pairs[(team, opponent)] = {
                "h2h_net_record": 2 * int(games["win"].sum()) - count,
                "h2h_meetings_so_far": count,
            }

    def get_h2h(self, team, opponent):
        return self.h2h_pairs.get((team, opponent), {
            "h2h_net_record": 0, "h2h_meetings_so_far": 0,
        })

    def build_row(self, home_team, away_team):
        if home_team == away_team:
            raise ValueError("Choose two different teams.")
        for team in (home_team, away_team):
            if team not in self.latest.index:
                raise ValueError(f"Unknown team: {team}")
        row = {}
        for side, team, opponent in [
            ("home", home_team, away_team),
            ("away", away_team, home_team),
        ]:
            stats = self.latest.loc[team]
            row.update({f"{side}_{name}": stats[name] for name in TEAM_FEATURES})
            row.update({
                f"{side}_{name}": value
                for name, value in self.get_h2h(team, opponent).items()
            })
        return pd.DataFrame([row])[FEATURES]

    def predict(self, home_team, away_team):
        inputs = self.build_row(home_team, away_team)
        models = self.artifacts
        scaled = models["scaler.joblib"].transform(inputs)
        rf_prob = float(models["model_rf_cls.joblib"].predict_proba(inputs)[0, 1])
        lr_prob = float(models["model_logreg.joblib"].predict_proba(scaled)[0, 1])
        home_pts = float(models["model_rf_home_pts.joblib"].predict(inputs)[0])
        away_pts = float(models["model_rf_away_pts.joblib"].predict(inputs)[0])
        h2h = self.get_h2h(home_team, away_team)
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_win_prob": round(rf_prob, 4),
            "home_win_prob_logreg": round(lr_prob, 4),
            "predicted_home_pts": round(home_pts, 1),
            "predicted_away_pts": round(away_pts, 1),
            "predicted_winner": home_team if rf_prob >= 0.5 else away_team,
            "h2h_meetings_this_season": h2h["h2h_meetings_so_far"],
            "h2h_net_record_home_perspective": h2h["h2h_net_record"],
        }

    def team_summaries(self):
        summaries = {}
        for team in self.teams:
            row = self.latest.loc[team]
            summaries[team] = {
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
        return summaries

    def predict_all(self):
        return {
            f"{home}|{away}": self.predict(home, away)
            for home in self.teams
            for away in self.teams
            if home != away
        }