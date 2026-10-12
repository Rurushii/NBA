import json

import joblib
import pandas as pd

from prediction import MatchupPredictor


def main():
    from project_paths import output_path

    history = pd.read_csv(output_path("nba_team_game_features.csv"), parse_dates=["date"])
    artifacts = {
        name: joblib.load(output_path(name))
        for name in (
            "scaler.joblib", "model_logreg.joblib", "model_rf_cls.joblib",
            "model_rf_home_pts.joblib", "model_rf_away_pts.joblib",
        )
    }
    predictor = MatchupPredictor(history, artifacts)
    matchups = predictor.predict_all()
    summaries = predictor.team_summaries()
    for filename, data in [
        ("matchups.json", matchups), ("team_summary.json", summaries),
    ]:
        with output_path(filename).open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=1)
    print(f"Wrote {len(matchups)} matchup predictions and {len(summaries)} team summaries")


if __name__ == "__main__":
    main()