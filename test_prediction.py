"""Tests for prediction inputs, model routing, and safe imports."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from model_training import FEATURES
from prediction import MatchupPredictor, TEAM_FEATURES


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.history = pd.DataFrame([
            {
                **{name: 10.0 for name in TEAM_FEATURES},
                "team": "Boston Celtics", "opponent": "Denver Nuggets",
                "season": "2025-26", "date": "2025-10-25", "win": 1,
                "games_played_so_far": 2,
            },
            {
                **{name: 20.0 for name in TEAM_FEATURES},
                "team": "Denver Nuggets", "opponent": "Boston Celtics",
                "season": "2025-26", "date": "2025-10-25", "win": 0,
                "games_played_so_far": 2,
            },
        ])
        self.models = {name: Mock() for name in (
            "scaler.joblib", "model_logreg.joblib", "model_rf_cls.joblib",
            "model_rf_home_pts.joblib", "model_rf_away_pts.joblib",
        )}
        self.models["scaler.joblib"].transform.return_value = np.zeros((1, 30))
        self.models["model_rf_cls.joblib"].predict_proba.return_value = np.array([[0.6, 0.4]])
        self.models["model_logreg.joblib"].predict_proba.return_value = np.array([[0.3, 0.7]])
        self.models["model_rf_home_pts.joblib"].predict.return_value = np.array([101.25])
        self.models["model_rf_away_pts.joblib"].predict.return_value = np.array([108.75])
        self.predictor = MatchupPredictor(self.history, self.models)

    def test_feature_order_and_home_away_mapping(self):
        original = self.history.copy(deep=True)
        row = self.predictor.build_row("Boston Celtics", "Denver Nuggets")
        self.assertEqual(list(row.columns), FEATURES)
        self.assertEqual(row.loc[0, "home_season_avg_pts_for"], 10.0)
        self.assertEqual(row.loc[0, "away_season_avg_pts_for"], 20.0)
        self.assertEqual(row.loc[0, "home_h2h_net_record"], 1)
        self.assertEqual(row.loc[0, "away_h2h_net_record"], -1)
        assert_frame_equal(self.history, original)

    def test_prediction_routes_scaled_and_unscaled_inputs(self):
        result = self.predictor.predict("Boston Celtics", "Denver Nuggets")
        self.assertEqual(result["home_win_prob"], 0.4)
        self.assertEqual(result["home_win_prob_logreg"], 0.7)
        self.assertEqual(result["predicted_winner"], "Denver Nuggets")
        self.assertEqual(result["predicted_home_pts"], 101.2)
        self.assertEqual(result["predicted_away_pts"], 108.8)
        scaled = self.models["model_logreg.joblib"].predict_proba.call_args.args[0]
        self.assertIs(scaled, self.models["scaler.joblib"].transform.return_value)
        raw = self.models["model_rf_cls.joblib"].predict_proba.call_args.args[0]
        self.assertEqual(raw.loc[0, "home_season_avg_pts_for"], 10.0)
        self.assertEqual(set(self.predictor.predict_all()), {
            "Boston Celtics|Denver Nuggets", "Denver Nuggets|Boston Celtics",
        })

    def test_invalid_matchups_are_rejected(self):
        for away in ("Boston Celtics", "Unknown Team"):
            with self.subTest(away=away), self.assertRaises(ValueError):
                self.predictor.predict("Boston Celtics", away)

    def test_imports_do_not_load_models_or_write_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-be-created"
            env = os.environ.copy()
            paths = [str(Path(__file__).resolve().parent), env.get("PYTHONPATH", "")]
            env.update({
                "PYTHONPATH": os.pathsep.join(path for path in paths if path),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NBA_OUTPUT_DIR": str(output),
            })
            result = subprocess.run(
                [sys.executable, "-c", "import prediction; import predict_matchups"],
                cwd=directory, env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()