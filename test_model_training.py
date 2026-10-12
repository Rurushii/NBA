"""Tests for training boundaries and reusable model training."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from model_training import FEATURES, chronological_split, train_and_evaluate


class ModelTrainingTests(unittest.TestCase):
    def test_cutoff_date_stays_in_test_set(self):
        data = pd.DataFrame({"date": [
            "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04",
            "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08",
            "2025-10-08", "2025-10-09",
        ]})
        train, test = chronological_split(data.iloc[::-1])
        self.assertEqual(len(train), 7)
        self.assertEqual(len(test), 3)
        self.assertLess(train["date"].max(), test["date"].min())

    def test_invalid_dates_and_empty_partitions_are_rejected(self):
        for dates in [[], ["2025-10-01"], ["2025-10-01"] * 5,
                      ["2025-10-01", None]]:
            with self.subTest(dates=dates):
                with self.assertRaises(ValueError):
                    chronological_split(pd.DataFrame({"date": dates}))

    def test_training_preserves_input_and_fits_scaler_on_training_only(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.normal(size=(60, len(FEATURES))), columns=FEATURES)
        data["date"] = pd.date_range("2025-10-01", periods=60)
        data["home_win"] = np.arange(60) % 2
        data["home_pts"] = rng.integers(90, 130, size=60)
        data["away_pts"] = rng.integers(90, 130, size=60)
        data.loc[48:, "home_rest_days"] += 1000
        original = data.copy(deep=True)
        results, artifacts = train_and_evaluate(data)
        assert_frame_equal(data, original)
        np.testing.assert_allclose(
            artifacts["scaler.joblib"].mean_, data.iloc[:48][FEATURES].mean(),
        )
        self.assertEqual(set(artifacts), {
            "scaler.joblib", "model_logreg.joblib", "model_rf_cls.joblib",
            "model_rf_home_pts.joblib", "model_rf_away_pts.joblib",
        })
        self.assertEqual(set(results["regression"]), {"home_pts", "away_pts"})

    def test_imports_do_not_run_training(self):
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
                [sys.executable, "-c", "import model_training; import train_models"],
                cwd=directory, env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()