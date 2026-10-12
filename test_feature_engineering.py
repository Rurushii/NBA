import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal
from feature_engineering import FEATURE_COLS, build_game_features


class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame([
            ["2025-26", "2025-10-23", "Denver Nuggets", 100, "Boston Celtics", 110],
            ["2025-26", "2025-10-25", "Boston Celtics", 95, "Denver Nuggets", 105],
            ["2025-26", "2025-10-27", "Denver Nuggets", 115, "Boston Celtics", 120],
            ["2025-26", "2025-10-29", "Boston Celtics", 112, "Denver Nuggets", 108],
        ], columns=["season", "date", "away_team", "away_pts", "home_team", "home_pts"])

    def test_input_is_preserved(self):
        original = self.raw.copy(deep=True)
        long_df, wide = build_game_features(self.raw)
        assert_frame_equal(self.raw, original)
        self.assertEqual(len(long_df), 6)
        self.assertEqual(len(wide), 3)

    def test_features_match_known_history(self):
        _, wide = build_game_features(self.raw)
        game = wide.loc[wide["date"].eq(pd.Timestamp("2025-10-27"))].iloc[0]
        self.assertEqual(game["home_season_avg_pts_for"], 102.5)
        self.assertEqual(game["home_season_win_pct"], 0.5)
        self.assertEqual(game["home_win_streak"], -1)
        self.assertEqual(game["home_rest_days"], 2)
        self.assertEqual(game["home_split_win_pct"], 1.0)
        self.assertEqual(game["away_split_win_pct"], 0.0)
        self.assertEqual(game["home_h2h_meetings_so_far"], 2)
        self.assertEqual(game["home_h2h_net_record"], 0)

    def test_current_and_future_results_do_not_change_earlier_features(self):
        _, original = build_game_features(self.raw)
        changed = self.raw.copy()
        changed.loc[2:, ["home_pts", "away_pts"]] = [[80, 140], [150, 75]]
        _, recomputed = build_game_features(changed)
        columns = [f"{side}_{name}" for side in ["home", "away"] for name in FEATURE_COLS]
        cutoff = pd.Timestamp("2025-10-27")
        assert_frame_equal(
            original.loc[original["date"].le(cutoff), columns],
            recomputed.loc[recomputed["date"].le(cutoff), columns],
        )

    def test_imports_do_not_run_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-be-created"
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": str(Path(__file__).resolve().parent),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NBA_DATA_DIR": str(Path(directory) / "missing-data"),
                "NBA_OUTPUT_DIR": str(output),
            })
            result = subprocess.run(
                [sys.executable, "-c", "import feature_engineering; import build_features"],
                cwd=directory, env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()