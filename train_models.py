"""
Train and evaluate:
  1. Classification model -> predict home team win/loss
  2. Regression models -> predict home_pts and away_pts (final score)

Uses a chronological train/test split (train on earlier games, test on the
most recent ~20% of games) since this is time-series data - a random
shuffle split would leak future information into training.
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, log_loss, confusion_matrix,
                              mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
import joblib

DATA = "/sessions/jolly-sleepy-wozniak/mnt/outputs/nba_games_features.csv"
df = pd.read_csv(DATA, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

FEATURES = [
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

split_idx = int(len(df) * 0.8)
split_date = df.iloc[split_idx]["date"]
train = df[df["date"] < split_date].copy()
test = df[df["date"] >= split_date].copy()
print(f"Train: {len(train)} games (through {train['date'].max().date()})")
print(f"Test:  {len(test)} games ({test['date'].min().date()} to {test['date'].max().date()})")

X_train, X_test = train[FEATURES], test[FEATURES]

results = {}

# ============ CLASSIFICATION: home win/loss ============
y_train_cls, y_test_cls = train["home_win"], test["home_win"]

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000).fit(X_train_s, y_train_cls)
rf_cls = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10,
                                 random_state=42).fit(X_train, y_train_cls)

# Baseline: home team always wins (classic home-court-advantage heuristic)
baseline_preds = np.ones(len(y_test_cls))
baseline_acc = accuracy_score(y_test_cls, baseline_preds)

for name, model, X_te in [("logistic_regression", logreg, X_test_s), ("random_forest", rf_cls, X_test)]:
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:, 1]
    results[f"classification_{name}"] = {
        "accuracy": accuracy_score(y_test_cls, preds),
        "precision": precision_score(y_test_cls, preds),
        "recall": recall_score(y_test_cls, preds),
        "f1": f1_score(y_test_cls, preds),
        "log_loss": log_loss(y_test_cls, probs),
        "confusion_matrix": confusion_matrix(y_test_cls, preds).tolist(),
    }

results["classification_baseline_home_always_wins"] = {"accuracy": baseline_acc}

# pick best classifier by accuracy
best_cls_name = max(
    ["logistic_regression", "random_forest"],
    key=lambda n: results[f"classification_{n}"]["accuracy"]
)
best_cls_model = {"logistic_regression": logreg, "random_forest": rf_cls}[best_cls_name]
results["best_classifier"] = best_cls_name

# ============ REGRESSION: final score for each team ============
reg_results = {}
for target in ["home_pts", "away_pts"]:
    y_train_r, y_test_r = train[target], test[target]

    lin = LinearRegression().fit(X_train_s, y_train_r)
    rf_r = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=10,
                                  random_state=42).fit(X_train, y_train_r)

    # naive baseline: predict team's own season average points-for entering the game
    naive_col = "home_season_avg_pts_for" if target == "home_pts" else "away_season_avg_pts_for"
    naive_preds = test[naive_col]

    reg_results[target] = {}
    for name, model, X_te in [("linear_regression", lin, X_test_s), ("random_forest", rf_r, X_test)]:
        preds = model.predict(X_te)
        reg_results[target][name] = {
            "mae": mean_absolute_error(y_test_r, preds),
            "rmse": mean_squared_error(y_test_r, preds) ** 0.5,
            "r2": r2_score(y_test_r, preds),
        }
    reg_results[target]["naive_season_avg_baseline"] = {
        "mae": mean_absolute_error(y_test_r, naive_preds),
        "rmse": mean_squared_error(y_test_r, naive_preds) ** 0.5,
        "r2": r2_score(y_test_r, naive_preds),
    }

results["regression"] = reg_results

# Save best models + scaler for the dashboard/notebook to reuse
joblib.dump(scaler, "/sessions/jolly-sleepy-wozniak/mnt/outputs/scaler.joblib")
joblib.dump(logreg, "/sessions/jolly-sleepy-wozniak/mnt/outputs/model_logreg.joblib")
joblib.dump(rf_cls, "/sessions/jolly-sleepy-wozniak/mnt/outputs/model_rf_cls.joblib")
joblib.dump(RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42)
            .fit(X_train, train["home_pts"]), "/sessions/jolly-sleepy-wozniak/mnt/outputs/model_rf_home_pts.joblib")
joblib.dump(RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42)
            .fit(X_train, train["away_pts"]), "/sessions/jolly-sleepy-wozniak/mnt/outputs/model_rf_away_pts.joblib")

# Feature importance from the random forest classifier (useful narrative for the writeup)
importances = pd.Series(rf_cls.feature_importances_, index=FEATURES).sort_values(ascending=False)
results["feature_importance_rf_classifier"] = importances.round(4).to_dict()

with open("/sessions/jolly-sleepy-wozniak/mnt/outputs/model_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(json.dumps(results, indent=2, default=str))
