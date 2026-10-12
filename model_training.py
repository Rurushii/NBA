import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from feature_engineering import FEATURE_COLS

FEATURES = [
    f"{side}_{name}"
    for side in ("home", "away")
    for name in FEATURE_COLS
    if name != "games_played_so_far"
]


def chronological_split(data):
    """Keep games on the cutoff date together in the test partition."""
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if df["date"].isna().any():
        raise ValueError("Game dates must not be missing.")
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("At least two games are required for training and testing.")
    split_date = df.iloc[int(len(df) * 0.8)]["date"]
    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()
    if train.empty or test.empty:
        raise ValueError("The chronological split needs nonempty train and test sets.")
    return train, test


def train_and_evaluate(data):
    """Return evaluation results and fitted artifacts using the existing settings."""
    train, test = chronological_split(data)
    X_train, X_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train["home_win"], test["home_win"]
    if set(y_train.unique()) != {0, 1} or set(y_test.unique()) != {0, 1}:
        raise ValueError("Both train and test sets must contain home wins and losses (0/1).")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
    rf_cls = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42,
    ).fit(X_train, y_train)

    results = {}
    artifacts = {
        "scaler.joblib": scaler,
        "model_logreg.joblib": logreg,
        "model_rf_cls.joblib": rf_cls,
    }
    for name, model, inputs in [
        ("logistic_regression", logreg, X_test_s),
        ("random_forest", rf_cls, X_test),
    ]:
        preds = model.predict(inputs)
        probs = model.predict_proba(inputs)[:, 1]
        results[f"classification_{name}"] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "log_loss": log_loss(y_test, probs),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }
    results["classification_baseline_home_always_wins"] = {
        "accuracy": accuracy_score(y_test, np.ones(len(y_test))),
    }
    results["best_classifier"] = max(
        ["logistic_regression", "random_forest"],
        key=lambda name: results[f"classification_{name}"]["accuracy"],
    )

    regression = {}
    for target in ("home_pts", "away_pts"):
        lin = LinearRegression().fit(X_train_s, train[target])
        forest = RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42,
        ).fit(X_train, train[target])
        # Save the already evaluated forest instead of fitting it a second time.
        artifacts[f"model_rf_{target}.joblib"] = forest
        naive_col = (
            "home_season_avg_pts_for" if target == "home_pts"
            else "away_season_avg_pts_for"
        )
        regression[target] = {}
        for name, preds in [
            ("linear_regression", lin.predict(X_test_s)),
            ("random_forest", forest.predict(X_test)),
            ("naive_season_avg_baseline", test[naive_col]),
        ]:
            regression[target][name] = {
                "mae": mean_absolute_error(test[target], preds),
                "rmse": mean_squared_error(test[target], preds) ** 0.5,
                "r2": r2_score(test[target], preds),
            }

    results["regression"] = regression
    importances = pd.Series(
        rf_cls.feature_importances_, index=FEATURES,
    ).sort_values(ascending=False)
    results["feature_importance_rf_classifier"] = importances.round(4).to_dict()
    return results, artifacts