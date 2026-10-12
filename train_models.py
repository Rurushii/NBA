"""Command-line entry point for training and saving NBA models."""
import json

import joblib
import pandas as pd

from model_training import train_and_evaluate


def main():
    from project_paths import output_path

    data = pd.read_csv(output_path("nba_games_features.csv"), parse_dates=["date"])
    results, artifacts = train_and_evaluate(data)
    for filename, model in artifacts.items():
        joblib.dump(model, output_path(filename))
    with output_path("model_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()