import pandas as pd

from feature_engineering import FEATURE_COLS, build_game_features


def main() -> None:
    # Resolve configured filesystem locations only when running the command.
    from project_paths import data_path, output_path

    raw = pd.read_csv(data_path("nba_games_raw.csv"))
    long_df, wide = build_game_features(raw)
    long_df.to_csv(output_path("nba_team_game_features.csv"), index=False)
    wide.to_csv(output_path("nba_games_features.csv"), index=False)

    print("long shape:", long_df.shape)
    print("wide shape:", wide.shape)
    print(wide.head(3).to_string())
    print("\nnull counts in wide feature cols:")
    check_cols = [f"home_{c}" for c in FEATURE_COLS] + [f"away_{c}" for c in FEATURE_COLS]
    print(wide[check_cols].isna().sum())


if __name__ == "__main__":
    main()