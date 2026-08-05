# NBA Game Outcome & Score Predictor

Predicts the winner and final score of any NBA matchup using a machine learning model trained on real game data — plus an interactive dashboard where you pick two teams and see the prediction live.

**Live demo:** [allendrjs.github.io/NBA/nba_predictor_dashboard.html](https://allendrjs.github.io/NBA/nba_predictor_dashboard.html) (once GitHub Pages is enabled — see below) · **Repo:** [github.com/allendrjs/NBA](https://github.com/allendrjs/NBA)

![accuracy](https://img.shields.io/badge/win%2Floss%20accuracy-74.4%25-brightgreen) ![baseline](https://img.shields.io/badge/vs%20baseline-54.9%25-lightgrey)

## What this is

A random forest classifier predicts which team wins an NBA game, and two random forest regressors predict each team's final point total — all using only information available *before* tip-off (no box scores, no hindsight). The model is trained on 2,268 real games from the 2023-24 through 2025-26 NBA regular seasons, scraped from basketball-reference.com.

- **Win/loss accuracy: 74.4%**, vs. 54.9% for the naive "home team always wins" baseline
- **Final score MAE: ~9.6 points**, vs. ~10.5 for a naive "predict the season average" baseline

## Try it

Open [`nba_predictor_dashboard.html`](./nba_predictor_dashboard.html) directly in any browser — it's a single self-contained file, no server or install needed. Pick a home team and an away team from the dropdowns to see the predicted winner, win probability, projected score, and both teams' current form (record, scoring margin, home/away split, strength of schedule, head-to-head history).

## How it works

Every feature is computed strictly from each team's games *before* the one being predicted, to avoid leaking the outcome into the input:

- season-to-date and last-10-game win percentage, scoring average, and point differential
- home/away-specific win percentage (a team's home record and road record are tracked separately)
- strength of schedule (average quality of opponents faced so far)
- current win/loss streak, rest days, and a back-to-back flag
- head-to-head record against this specific opponent, this season

The model is evaluated on a **chronological** train/test split (train on the earliest ~80% of games by date, test on the most recent ~20%) rather than a random shuffle, since shuffling would let the model train on future games to predict the past.

Full methodology, data quality checks, charts, and an honest discussion of limitations are in [`nba_prediction_analysis.ipynb`](./nba_prediction_analysis.ipynb).

## Repo structure

```
nba_predictor_dashboard.html    interactive dashboard (the deliverable — open this)
nba_prediction_analysis.ipynb   full analysis notebook: data cleaning, feature
                                 engineering, model training, evaluation, charts
nba_games_raw.csv               raw scraped data: one row per game (date, teams, score)
nba_games_features.csv          engineered features, one row per game
nba_team_game_features.csv      engineered features, one row per team-per-game
build_features.py               feature engineering pipeline
train_models.py                 model training + evaluation
predict_matchups.py             precomputes predictions for all 870 possible matchups
build_dashboard.py              generates the dashboard HTML from model outputs
model_results.json              evaluation metrics (accuracy, MAE, feature importance)
*.joblib                        trained model files (scikit-learn)
```

## Running it yourself

```bash
pip install pandas numpy scikit-learn joblib
python3 build_features.py      # raw CSV -> engineered features
python3 train_models.py        # trains + evaluates models, saves .joblib files
python3 predict_matchups.py    # precomputes all matchup predictions
python3 build_dashboard.py     # generates the dashboard HTML
```

## Known limitations

- ~1,400 games are missing from the raw dataset because a handful of monthly schedule pages failed to load during collection — excluded rather than backfilled with guessed values.
- This is a team-level model only; it doesn't account for injuries, trades, or specific lineups.
- Head-to-head history is one of the weakest features, since most team pairs only meet 1-3 times per season — not enough of a sample to be highly predictive on its own.

See the notebook's conclusion section for the full discussion, including what I'd add with more time (a player-availability flag would likely be the single biggest improvement).
