# NBA Game Outcome & Score Predictor

Predicts the winner and final score of any NBA matchup using a machine learning model trained on real game data — plus an interactive dashboard where you pick two teams and see the prediction live.

**Live demo:** [allendrjs.github.io/NBA/nba_predictor_dashboard.html](https://allendrjs.github.io/NBA/nba_predictor_dashboard.html) (once GitHub Pages is enabled — see below) · **Repo:** [github.com/allendrjs/NBA](https://github.com/allendrjs/NBA)

![accuracy](https://img.shields.io/badge/win%2Floss%20accuracy-74.4%25-brightgreen) ![baseline](https://img.shields.io/badge/vs%20baseline-54.9%25-lightgrey)

A random forest classifier predicts the winner, and two random forest
regressors estimate final scores from historical team statistics.

The bundled snapshot contains 2,317 game-result rows across the 2023-24,
2024-25, and 2025-26 season labels. Feature generation produces 2,268 eligible
examples; the current chronological split uses 1,811 training examples and
457 holdout examples.

The dataset is incomplete, and source provenance and game-type classification
remain unverified. See [DATA_AUDIT.md](DATA_AUDIT.md) for coverage and validation.

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

Use **Python 3.12.x**. The `.python-version` file records this target for tools
that support it; it does not install Python or change your system interpreter.

`requirements.txt` pins the pipeline dependencies, including scikit-learn 1.7.2
to match the saved models. Run the commands below from the repository root.
They use a local virtual environment without requiring activation.

### Windows Command Prompt

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip check

set "NBA_OUTPUT_DIR=outputs"
.venv\Scripts\python.exe build_features.py
.venv\Scripts\python.exe train_models.py
.venv\Scripts\python.exe predict_matchups.py
.venv\Scripts\python.exe build_dashboard.py
```

If you use PowerShell, prefix the virtual-environment commands with `.\`
(for example, `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`)
and replace the `set` line with `$env:NBA_OUTPUT_DIR = "outputs"`.

### Linux/macOS (Bash or Zsh)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check

export NBA_OUTPUT_DIR=outputs
.venv/bin/python build_features.py
.venv/bin/python train_models.py
.venv/bin/python predict_matchups.py
.venv/bin/python build_dashboard.py
```

The dashboard is written to `outputs/nba_predictor_dashboard.html`. Open it in
your browser. The `outputs` directory and `.venv` are ignored by Git.

### File locations

| Variable | Default | Purpose |
| --- | --- | --- |
| `NBA_DATA_DIR` | Repository root | Directory containing `nba_games_raw.csv` |
| `NBA_OUTPUT_DIR` | Repository root | Generated CSVs, models, JSON, and HTML |

The examples above use the bundled raw CSV and direct generated files to
`outputs`. Set `NBA_DATA_DIR` only when using a raw CSV in another directory.
Use `set "NBA_DATA_DIR=C:\path\to\nba-data"` in Command Prompt,
`$env:NBA_DATA_DIR = "C:\path\to\nba-data"` in PowerShell, or
`export NBA_DATA_DIR="/path/to/nba-data"` in Bash/Zsh.

Unset or empty values use the defaults. Relative directories are resolved from
the repository root. The output directory is created automatically; the input
CSV must already exist. Run all four stages in order with the same settings.
Later stages read their generated inputs from `NBA_OUTPUT_DIR`.

Without `NBA_OUTPUT_DIR`, generated files overwrite the existing artifacts in
the repository root. `.gitignore` does not stop Git tracking existing files.

### Optional notebook tools

Install notebook dependencies only if you want to work with
`nba_prediction_analysis.ipynb`. This also installs the pipeline dependencies.

Windows Command Prompt:

```bat
.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.venv\Scripts\jupyter.exe lab
```

Linux/macOS:

```bash
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/jupyter lab
```

Launch Jupyter from the repository root so the notebook can find
`nba_games_raw.csv`. Select the virtual environment's Python kernel. The
notebook uses its working directory, rather than the pipeline's path variables.

## Known limitations

- The dataset is incomplete: missing months affect records, rolling statistics,
  rest calculations, and head-to-head history.
- This is a team-level model; it does not account for injuries, trades,
  or specific lineups.
- Predictions use a historical data snapshot and are not updated live.
- Head-to-head history is limited because many team pairs have only a few
  recorded meetings per season.

See `nba_prediction_analysis.ipynb` for methodology and further discussion.


