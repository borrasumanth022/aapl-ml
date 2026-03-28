Run the aapl_ml single-ticker pipeline for $ARGUMENTS (default: full pipeline AAPL, all steps).

Read CLAUDE.local.md for {PYTHON_EXE}. Steps to run in order:

1. `{PYTHON_EXE} src/01_fetch_data.py` — download AAPL OHLCV from Yahoo Finance
2. `{PYTHON_EXE} src/02_features.py` — compute 36 technical features (7,600+ rows)
3. `{PYTHON_EXE} src/03_labels.py` — compute 20 label columns (5 horizons × 4 types)
4. `{PYTHON_EXE} src/04_events.py` — collect FRED macro + Apple product events (1,320+ macro, 85 earnings, 64 product events)
5. `{PYTHON_EXE} src/05_event_features.py` — merge events → 57-column feature matrix
6. `{PYTHON_EXE} src/06_train.py` — train XGBoost champion model

Before step 2, run `python .claude/hooks/check-lookahead.py src/02_features.py` and confirm no CRITICAL findings.
After each step, run `python .claude/hooks/validate-data.py` to verify output parquets.

Expected outputs:
- After step 1: `data/raw/aapl_daily_raw.parquet` (≥6,000 rows)
- After step 2: `data/processed/aapl_features.parquet` (≥7,000 rows, 36 cols)
- After step 3: `data/processed/aapl_labeled.parquet` (≥7,000 rows, 57 cols)
- After step 5: `data/processed/aapl_with_events.parquet` (≥7,000 rows, ≥90 cols)
- After step 6: `models/xgb_best_interactions.pkl` + `data/processed/aapl_predictions_interactions.parquet`

Report after training:
- OOS Accuracy vs naive baseline (always-Bull = 37.50% for dir_1w)
- Macro F1 (target: ≥ 0.375, current champion = 0.375)
- Per-class recall: Bear / Sideways / Bull

If $ARGUMENTS specifies `--step N`, run only that step.
If $ARGUMENTS specifies `--force`, overwrite existing files.
