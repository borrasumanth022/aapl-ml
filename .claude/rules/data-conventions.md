# Data Conventions — aapl_ml

## Parquet file locations
- Raw: `data/raw/`
- Processed: `data/processed/`
- Key files:
  - `aapl_features.parquet` — 7,657 rows, 57 cols, 1995-10-16 to 2026-03-20
  - `aapl_labeled.parquet` — ends 2025-03-19 (last 252 rows dropped for 1Y forward labels)
  - `aapl_with_events.parquet` — labeled + event features (≥7,000 rows, ≥90 cols)
  - `aapl_predictions_interactions.parquet` — OOS walk-forward predictions

## Prediction columns (required schema)
All prediction parquets must have exactly these columns:
- `actual` — true label (-1/0/1)
- `predicted` — model prediction (-1/0/1)
- `correct` — bool, `actual == predicted`
- `prob_bear` — probability of Bear class
- `prob_sideways` — probability of Sideways class
- `prob_bull` — probability of Bull class

## Date column
- Date column must be named `date` (not `Date` or `index`)
- Type: `datetime64[ns]`

## Column naming
- Use snake_case for all feature columns
- No spaces or special characters in column names
- Ratio features: `<numerator>_<denominator>_ratio` (e.g., `price_sma50_ratio`)
- Rolling features: `<feature>_<window>d` (e.g., `hvol_21d`, `hvol_63d`)
- Regime features: `<indicator>_regime` (e.g., `rate_vol_regime`)
- Event features: `<event>_proximity`, `<event>_surprise`

## Train/test split
- Walk-forward only — no random splits on time-series data
- Holdout cutoff: **2024-01-01** (do not touch until final evaluation)

## Row counts (sanity checks)
- `aapl_features.parquet`: ≥7,500 rows
- `aapl_with_events.parquet`: ≥7,000 rows, ≥90 columns
- Fewer rows than expected → re-run pipeline step 1 and check data source

## No data/model files in git
- `data/` and `models/` are gitignored
- Never stage `.parquet`, `.pkl`, or `.pt` files
