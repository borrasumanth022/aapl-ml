# Data Conventions -- aapl_ml

## File format
All data is Parquet. Never CSV. Never SQLite.

## Naming schema
    data/raw/         aapl_daily_raw.parquet
    data/processed/   aapl_features.parquet
                      aapl_labeled.parquet
                      aapl_with_events.parquet
                      aapl_predictions*.parquet
    models/           feature_list.json
                      xgb_phase3_champion.pkl  (CURRENT champion)
                      xgb_best_interactions.pkl
                      lgbm_dir_1w.pkl
                      lstm_dir_1w.pt
                      lstm_raw_ohlcv.pt
                      shap_summary*.csv

## Date column
Always named date, type datetime64[D].
Normalize mixed datetime64[ms]/datetime64[us] before joins.

## NaN sentinels
- days_since_last_earnings = 90 for pre-2005 rows
- days_since_last_product_event = 180 for pre-2000 rows
- Never use 0 or -1 as sentinels (they are valid feature values)

## Parquet write standard
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")

## Skip-if-exists pattern
    if path.exists():
        print(f"Skipping {path.name} -- exists. Delete to recompute.")
        return

