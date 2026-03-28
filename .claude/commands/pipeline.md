# /project:pipeline -- Run the aapl_ml pipeline

**Usage:**
- /project:pipeline 1 -- Phase 1: data foundation (steps 01-03)
- /project:pipeline 2 -- Phase 2: pattern recognition (steps 04-07)
- /project:pipeline 3 -- Phase 3: event linkage (steps 08-11)
- /project:pipeline 11 -- run single script (e.g., script 11)

## Instructions

1. Read CLAUDE.local.md to get PYTHON_EXE.

2. Phase-to-script mapping:
   Phase 1 -- Data Foundation:
     01_fetch_data.py        Download AAPL OHLCV (7,856 rows)
     02_features.py          57-column feature matrix
     03_labels.py            20 label columns (5 horizons)

   Phase 2 -- Pattern Recognition:
     04_feature_selection.py 36 selected features
     05_train_baseline.py    XGBoost dir_1m baseline
     07_best_model.py        Best model + SHAP

   Phase 3 -- Event Linkage:
     08_events.py            1,474 events (yfinance/FRED/hardcoded)
     09_event_features.py    16 event features -> 93 columns
     11_interaction_features.py  57 features CHAMPION (F1=0.375)

   Phase 4 -- Fusion:
     12_lgbm.py              LightGBM (F1=0.353, below champion)
     13_ensemble.py          XGB+LGBM stacking (F1=0.347, below champion)

   Phase 5 -- Prediction Engine:
     14_lstm.py              LSTM 57-feat (CV F1=0.297, below champion)
     15_lstm_raw.py          LSTM raw OHLCV (CV F1=0.311, below champion)

3. Current champion: models/xgb_phase3_champion.pkl (dir_1w, 57 feat, F1=0.375)

4. If any script exits non-zero, stop and report. Do not continue.

5. After any training script, compare to champion F1=0.375 and report the delta.

