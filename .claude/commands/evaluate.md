# /project:evaluate -- Evaluate aapl_ml models

**Usage:**
- /project:evaluate -- compare all saved model predictions
- /project:evaluate holdout -- evaluate on 2024-01-01+ holdout only
- /project:evaluate shap -- re-run SHAP on champion

## Instructions

1. Read CLAUDE.local.md to get PYTHON_EXE.

2. Load predictions from data/processed/:
   - aapl_predictions_phase3_champion.parquet (CURRENT champion)
   - aapl_predictions_best.parquet (Phase 2)
   - aapl_predictions_lgbm.parquet
   - aapl_predictions_lstm.parquet
   - aapl_predictions_lstm_raw.parquet

3. Print the full model leaderboard:
     Model                             Acc       F1     Bear     Side     Bull
     XGB P3 champion (57 feat)      38.30%   0.375   30.6%   39.9%   42.0%  <- CHAMPION
     XGB P2 best (weighted dir_1w)  38.35%   0.367   23.1%   45.5%   41.8%
     LGBM (57 feat)                 36.07%   0.353   29.6%   34.0%   42.8%
     LSTM engineered (CV)              --    0.297     --       --       --
     LSTM raw OHLCV (CV)               --    0.311     --       --       --
     Naive (always Bull)            52.90%   0.230    0.0%    0.0%  100.0%

4. Holdout evaluation (2024-01-01 to present): all models on holdout rows.

5. Check calibration and directional accuracy.

