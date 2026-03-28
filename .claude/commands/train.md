# /project:train -- Train an aapl_ml model

**Usage:**
- /project:train xgb -- train XGBoost (champion architecture)
- /project:train lgbm -- train LightGBM (reference, known to underperform)
- /project:train lstm -- train LSTM on engineered 57 features
- /project:train lstm_raw -- train LSTM on raw OHLCV (5 features)

## Instructions

1. Read CLAUDE.local.md to get PYTHON_EXE.

2. Map model type to script:
   - xgb -> src/11_interaction_features.py
   - lgbm -> src/12_lgbm.py
   - lstm -> src/14_lstm.py
   - lstm_raw -> src/15_lstm_raw.py

3. After training, print the full comparison table:
     Model                          Acc       F1     Bear     Side     Bull
     XGB P3 champion (57 feat)   38.30%   0.375   30.6%   39.9%   42.0%  <- CHAMPION
     New model                   xx.xx%   0.xxx   xx.x%   xx.x%   xx.x%
     Naive (always Bull)         52.90%   0.230    0.0%    0.0%  100.0%

4. CHAMPION decision rule:
   - New champion if macro F1 > 0.375
   - Also check: Sideways recall > 30%, Bear recall > 20%
   - If F1 neutral but Bear recall improves > 5pp, note it but keep champion

5. If new champion: save as models/xgb_phase3_champion.pkl and run SHAP.

6. Always report SHAP top-5 features per class.

