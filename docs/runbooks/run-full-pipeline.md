# Runbook: Run Full Pipeline from Scratch

This runbook rebuilds every artefact from raw data to trained models.
Run from the project root (`c:\Users\borra\OneDrive\Desktop\ML Projects\aapl_ml`).

**Python**: `C:\Users\borra\anaconda3\python.exe`

---

## Prerequisites
```bash
# Verify key packages
C:\Users\borra\anaconda3\python.exe -c "import yfinance, xgboost, shap, lightgbm; print('OK')"
```

---

## Phase 1 — Data Foundation

### Step 1: Fetch raw OHLCV data (~30s, requires internet)
```bash
C:\Users\borra\anaconda3\python.exe src/01_fetch_data.py
```
**Output**: `data/raw/aapl_daily_raw.parquet` (~7,800 rows)

### Step 2: Feature engineering (~10s)
```bash
C:\Users\borra\anaconda3\python.exe src/02_features.py
```
**Output**: `data/processed/aapl_features.parquet` (~7,600 rows × 57 cols)

### Step 3: Build labels (~5s)
```bash
C:\Users\borra\anaconda3\python.exe src/03_labels.py
```
**Output**: `data/processed/aapl_labeled.parquet` (~7,400 rows × 77 cols)

### Step 4: Feature selection (~2 min — computes correlation matrix)
```bash
C:\Users\borra\anaconda3\python.exe src/04_feature_selection.py
```
**Output**: `models/feature_list.json` (36 features)

---

## Phase 2 — Pattern Recognition

### Step 5: Baseline XGBoost on dir_1m (~3 min)
```bash
C:\Users\borra\anaconda3\python.exe src/05_train_baseline.py
```
**Output**: `models/xgb_dir_1m.pkl`, `data/processed/aapl_predictions.parquet`

### Step 6a: Experiment A — dir_1w target (~3 min)
```bash
C:\Users\borra\anaconda3\python.exe src/06a_exp_dir1w.py
```

### Step 6b: Experiment B — dir_1m weighted (~3 min)
```bash
C:\Users\borra\anaconda3\python.exe src/06b_exp_weighted.py
```

### Step 7: Phase 2 best model + SHAP (~5 min)
```bash
C:\Users\borra\anaconda3\python.exe src/07_best_model.py
```
**Output**: `models/xgb_dir_1w_weighted.pkl`, `data/processed/aapl_predictions_best.parquet`, `models/shap_summary.csv`

---

## Phase 3 — Event Linkage

### Step 8: Collect event data (~1 min, requires internet)
```bash
C:\Users\borra\anaconda3\python.exe src/08_events.py
```
**Output**: `data/processed/aapl_events.parquet` (~1,474 events)

### Step 9: Engineer event features (~2 min, requires internet for FRED)
```bash
C:\Users\borra\anaconda3\python.exe src/09_event_features.py
```
**Output**: `data/processed/aapl_with_events.parquet` (~7,400 rows × 93 cols)

### Step 10: Retrain with event features (~5 min)
```bash
C:\Users\borra\anaconda3\python.exe src/10_retrain_with_events.py
```

### Step 11: Interaction features + Phase 3 champion (~10 min)
```bash
C:\Users\borra\anaconda3\python.exe src/11_interaction_features.py
```
**Output**: `models/xgb_best_interactions.pkl` (F1=0.375), `data/processed/aapl_predictions_interactions.parquet`

---

## Phase 4 — Fusion Model (in progress)

### Step 12: LightGBM baseline
```bash
C:\Users\borra\anaconda3\python.exe src/12_lgbm.py
```

### Step 13: Ensemble meta-learner
```bash
C:\Users\borra\anaconda3\python.exe src/13_ensemble.py
```

---

## Validate outputs
```bash
C:\Users\borra\anaconda3\python.exe .claude/hooks/validate-data.py --verbose
C:\Users\borra\anaconda3\python.exe .claude/hooks/check-model-eval.py models/xgb_best_interactions.pkl
```

---

## Notes
- Steps 1, 8, 9 require internet access (yfinance / FRED HTTP)
- Steps 2–7, 10–13 are fully offline (read from parquet)
- Data files are gitignored — clone + run this runbook to reproduce
- Expected total time: ~35 minutes for Phases 1–3
