# ML Standards — aapl_ml

## Label encoding (AAPL-specific — do NOT use market_ml encoding)
- **-1 = Bear**, **0 = Sideways**, **1 = Bull**
- This differs from market_ml which uses 0/1/2 — never mix encodings

## Walk-forward validation
- `TimeSeriesSplit(n_splits=5)` with expanding window — never KFold with shuffle=True
- Test indices must always be strictly after training indices
- No metrics computed on training folds

## Baselines (always report alongside model metrics)
- dir_1w naive baseline: **37.50% accuracy** (always-Bull)
- Random macro F1: **0.333**
- Current champion: F1=0.375, Acc=38.30% (xgb_best_interactions, 57 features)

## Metrics
- Always report both **OOS Accuracy** and **Macro F1**
- Also report per-class recall: Bear / Sideways / Bull
- F1 < 0.35 → do NOT accept as new champion

## Scalers and encoders
- `fit(X_train)` then `transform(X_test)` — never `fit_transform(X_full)`
- Scaler must be fit on training fold only within each walk-forward split

## Class imbalance
- `class_weight="balanced"` required on all classifiers

## Lookahead bias
- No `shift(-N)` where N > 0 on non-label columns
- No `fit_transform()` on full dataset
- No `.rolling()` windows that peek forward
- Run `python .claude/hooks/check-lookahead.py src/<file>.py` before any training

## SHAP
- Run SHAP after every training run; report top 10 features by mean |SHAP|
- Key anchors to verify: `atr_pct` #1, `rate_vol_regime` in top 5

## Model saving
- Save to `models/` with descriptive name
- Keep `xgb_phase3_champion.pkl` as safety net — never overwrite unless F1 > 0.375
