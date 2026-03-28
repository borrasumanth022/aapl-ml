Retrain the aapl_ml XGBoost champion model for $ARGUMENTS (default: xgb_best_interactions, dir_1w target).

Steps:
1. Confirm `data/processed/aapl_with_events.parquet` exists (≥7,000 rows, ≥90 cols).
2. Run `python .claude/hooks/check-lookahead.py src/06_train.py` — must pass.
3. Run `{PYTHON_EXE} src/06_train.py` (read path from CLAUDE.local.md).
4. Report walk-forward metrics (5 folds, expanding window, TimeSeriesSplit):
   - OOS Accuracy vs naive baseline (always-Bull = 37.50% for dir_1w)
   - Macro F1 — must be ≥ 0.35 to accept
   - Per-class recall: Bear / Sideways / Bull
   - Comparison vs current champion: F1=0.375 (xgb_best_interactions, 57 features)
5. Run SHAP analysis — show top 10 features by mean |SHAP|.
6. If new model beats champion (F1 > 0.375), save as new champion.
   If not, do NOT overwrite. Report gap.

Current champion benchmarks (do not regress):
- Model: xgb_best_interactions.pkl (57 features, dir_1w target)
- OOS Accuracy: 38.30%
- Macro F1: 0.375
- Top features: atr_pct, hvol_21d/63d, rate_vol_regime, price_52w_pct

Standard hyperparams: n_estimators=300, max_depth=4, lr=0.05, subsample=0.8, min_child_weight=20, class_weight="balanced".
Label encoding for aapl_ml (old pipeline): **-1=Bear, 0=Sideways, 1=Bull**
Note: market_ml uses 0/1/2 — do not mix encodings.
