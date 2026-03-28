Evaluate the aapl_ml champion model and produce a structured report for $ARGUMENTS (default: xgb_best_interactions, dir_1w).

Steps:
1. Load `models/xgb_best_interactions.pkl` (or `xgb_phase3_champion.pkl` if specified).
2. Load `data/processed/aapl_predictions_interactions.parquet`.
3. Verify required columns: `actual`, `predicted`, `correct`, `prob_bear`, `prob_sideways`, `prob_bull`.
4. Compute and report:

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| OOS Accuracy | ... | 37.50% (always-Bull) |
| Macro F1 | ... | 0.333 (random) |
| Bull Recall | ... | — |
| Bear Recall | ... | — |
| Sideways Recall | ... | — |

5. Print confusion matrix.
6. Compare vs previous champions:
   - Phase 2 baseline (dir_1w, no events): F1=0.367, Acc=38.35%
   - Phase 3 champion (57 features + events): F1=0.375, Acc=38.30% ← current
7. Run SHAP: top 10 features by mean |SHAP|.
   Key findings to verify: atr_pct #1, rate_vol_regime in top 5, earnings_proximity_surprise in top 10.
8. Report OOS date range and number of samples.

Note: aapl_ml uses -1/0/1 label encoding (Bear/Sideways/Bull), not the 0/1/2 used in market_ml.

If $ARGUMENTS specifies `--phase N`, compare against the Phase N model file.
