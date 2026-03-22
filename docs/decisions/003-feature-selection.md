# ADR 003 — Feature selection: drop price-level proxies and high-correlation pairs

**Status**: Accepted
**Date**: 2025 (Phase 2)

## Context

After computing 52 candidate technical features (from `src/02_features.py`), we need to select a subset for training. Raw features include both price-level quantities (e.g., `sma_50` in dollars) and normalised relative quantities (e.g., `close_vs_sma50` as a fraction).

## Decision

**Step 1**: Drop 12 price-level proxy features manually:
`log_close`, `sma_10/20/50/100/200`, `ema_12/26`, `bb_upper`, `bb_lower`, `atr_14`, `macd`

**Step 2**: Drop 4 high-correlation duplicates (|r| > 0.95):
`log_return_1d`, `quarter`, `roc_5`, `williams_r`

Result: **36 selected features** saved to `models/feature_list.json`.

## Rationale

### Why drop price-level features?
- `sma_50` in 2005 is ~$8; in 2024 it's ~$200. The model sees these as completely different regimes.
- Normalised equivalents (`close_vs_sma50 = close/sma_50 - 1`) are regime-independent and provide the same information.
- Tree models can work around this with splits at each scale, but it wastes tree capacity.

### Why drop high-correlation duplicates?
- `log_return_1d` ≈ `return_1d` (|r| > 0.999 over this dataset)
- Adding a near-duplicate feature doesn't give the model new information, but doubles the noise it must model for that signal.
- Threshold |r| > 0.95 is conservative — only drops near-identical features.

### Why not use LASSO or automatic feature selection?
- Automatic selection methods can inadvertently introduce leakage if features encode future information
- The manual rules are transparent, auditable, and interpretable
- SHAP analysis after training confirms that the selected features are meaningful

## Consequences

- From 52 candidates → 36 features (31% reduction)
- All 36 features are normalised/relative — robust across 30 years of price history
- Feature list is fixed after this step; adding new features goes through the event feature pipeline (Phase 3+)

## Re-running
```bash
C:\Users\borra\anaconda3\python.exe src/04_feature_selection.py
```
This recomputes correlations and regenerates `models/feature_list.json`. Re-run if `src/02_features.py` adds new features.
