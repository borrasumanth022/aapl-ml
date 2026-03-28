Review $ARGUMENTS (default: all recently modified files in src/) for lookahead bias, ML best practices, and coding standards.

Run checks in priority order:

## 1. Lookahead bias (CRITICAL — run hook first)
```bash
python .claude/hooks/check-lookahead.py src/<file>.py
```
Manual checks:
- Any `shift(-N)` where N > 0 on non-label columns
- Column names: `future_`, `fwd_`, `_forward`, `next_`, `tomorrow`, `lead_`
- `fit_transform()` on full dataset (must fit on train fold only)
- `.rolling()` windows that peek forward

## 2. Walk-forward validation
- `TimeSeriesSplit(n_splits=5)` — never KFold with shuffle=True
- Test indices always strictly after training indices
- No metrics computed on training fold

## 3. Data leakage
- Scalers: `fit(X_train).transform(X_test)` — never `fit_transform(X_full)`
- No feature computed from future price, volume, or labels

## 4. Coding standards
- File paths via `pathlib.Path` — no raw Windows strings in src/
- `print()` output ASCII-only
- Type hints on function signatures
- Skip-if-exists on all output files

## 5. ML standards
- Naive baseline (37.50% always-Bull for dir_1w) reported alongside accuracy
- Both OOS Accuracy AND Macro F1 reported
- `class_weight="balanced"` set

## 6. aapl_ml-specific
- Label encoding: -1=Bear, 0=Sideways, 1=Bull (aapl_ml convention — different from market_ml)
- Prediction columns: `actual`, `predicted`, `correct`, `prob_bear`, `prob_sideways`, `prob_bull`
- Model file: save to `models/` with clear name, keep `xgb_phase3_champion.pkl` as safety net

Format: `[SEVERITY] file.py:line — description — consequence`
