# Runbook: Add New Features to the Pipeline

Use this when you want to add a technical indicator, FRED series, or event-derived feature.

---

## Decision: Where does the feature go?

| Feature type | Script to edit | Output file |
|---|---|---|
| New technical indicator (price/volume based) | `src/02_features.py` | `aapl_features.parquet` |
| New label horizon or threshold | `src/03_labels.py` | `aapl_labeled.parquet` |
| New external event (earnings, FRED series, product event) | `src/08_events.py` | `aapl_events.parquet` |
| New event-derived feature (proximity, regime, composite) | `src/09_event_features.py` | `aapl_with_events.parquet` |
| New interaction between existing features | `src/11_interaction_features.py` | (computed inline) |

---

## Step-by-step: Adding a technical indicator

1. **Add computation to `src/02_features.py`**
   Add a new function `add_<group>(df)` or extend an existing one.
   All values must be computed from `df["close"]` / `df["high"]` / etc. — no forward references.

2. **Run the lookahead check**
   ```bash
   python .claude/hooks/check-lookahead.py src/02_features.py
   ```

3. **Re-run features + labels + feature selection**
   ```bash
   C:\Users\borra\anaconda3\python.exe src/02_features.py
   C:\Users\borra\anaconda3\python.exe src/03_labels.py
   C:\Users\borra\anaconda3\python.exe src/04_feature_selection.py
   ```
   Check: does the new feature survive the correlation filter? (|r| < 0.95 with all others)

4. **Retrain and compare**
   Run the appropriate training script. Compare Macro F1 vs `config/settings.py:PHASE3_CHAMPION`.

---

## Step-by-step: Adding a new FRED macro series

1. **Add to `FRED_SERIES` dict in `src/08_events.py`**
   ```python
   "SOFR": {
       "name": "SOFR Rate",
       "subtype": "sofr_change",
       "change": "mom",
       "unit": "bps",
       "positive_is_good": False,
       "desc": "SOFR: {val:.3f}% (change {chg:+.0f}bps)",
   }
   ```

2. **Add the derived feature in `src/09_event_features.py`**
   Use `fetch_fred_level()` + `align_monthly_to_daily()` pattern from existing FRED features.

3. **Add sentinel fill if coverage < 100%**
   For series that don't go back to 1995, add a sentinel fill before `dropna()`:
   ```python
   df["new_feature"].fillna(sentinel_value)
   ```
   Document the sentinel in `config/settings.py`.

4. **Add the feature name to `config/settings.py:EVENT_FEATURES`**

5. **Re-run pipeline from step 8:**
   ```bash
   C:\Users\borra\anaconda3\python.exe src/08_events.py
   C:\Users\borra\anaconda3\python.exe src/09_event_features.py
   C:\Users\borra\anaconda3\python.exe src/10_retrain_with_events.py
   ```

---

## Anti-patterns to avoid

**Forward-fill timing**: When aligning monthly FRED data to daily, always use `.ffill()` after a reindex — never `.bfill()`. Backfill would use future published values.

**Target leakage**: Never include a column whose name starts with `ret_`, `dir_`, `bin_`, or `adj_ret_` in the feature list.

**Rolling windows that extend into the future**: `.rolling(N).mean()` is safe (uses past N days). `.rolling(N, center=True).mean()` is NOT (uses N/2 future days).

**Computing z-scores on test data separately**: When computing global z-scores for interaction features (e.g., `macro_stress_score`), use the global mean/std from the full training+test dataset — this is acceptable since z-score parameters don't encode label information.
