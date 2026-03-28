# Skill: data-pipeline

Reusable workflow for adding new features or data sources to the aapl_ml pipeline.

## When to use
- Adding a new technical indicator to `src/02_features.py`
- Adding a new external data source (FRED series, yfinance field, hardcoded events)
- Adding new event-derived features to `src/09_event_features.py`

## Workflow

### 1. Understand the data flow
Read `docs/architecture.md` to confirm where the new feature fits in the pipeline:
```
01_fetch_data → 02_features → 03_labels → 04_feature_selection
                                        → 08_events → 09_event_features
```

### 2. Implement the feature
- Add the computation to the appropriate script in `src/`
- All features must be **backward-looking only** — no use of future data
- Use `config/paths.py` for all file paths
- Use `config/settings.py` for shared thresholds

### 3. Run the lookahead check
```bash
python .claude/hooks/check-lookahead.py src/<script>.py
```
Expected: no forward-looking patterns flagged.

### 4. Re-run the pipeline from the changed step
```bash
C:\Users\borra\anaconda3\python.exe src/09_event_features.py
```
Check output shape matches expected (+N columns).

### 5. Run data validation
```bash
python .claude/hooks/validate-data.py
```

### 6. Retrain and compare
Run the relevant training script and compare metrics against the phase champion:
- Phase 3 champion: Macro F1=0.375 (see `config/settings.py:PHASE3_CHAMPION`)

### 7. Run model evaluation check
```bash
python .claude/hooks/check-model-eval.py models/<new_model>.pkl
```

## Guardrails
- **No lookahead**: Never use `.shift(-n)` on target-related columns in features
- **No raw price levels**: Use normalised equivalents (`close_vs_smaX`, `atr_pct`, etc.)
- **Correlation check**: Run `src/04_feature_selection.py` after adding new features to detect |r|>0.95 duplicates
- **Sentinel fills**: Partial-coverage features must use sentinel values (not be dropped)
- **Walk-forward only**: Never use random train/test splits for evaluation
