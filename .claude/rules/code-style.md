# Python Coding Standards -- aapl_ml

## Script structure
- Each script (01-15) is standalone and re-runnable
- Print clear progress messages as they run
- Skip if output file exists; delete to force re-run
- sys.exit(1) on bad data -- do not silently continue

## Naming
- Files: NN_description.py (step number prefix)
- Functions: snake_case verbs (load_features, build_labels)
- Variables: descriptive lowercase (df_features, fold_results)
- Constants: UPPER_SNAKE_CASE (SIDEWAYS_BAND, HOLDOUT_DATE)

## Error handling
At system boundaries only (yfinance, FRED, parquet reads).
Do not wrap internal logic in try/except.

## No speculative complexity
- No config flags for things with one value
- No helper functions used only once
- Prove value at each phase before adding complexity

