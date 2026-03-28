"""
.claude/hooks/validate-data.py — Parquet output validation

Called as a PostToolUse hook after Bash commands. Checks that key pipeline
output files exist and have reasonable row/column counts.

Can also be run standalone:
    python .claude/hooks/validate-data.py

Exit 0 = all present files are valid
Exit 1 = at least one file fails validation (prints details)
"""

import sys
from pathlib import Path

# Allow import from project root
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Expected file specs ───────────────────────────────────────────────────────
# (path, min_rows, min_cols, description)
EXPECTED = [
    ("data/raw/aapl_daily_raw.parquet",          6000,  5, "Raw OHLCV"),
    ("data/processed/aapl_features.parquet",     7000, 50, "Feature matrix"),
    ("data/processed/aapl_labeled.parquet",      7000, 70, "Labeled dataset"),
    ("data/processed/aapl_events.parquet",        500,  6, "Event table"),
    ("data/processed/aapl_with_events.parquet",  7000, 90, "Dataset + event features"),
    ("models/feature_list.json",                    0,  0, "Feature list JSON"),
]

PRED_FILES = [
    ("data/processed/aapl_predictions_best.parquet",         5000, 8, "Phase 2 predictions"),
    ("data/processed/aapl_predictions_interactions.parquet", 5000, 8, "Phase 3 predictions"),
]


def validate_parquet(rel_path: str, min_rows: int, min_cols: int, desc: str) -> list[str]:
    """Returns list of error strings (empty = OK)."""
    errors = []
    path = ROOT / rel_path
    if not path.exists():
        return []   # File not yet generated — skip (not an error)

    try:
        import pandas as pd
        df = pd.read_parquet(path)
        if len(df) < min_rows:
            errors.append(f"{rel_path}: only {len(df)} rows (expected ≥{min_rows})")
        if len(df.columns) < min_cols:
            errors.append(f"{rel_path}: only {len(df.columns)} columns (expected ≥{min_cols})")
        if df.empty:
            errors.append(f"{rel_path}: DataFrame is empty")
    except Exception as e:
        errors.append(f"{rel_path}: could not read — {e}")

    return errors


def validate_json(rel_path: str) -> list[str]:
    import json
    path = ROOT / rel_path
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        if "features" not in data:
            return [f"{rel_path}: missing 'features' key"]
        if len(data["features"]) < 30:
            return [f"{rel_path}: only {len(data['features'])} features (expected ≥30)"]
    except Exception as e:
        return [f"{rel_path}: could not read — {e}"]
    return []


def main():
    all_errors = []

    for rel_path, min_rows, min_cols, desc in EXPECTED + PRED_FILES:
        if rel_path.endswith(".json"):
            errors = validate_json(rel_path)
        else:
            errors = validate_parquet(rel_path, min_rows, min_cols, desc)
        all_errors.extend(errors)

    if all_errors:
        print("DATA VALIDATION WARNINGS:")
        for e in all_errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        # Silent success when run as hook
        if len(sys.argv) > 1 and sys.argv[1] == "--verbose":
            print("DATA VALIDATION: all present files OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
