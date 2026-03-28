"""
.claude/hooks/check-model-eval.py — Model evaluation completeness check

Verifies that a saved model has a corresponding predictions parquet and
that the predictions contain the required evaluation columns.

Usage:
    python .claude/hooks/check-model-eval.py models/xgb_dir_1w.pkl

Exit 0 = evaluation artefacts present and valid
Exit 1 = missing or incomplete artefacts
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Model → expected prediction file mapping ──────────────────────────────────
MODEL_TO_PRED = {
    "xgb_dir_1m.pkl"            : "data/processed/aapl_predictions.parquet",
    "xgb_dir_1w.pkl"            : "data/processed/aapl_predictions_1w.parquet",
    "xgb_dir_1m_weighted.pkl"   : "data/processed/aapl_predictions_1m_weighted.parquet",
    "xgb_dir_1w_weighted.pkl"   : "data/processed/aapl_predictions_best.parquet",
    "xgb_dir_1w_events.pkl"     : "data/processed/aapl_predictions_events.parquet",
    "xgb_best_interactions.pkl" : "data/processed/aapl_predictions_interactions.parquet",
    "lgbm_dir_1w.pkl"           : "data/processed/aapl_predictions_lgbm.parquet",
    "ensemble_meta.pkl"         : "data/processed/aapl_predictions_ensemble.parquet",
}

REQUIRED_PRED_COLS = {"actual", "predicted", "correct", "prob_bear", "prob_sideways", "prob_bull"}
MIN_OOS_ROWS = 1000


def check_model(model_path_str: str) -> list[str]:
    model_path = Path(model_path_str)
    errors = []

    if not model_path.exists():
        errors.append(f"Model file not found: {model_path}")
        return errors

    model_name = model_path.name
    pred_rel   = MODEL_TO_PRED.get(model_name)
    if not pred_rel:
        # Unknown model — just confirm it exists (can't check predictions)
        return []

    pred_path = ROOT / pred_rel
    if not pred_path.exists():
        errors.append(
            f"Missing predictions for {model_name}: expected {pred_rel}\n"
            f"  Run the training script and save OOS predictions before considering this model evaluated."
        )
        return errors

    try:
        import pandas as pd
        pred = pd.read_parquet(pred_path)
        missing_cols = REQUIRED_PRED_COLS - set(pred.columns)
        if missing_cols:
            errors.append(f"{pred_rel}: missing columns {missing_cols}")
        if len(pred) < MIN_OOS_ROWS:
            errors.append(f"{pred_rel}: only {len(pred)} rows (expected ≥{MIN_OOS_ROWS})")

        # Quick sanity: accuracy should be in plausible range
        if "correct" in pred.columns:
            acc = pred["correct"].mean()
            if acc < 0.25 or acc > 0.75:
                errors.append(
                    f"{pred_rel}: OOS accuracy {acc:.3f} is outside plausible range [0.25, 0.75]"
                    " — check for data leakage or a broken pipeline"
                )
    except Exception as e:
        errors.append(f"{pred_rel}: could not read — {e}")

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python check-model-eval.py models/<model>.pkl")
        sys.exit(0)

    errors = check_model(sys.argv[1])

    if errors:
        print(f"MODEL EVALUATION CHECK FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        model_name = Path(sys.argv[1]).name
        print(f"MODEL EVALUATION CHECK: {model_name} — OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
