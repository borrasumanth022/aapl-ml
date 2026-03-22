"""
config/paths.py — Canonical file paths for the aapl_ml pipeline.

All pipeline scripts import from here so that path changes only ever
need to happen in one place.

Usage:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import paths as P

    df = pd.read_parquet(P.DATA_LABELED)
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

# ── Raw data ──────────────────────────────────────────────────────────────────
DATA_RAW      = ROOT / "data" / "raw" / "aapl_daily_raw.parquet"

# ── Processed data ────────────────────────────────────────────────────────────
DATA_FEATURES   = ROOT / "data" / "processed" / "aapl_features.parquet"
DATA_LABELED    = ROOT / "data" / "processed" / "aapl_labeled.parquet"
DATA_EVENTS     = ROOT / "data" / "processed" / "aapl_events.parquet"
DATA_WITH_EVENTS = ROOT / "data" / "processed" / "aapl_with_events.parquet"

# ── Prediction outputs ────────────────────────────────────────────────────────
PRED_BASELINE      = ROOT / "data" / "processed" / "aapl_predictions.parquet"
PRED_DIR1W         = ROOT / "data" / "processed" / "aapl_predictions_1w.parquet"
PRED_WEIGHTED_1M   = ROOT / "data" / "processed" / "aapl_predictions_1m_weighted.parquet"
PRED_BEST          = ROOT / "data" / "processed" / "aapl_predictions_best.parquet"
PRED_EVENTS        = ROOT / "data" / "processed" / "aapl_predictions_events.parquet"
PRED_INTERACTIONS  = ROOT / "data" / "processed" / "aapl_predictions_interactions.parquet"
PRED_PHASE3_CHAMP  = ROOT / "data" / "processed" / "aapl_predictions_phase3_champion.parquet"

# ── Models ────────────────────────────────────────────────────────────────────
FEATURE_LIST        = ROOT / "models" / "feature_list.json"
MODEL_BASELINE      = ROOT / "models" / "xgb_dir_1m.pkl"
MODEL_DIR1W         = ROOT / "models" / "xgb_dir_1w.pkl"
MODEL_WEIGHTED_1M   = ROOT / "models" / "xgb_dir_1m_weighted.pkl"
MODEL_WEIGHTED_1W   = ROOT / "models" / "xgb_dir_1w_weighted.pkl"
MODEL_EVENTS        = ROOT / "models" / "xgb_dir_1w_events.pkl"
MODEL_INTERACTIONS  = ROOT / "models" / "xgb_best_interactions.pkl"
MODEL_PHASE3_CHAMP  = ROOT / "models" / "xgb_phase3_champion.pkl"

# ── SHAP outputs ──────────────────────────────────────────────────────────────
SHAP_SUMMARY       = ROOT / "models" / "shap_summary.csv"
SHAP_EVENTS        = ROOT / "models" / "shap_summary_events.csv"
SHAP_INTERACTIONS  = ROOT / "models" / "shap_summary_interactions.csv"

# ── Phase 4 (placeholders — filled in as models are trained) ──────────────────
MODEL_LGBM          = ROOT / "models" / "lgbm_dir_1w.pkl"
PRED_LGBM           = ROOT / "data" / "processed" / "aapl_predictions_lgbm.parquet"
MODEL_ENSEMBLE      = ROOT / "models" / "ensemble_meta.pkl"
PRED_ENSEMBLE       = ROOT / "data" / "processed" / "aapl_predictions_ensemble.parquet"
