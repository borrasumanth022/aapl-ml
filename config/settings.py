"""
config/settings.py — Shared hyperparameters, label encoding, and thresholds.

Centralised here so that all scripts that need to compare models use identical
parameters, and so that Phase 4 experiments only need to change one file.

Usage:
    from config import settings as S
    model = XGBClassifier(**S.XGB_PARAMS)
    y_enc = np.array([S.LABEL_ENCODE[v] for v in y])
"""

# ── Ticker / data range ───────────────────────────────────────────────────────
TICKER     = "AAPL"
START_DATE = "1995-01-01"

# ── Label engineering (03_labels.py) ─────────────────────────────────────────
HORIZONS = {
    "1w":  5,
    "1m":  21,
    "3m":  63,
    "6m":  126,
    "1y":  252,
}
DIRECTION_THRESHOLD = 0.02   # ±2% sideways band

# ── Walk-forward validation ───────────────────────────────────────────────────
N_SPLITS = 5

# ── XGBoost — baseline params (used across Phase 2 and 3) ────────────────────
# Conservative to avoid overfitting on early folds. Kept fixed across phases
# so improvements are attributable to features / architecture, not tuning.
XGB_PARAMS = {
    "n_estimators"    : 300,
    "max_depth"       : 4,
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "eval_metric"     : "mlogloss",
    "random_state"    : 42,
    "n_jobs"          : -1,
    "verbosity"       : 0,
}

# ── Label encoding ────────────────────────────────────────────────────────────
# XGBoost requires contiguous integer labels starting at 0.
# Original labels: Bear=-1, Sideways=0, Bull=+1
LABEL_ENCODE = {-1: 0,  0: 1,  1: 2}
LABEL_DECODE = { 0:-1,  1: 0,  2: 1}
CLASS_NAMES  = {-1: "Bear (-1)", 0: "Side ( 0)", 1: "Bull (+1)"}
CLASS_LABELS = ["Bear", "Sideways", "Bull"]   # index-aligned with encoded 0,1,2
CLASSES      = [-1, 0, 1]

# ── Feature engineering — event features ─────────────────────────────────────
CAP_EARNINGS  = 90    # days: sentinel for pre-2005 rows with no earnings history
CAP_PRODUCT   = 180   # days: sentinel for pre-2000 rows with no product event history
IPHONE_WINDOW = 60    # ±days around iPhone launch to flag is_iphone_cycle

INFLATION_HIGH   = 4.0   # CPI YoY % above which → high regime
INFLATION_LOW    = 1.5   # CPI YoY % below which → low regime
RATE_RISING_BPS  =  10   # 3m fed change above which → rising
RATE_FALLING_BPS = -10   # 3m fed change below which → falling

# ── Event feature names (used in Phase 3 scripts) ────────────────────────────
EVENT_FEATURES = [
    "days_to_next_earnings", "days_since_last_earnings", "has_earnings_data",
    "last_eps_surprise_pct", "earnings_streak",
    "fed_rate_level", "fed_rate_change_1m", "fed_rate_change_3m",
    "cpi_yoy_change", "unemployment_level", "unemployment_change_3m",
    "days_to_next_product_event", "days_since_last_product_event",
    "is_iphone_cycle", "rate_environment", "inflation_regime",
]

INTERACTION_FEATURES = [
    "earnings_proximity_surprise",
    "macro_stress_score",
    "vol_macro_interaction",
    "earnings_momentum",
    "rate_vol_regime",
]

# ── Phase comparison baselines ────────────────────────────────────────────────
# Stored here so evaluation scripts can import directly instead of hardcoding.
PHASE2_BEST = {
    "model"  : "xgb_dir_1w_weighted",
    "acc"    : 0.3835,
    "f1"     : 0.367,
    "recall" : {"Bear": 0.2310, "Sideways": 0.4549, "Bull": 0.4177},
}
PHASE3_CHAMPION = {
    "model"  : "xgb_best_interactions",
    "acc"    : 0.3830,
    "f1"     : 0.375,
    "recall" : {"Bear": 0.3060, "Sideways": 0.3990, "Bull": 0.4200},
}

# ── Phase 5 — LSTM hyperparameters ────────────────────────────────────────────
SEQ_LEN = 100   # 20 trading weeks * 5 days/week

LSTM_ARCH = {
    "hidden_size" : 128,
    "num_layers"  : 2,
    "dropout"     : 0.3,
}

LSTM_TRAIN = {
    "lr"          : 0.001,
    "batch_size"  : 64,
    "max_epochs"  : 100,
    "patience_es" : 10,    # early stopping
    "patience_lr" : 5,     # ReduceLROnPlateau
    "lr_factor"   : 0.5,
    "lr_min"      : 1e-5,
    "grad_clip"   : 1.0,
}

HOLDOUT_START = "2024-01-01"

# ── Phase 5 Step 2 — raw OHLCV LSTM ──────────────────────────────────────────
OHLCV_FEATURES       = ["open", "high", "low", "close", "volume"]
ROLLING_ZSCORE_WIN   = 252   # trading days for rolling mean/std normalization
SEQ_LEN_RAW          = 60    # 3 months of daily data

LSTM_RAW_ARCH = {
    "hidden_size" : 256,
    "num_layers"  : 2,
    "dropout"     : 0.4,
}

LSTM_RAW_TRAIN = {
    "lr"          : 0.001,
    "weight_decay": 1e-4,
    "batch_size"  : 64,
    "max_epochs"  : 100,
    "patience_es" : 10,
    "patience_lr" : 5,
    "lr_factor"   : 0.5,
    "lr_min"      : 1e-5,
    "grad_clip"   : 1.0,
}
