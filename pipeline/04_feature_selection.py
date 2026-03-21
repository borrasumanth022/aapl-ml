"""
Phase 2 — Step 1: Feature Selection
====================================
Loads aapl_labeled.parquet, separates features from labels, checks for
lookahead bias, identifies highly correlated feature pairs, and outputs
a recommended clean feature list for model training.

Output: models/feature_list.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_FILE        = Path(__file__).parent.parent / "data" / "processed" / "aapl_labeled.parquet"
OUT_FILE         = Path(__file__).parent.parent / "models" / "feature_list.json"
CORR_THRESHOLD   = 0.95

LABEL_PREFIXES   = ("ret_", "dir_", "bin_", "adj_ret_")
OHLCV_COLS       = {"open", "high", "low", "close", "volume"}

# Absolute price-level features: not lookahead, but scale-dependent and
# redundant once we have normalised equivalents (close_vs_smaX, bb_pct, atr_pct).
# The raw SMA/EMA values move with price level and will dominate distance-based
# models and cause spurious correlations across different market regimes.
PRICE_LEVEL_FEATURES = {
    "log_close",   # redundant with sma_*/ema_* — all track the same price level
    "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
    "ema_12", "ema_26",
    "bb_upper", "bb_lower",   # use bb_pct / bb_width instead
    "atr_14",                 # use atr_pct (ATR as % of price) instead
    "macd",                   # raw pip value moves with price; macd_hist/signal are relative
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def find_high_corr_pairs(df, threshold):
    """Return list of (feat_a, feat_b, corr) for |corr| > threshold, upper triangle only."""
    corr = df.corr().abs()
    cols = corr.columns.tolist()
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and r > threshold:
                pairs.append((a, b, round(float(r), 4)))
    return sorted(pairs, key=lambda x: x[2], reverse=True)


def drop_recommendation(pairs):
    """
    For each correlated pair decide which to drop.
    Priority rules (keep the higher-priority one):
      1. Keep normalised/relative features over absolute price-level ones.
      2. Keep the feature that appears first positionally (lower index = keep).
    Returns a set of column names to drop.
    """
    # Lower score = higher priority (keep)
    def priority(col):
        if col in PRICE_LEVEL_FEATURES:
            return 10
        if col.startswith("close_vs_"):
            return 1
        if col.endswith("_pct") or col.endswith("_zscore") or col.endswith("_width"):
            return 2
        return 5

    to_drop = set()
    for a, b, _ in pairs:
        # If a is already marked for drop, b is the survivor — skip
        if a in to_drop:
            continue
        drop = b if priority(a) <= priority(b) else a
        to_drop.add(drop)
    return to_drop


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── 1. Load ────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    df = pd.read_parquet(DATA_FILE)
    print(f"  {len(df):,} rows, {len(df.columns)} total columns")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")

    # ── 2. Separate features from labels ──────────────────────────────────────
    section("STEP 1 — SEPARATING FEATURES FROM LABELS")

    label_cols   = [c for c in df.columns if c.startswith(LABEL_PREFIXES)]
    ohlcv_present = OHLCV_COLS & set(df.columns)
    drop_step1   = set(label_cols) | ohlcv_present

    feat_df = df.drop(columns=list(drop_step1))

    print(f"\n  Dropped {len(label_cols):>2} label columns   (ret_*, dir_*, bin_*, adj_ret_*)")
    print(f"  Dropped {len(ohlcv_present):>2} OHLCV columns   (open, high, low, close, volume)")
    print(f"  Remaining candidate features: {len(feat_df.columns)}")
    print(f"\n  All {len(feat_df.columns)} candidate features:")
    for i, col in enumerate(sorted(feat_df.columns), 1):
        print(f"    {i:2d}. {col}")

    # ── 3. Lookahead bias check ────────────────────────────────────────────────
    section("STEP 2 — LOOKAHEAD BIAS CHECK")

    # Scan for any column names containing forward-looking keywords
    fwd_keywords      = ["future", "fwd", "forward", "next", "tomorrow", "lead"]
    direct_lookahead  = [c for c in feat_df.columns
                         if any(kw in c.lower() for kw in fwd_keywords)]

    if direct_lookahead:
        print("\n  [FAIL] Direct lookahead detected — DROP IMMEDIATELY:")
        for c in direct_lookahead:
            print(f"         {c}")
    else:
        print("\n  [PASS] No direct lookahead column names found.")

    # Flag price-level proxies (not lookahead but noted)
    price_level_present = sorted(PRICE_LEVEL_FEATURES & set(feat_df.columns))
    print(f"\n  [NOTE] {len(price_level_present)} price-level proxy features flagged:")
    print(f"         These are computed from past data only (no lookahead),")
    print(f"         but encode absolute price magnitude — makes models brittle")
    print(f"         across different price regimes. Normalised alternatives exist.")
    for c in price_level_present:
        alt = {
            "log_close":  "->  use close_vs_sma* / price_52w_pct",
            "sma_10":     "->  use close_vs_sma10",
            "sma_20":     "->  use close_vs_sma20",
            "sma_50":     "->  use close_vs_sma50",
            "sma_100":    "->  use close_vs_sma100",
            "sma_200":    "->  use close_vs_sma200",
            "ema_12":     "->  use macd_hist (ema_12 - ema_26, normalised by price)",
            "ema_26":     "->  use macd_hist",
            "bb_upper":   "->  use bb_pct / bb_width",
            "bb_lower":   "->  use bb_pct / bb_width",
            "atr_14":     "->  use atr_pct",
            "macd":       "->  use macd_hist / macd_signal (relative)",
        }.get(c, "")
        print(f"         {c:<15s}  {alt}")

    # ── 4. Correlation analysis ────────────────────────────────────────────────
    section("STEP 3 — CORRELATION ANALYSIS  (|r| > {:.2f})".format(CORR_THRESHOLD))

    print("\n  Computing correlation matrix...")
    pairs = find_high_corr_pairs(feat_df, CORR_THRESHOLD)

    if pairs:
        print(f"\n  Found {len(pairs)} highly correlated pairs:\n")
        print(f"  {'Feature A':<25}  {'Feature B':<25}  {'|r|':>6}  Action")
        print(f"  {'-'*25}  {'-'*25}  {'-'*6}  {'-'*20}")
        to_drop_corr = drop_recommendation(pairs)
        for a, b, r in pairs:
            drop = b if b in to_drop_corr else a
            keep = a if drop == b else b
            print(f"  {a:<25}  {b:<25}  {r:>6.3f}  drop {drop}")
    else:
        to_drop_corr = set()
        print("  No pairs exceed the threshold.")

    # ── 5. Final recommended feature list ──────────────────────────────────────
    section("STEP 4 — RECOMMENDED FEATURE LIST")

    to_drop_all  = (PRICE_LEVEL_FEATURES & set(feat_df.columns)) | to_drop_corr
    # Don't double-count drops that are in both sets
    drop_price   = sorted(PRICE_LEVEL_FEATURES & set(feat_df.columns))
    drop_corr    = sorted(to_drop_corr - (PRICE_LEVEL_FEATURES & set(feat_df.columns)))
    recommended  = sorted([c for c in feat_df.columns if c not in to_drop_all])

    print(f"\n  Input features (after removing labels/OHLCV) : {len(feat_df.columns)}")
    print(f"  Dropped — price-level proxies                : {len(drop_price)}")
    print(f"  Dropped — correlation duplicates             : {len(drop_corr)}")
    print(f"  -------------------------------------------------")
    print(f"  Recommended features for training            : {len(recommended)}")

    print(f"\n  Dropped (price-level): {drop_price}")
    print(f"  Dropped (correlation): {drop_corr}")

    print(f"\n  Final feature list ({len(recommended)}):\n")
    for i, f in enumerate(recommended, 1):
        print(f"    {i:2d}. {f}")

    # ── 6. Save ────────────────────────────────────────────────────────────────
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "n_features"          : len(recommended),
        "features"            : recommended,
        "dropped_price_level" : drop_price,
        "dropped_correlation" : drop_corr,
        "lookahead_flags"     : direct_lookahead,
        "corr_threshold"      : CORR_THRESHOLD,
    }
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {OUT_FILE}")
    print("\nStep 4 complete. Run pipeline/05_train_baseline.py next.\n")
