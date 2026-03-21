"""
Step 3 — Build prediction labels
Computes forward returns at multiple horizons and converts them to
classification targets (direction) and regression targets (magnitude).

Output: data/processed/aapl_labeled.parquet

Label design:
  Regression  → actual % forward return
  Direction   → +1 (up) / 0 (sideways) / -1 (down) using a threshold
                so the model doesn't have to predict noise around 0
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES_FILE = Path(__file__).parent.parent / "data" / "processed" / "aapl_features.parquet"
OUT_FILE      = Path(__file__).parent.parent / "data" / "processed" / "aapl_labeled.parquet"

# Horizons (trading days)
HORIZONS = {
    "1w":  5,
    "1m":  21,
    "3m":  63,
    "6m":  126,
    "1y":  252,
}

# "Sideways" band — returns within ±THRESHOLD% are labelled 0 (no clear direction)
# Helps the model focus on meaningful moves, not noise
DIRECTION_THRESHOLD = 0.02   # 2%


# ── Label builders ────────────────────────────────────────────────────────────
def forward_return(close: pd.Series, n: int) -> pd.Series:
    """Actual % return n trading days into the future."""
    return close.shift(-n) / close - 1


def direction_label(fwd_ret: pd.Series, threshold: float) -> pd.Series:
    """
    +1  →  return > +threshold  (bullish)
     0  →  abs(return) <= threshold  (sideways / noise)
    -1  →  return < -threshold  (bearish)
    """
    labels = pd.Series(0, index=fwd_ret.index, dtype=int)
    labels[fwd_ret >  threshold] =  1
    labels[fwd_ret < -threshold] = -1
    return labels


def binary_direction(fwd_ret: pd.Series) -> pd.Series:
    """Simple up/down (1/0) — for binary classification models."""
    return (fwd_ret > 0).astype(int)


# ── Main ──────────────────────────────────────────────────────────────────────
def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]

    for name, n_days in HORIZONS.items():
        fwd   = forward_return(close, n_days)

        # Regression target (continuous)
        df[f"ret_{name}"]       = fwd

        # 3-class direction label  (+1 / 0 / -1)
        df[f"dir_{name}"]       = direction_label(fwd, DIRECTION_THRESHOLD)

        # Binary direction (1=up, 0=down) — simpler, used for first models
        df[f"bin_{name}"]       = binary_direction(fwd)

        # Volatility-adjusted return  (return / realised vol)
        # A 2% move in a calm period is more meaningful than in a volatile one
        vol_col = "hvol_21d" if "hvol_21d" in df.columns else None
        if vol_col:
            df[f"adj_ret_{name}"] = fwd / (df[vol_col] + 1e-9)

    # Drop the last N rows where forward return can't be computed
    # (the maximum horizon determines how many rows are NaN at the end)
    max_horizon = max(HORIZONS.values())
    df = df.iloc[:-max_horizon]

    return df


def label_summary(df: pd.DataFrame) -> None:
    print("\nLabel distribution summary:")
    print("-" * 55)
    for name in HORIZONS:
        col = f"dir_{name}"
        if col not in df.columns:
            continue
        counts  = df[col].value_counts().sort_index()
        total   = len(df)
        up_pct  = counts.get( 1, 0) / total * 100
        side_pct= counts.get( 0, 0) / total * 100
        dn_pct  = counts.get(-1, 0) / total * 100
        print(f"  {name:>4}  UP {up_pct:5.1f}%  SIDE {side_pct:5.1f}%  DN {dn_pct:5.1f}%  "
              f"(threshold +/-{DIRECTION_THRESHOLD*100:.0f}%)")
    print("-" * 55)
    print("  If UP and DN are roughly balanced, the labels are healthy.")
    print("  A strong UP skew is normal for AAPL over the long run.\n")


if __name__ == "__main__":
    print("Loading feature matrix ...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    print("Building labels ...")
    df = build_labels(df)

    label_summary(df)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE)

    print(f"Final dataset shape : {df.shape}")
    print(f"Date range          : {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Saved: {OUT_FILE}")

    # Quick peek at label columns
    label_cols = [c for c in df.columns if c.startswith(("ret_","dir_","bin_","adj_ret_"))]
    print(f"\nLabel columns added ({len(label_cols)}):")
    for c in label_cols:
        print(f"  {c}")

    print("\nStep 3 complete. Your dataset is ready for modelling.\n")
    print("Next step: open notebooks/01_explore.ipynb to inspect the data")
