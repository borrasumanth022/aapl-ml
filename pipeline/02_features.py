"""
Step 2 — Feature engineering
Takes raw OHLCV and computes technical indicators + price-derived features.
Output: data/processed/aapl_features.parquet

Feature groups:
  A. Price & volume transforms   — returns, log-price, volume z-score
  B. Trend indicators            — SMA, EMA, MACD, ADX
  C. Momentum indicators         — RSI, Stochastic, Rate of Change
  D. Volatility indicators       — Bollinger Bands, ATR, historical vol
  E. Market microstructure       — candle body, shadows, gap
  F. Calendar features           — day-of-week, month, quarter
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_FILE    = Path(__file__).parent.parent / "data" / "raw"  / "aapl_daily_raw.parquet"
OUT_FILE    = Path(__file__).parent.parent / "data" / "processed" / "aapl_features.parquet"


# ══════════════════════════════════════════════════════════════════════════════
# A. Price & volume transforms
# ══════════════════════════════════════════════════════════════════════════════
def add_price_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df["log_close"]      = np.log(df["close"])
    df["return_1d"]      = df["close"].pct_change(1)
    df["return_2d"]      = df["close"].pct_change(2)
    df["return_5d"]      = df["close"].pct_change(5)
    df["log_return_1d"]  = np.log(df["close"] / df["close"].shift(1))

    # Volume: z-score over a 20-day rolling window (normalises for splits/growth)
    vol_mean             = df["volume"].rolling(20).mean()
    vol_std              = df["volume"].rolling(20).std()
    df["volume_zscore"]  = (df["volume"] - vol_mean) / (vol_std + 1e-9)

    # Price relative to its own 52-week range (0 = at low, 1 = at high)
    hi52 = df["high"].rolling(252).max()
    lo52 = df["low"].rolling(252).min()
    df["price_52w_pct"]  = (df["close"] - lo52) / (hi52 - lo52 + 1e-9)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# B. Trend indicators
# ══════════════════════════════════════════════════════════════════════════════
def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    # Simple moving averages
    for w in [10, 20, 50, 100, 200]:
        df[f"sma_{w}"]        = df["close"].rolling(w).mean()
        df[f"close_vs_sma{w}"] = df["close"] / df[f"sma_{w}"] - 1  # % above/below

    # Exponential moving averages
    for w in [12, 26]:
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()

    # MACD  (EMA12 - EMA26) and signal line (EMA9 of MACD)
    df["macd"]         = df["ema_12"] - df["ema_26"]
    df["macd_signal"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]    = df["macd"] - df["macd_signal"]

    # Golden/death cross flags
    df["cross_50_200"]  = (df["sma_50"] > df["sma_200"]).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# C. Momentum indicators
# ══════════════════════════════════════════════════════════════════════════════
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0).rolling(window).mean()
    loss   = (-delta.clip(upper=0)).rolling(window).mean()
    rs     = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)

def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi_14"]  = rsi(df["close"], 14)
    df["rsi_7"]   = rsi(df["close"],  7)

    # Rate of Change
    for w in [5, 10, 21]:
        df[f"roc_{w}"] = df["close"].pct_change(w) * 100

    # Stochastic %K  (close position within high-low range over 14 days)
    lo14             = df["low"].rolling(14).min()
    hi14             = df["high"].rolling(14).max()
    df["stoch_k"]    = 100 * (df["close"] - lo14) / (hi14 - lo14 + 1e-9)
    df["stoch_d"]    = df["stoch_k"].rolling(3).mean()   # signal line

    # Williams %R (inverse of Stochastic, -100 to 0)
    df["williams_r"] = -100 * (hi14 - df["close"]) / (hi14 - lo14 + 1e-9)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# D. Volatility indicators
# ══════════════════════════════════════════════════════════════════════════════
def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    # Bollinger Bands (20-day, 2σ)
    sma20            = df["close"].rolling(20).mean()
    std20            = df["close"].rolling(20).std()
    df["bb_upper"]   = sma20 + 2 * std20
    df["bb_lower"]   = sma20 - 2 * std20
    df["bb_width"]   = (df["bb_upper"] - df["bb_lower"]) / sma20   # normalised width
    df["bb_pct"]     = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # Average True Range  (14-day)
    prev_close       = df["close"].shift(1)
    tr               = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr_14"]     = tr.rolling(14).mean()
    df["atr_pct"]    = df["atr_14"] / df["close"]   # ATR as % of price

    # Historical (realised) volatility at multiple windows
    log_ret          = np.log(df["close"] / df["close"].shift(1))
    for w in [10, 21, 63]:
        df[f"hvol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)  # annualised

    return df


# ══════════════════════════════════════════════════════════════════════════════
# E. Market microstructure (candle features)
# ══════════════════════════════════════════════════════════════════════════════
def add_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    candle_range         = (df["high"] - df["low"]).replace(0, np.nan)

    # Body: how much of the candle range is body vs shadows
    df["candle_body"]    = (df["close"] - df["open"]).abs() / candle_range
    df["upper_shadow"]   = (df["high"] - df[["close","open"]].max(axis=1)) / candle_range
    df["lower_shadow"]   = (df[["close","open"]].min(axis=1) - df["low"])  / candle_range

    # Direction of the day's candle
    df["candle_dir"]     = np.sign(df["close"] - df["open"])

    # Gap from prior close
    df["gap_pct"]        = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # High-Low range as % of close
    df["hl_range_pct"]   = (df["high"] - df["low"]) / df["close"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
# F. Calendar features
# ══════════════════════════════════════════════════════════════════════════════
def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    idx                  = df.index
    df["day_of_week"]    = idx.dayofweek          # 0=Mon, 4=Fri
    df["month"]          = idx.month
    df["quarter"]        = idx.quarter
    df["is_month_end"]   = idx.is_month_end.astype(int)
    df["is_month_start"] = idx.is_month_start.astype(int)
    df["is_quarter_end"] = idx.is_quarter_end.astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

    print("  Computing price transforms ...")
    df = add_price_transforms(df)

    print("  Computing trend indicators ...")
    df = add_trend(df)

    print("  Computing momentum indicators ...")
    df = add_momentum(df)

    print("  Computing volatility indicators ...")
    df = add_volatility(df)

    print("  Computing microstructure features ...")
    df = add_microstructure(df)

    print("  Computing calendar features ...")
    df = add_calendar(df)

    # Drop the early rows that are NaN due to rolling windows (200-day SMA needs 200 rows)
    before = len(df)
    df = df.dropna(subset=["sma_200"])
    print(f"  Dropped {before - len(df)} warm-up rows (needed for 200-day SMA)")

    return df


if __name__ == "__main__":
    print("Loading raw data ...")
    raw = pd.read_parquet(RAW_FILE)

    print(f"  {len(raw)} rows loaded")
    print("Building features ...")

    features = build_features(raw)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT_FILE)

    print(f"\nFeature matrix shape : {features.shape}")
    print(f"Columns ({len(features.columns)}) :")
    for col in sorted(features.columns):
        print(f"  {col}")

    print(f"\nSaved → {OUT_FILE}")
    print("\n✓ Step 2 complete. Run 03_labels.py next.\n")
