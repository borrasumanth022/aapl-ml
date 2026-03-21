"""
Step 1 — Fetch raw AAPL data
Downloads daily OHLCV from 1995 to today and saves to Parquet.
Run this once, then re-run periodically to update.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
TICKER      = "AAPL"
START_DATE  = "1995-01-01"
END_DATE    = datetime.today().strftime("%Y-%m-%d")
RAW_DIR     = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "aapl_daily_raw.parquet"

# ── Fetch ─────────────────────────────────────────────────────────────────────
def fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    print(f"Downloading {ticker}  {start} → {end} ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=True)

    # yfinance returns MultiIndex columns when downloading a single ticker.
    # Flatten them to plain strings: ('Close', 'AAPL') → 'close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df.index.name = "date"
    df.index = pd.to_datetime(df.index)

    # Basic sanity checks
    assert "close" in df.columns, "Missing 'close' column"
    assert len(df) > 500,         "Too few rows — something went wrong"

    # Drop any rows where close is NaN (rare but happens at edges)
    df = df.dropna(subset=["close"])

    print(f"  Got {len(df)} trading days  ({df.index[0].date()} → {df.index[-1].date()})")
    return df

# ── Save ──────────────────────────────────────────────────────────────────────
def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    size_kb = path.stat().st_size / 1024
    print(f"  Saved → {path}  ({size_kb:.1f} KB)")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = fetch(TICKER, START_DATE, END_DATE)

    print("\nSample (last 5 rows):")
    print(df.tail())
    print(f"\nColumns : {list(df.columns)}")
    print(f"Date range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Total rows : {len(df)}")

    save(df, OUTPUT_FILE)
    print("\n✓ Step 1 complete. Run 02_features.py next.\n")
