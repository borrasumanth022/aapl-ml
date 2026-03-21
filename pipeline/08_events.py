"""
Phase 3 - Step 1: AAPL Event Data Collection
=============================================
Collects three categories of events and unifies them into a single parquet.

Source 1 — yfinance earnings
  AAPL quarterly EPS (actual vs estimate) and revenue where available.
  ~25 quarters of history (~2020 onwards for EPS surprise, 5Q for revenue).

Source 2 — FRED macroeconomic series (no API key needed — CSV download)
  FEDFUNDS : Fed funds rate (monthly) — rate change as surprise proxy
  CPIAUCSL : CPI (monthly)           — MoM change
  UNRATE   : Unemployment (monthly)  — MoM change
  GDP      : GDP (quarterly)         — QoQ change

Source 3 — Hardcoded Apple events 2000-2026
  iPhone launches, WWDC dates, major product launches, stock splits.

Output schema (one row per event):
  date         — event date (DatetimeIndex)
  event_type   — earnings / macro / product / split
  event_subtype— e.g. "eps_surprise", "fed_rate_change", "iphone_launch"
  magnitude    — numeric size of the event (surprise %, bps change, split ratio, etc.)
  direction    — positive / negative / neutral
  source       — yfinance / fred / hardcoded
  description  — human-readable one-liner

Output: data/processed/aapl_events.parquet
"""

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

OUT_FILE = Path(__file__).parent.parent / "data" / "processed" / "aapl_events.parquet"

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FRED_START = "1993-01-01"   # enough history to align with price data start (1995)


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


def direction_from(value, positive_is_good=True):
    """Return 'positive'/'negative'/'neutral' based on sign."""
    if pd.isna(value) or value == 0:
        return "neutral"
    if positive_is_good:
        return "positive" if value > 0 else "negative"
    else:
        return "negative" if value > 0 else "positive"


def fetch_fred(series_id):
    """Download a FRED series as a dated pandas Series. No API key required."""
    url = FRED_BASE.format(series_id)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    # FRED CSV header is "observation_date,<SERIES_ID>"
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["observation_date"],
                     index_col="observation_date")
    df.columns = [series_id]
    df = df[df[series_id] != "."]   # FRED uses "." for missing
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df = df.dropna()
    return df[series_id].sort_index()


def make_event_row(date, event_type, event_subtype, magnitude,
                   direction, source, description):
    ts = pd.Timestamp(date)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return {
        "date":          ts.normalize(),
        "event_type":    event_type,
        "event_subtype": event_subtype,
        "magnitude":     magnitude,
        "direction":     direction,
        "source":        source,
        "description":   description,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Source 1 — yfinance earnings
# ══════════════════════════════════════════════════════════════════════════════

section("SOURCE 1 — yfinance Earnings")

rows = []
ticker = yf.Ticker("AAPL")

# EPS: get_earnings_dates gives ~25 quarters back (~2020-present)
print("\n  Fetching EPS data via get_earnings_dates ...")
try:
    eps_df = ticker.get_earnings_dates(limit=100)
    eps_df = eps_df.dropna(subset=["Reported EPS"])  # drop future/missing
    eps_df.index = pd.to_datetime(eps_df.index).normalize()
    eps_df = eps_df.sort_index()

    for date, row in eps_df.iterrows():
        actual   = row["Reported EPS"]
        estimate = row["EPS Estimate"]
        surprise = row["Surprise(%)"]   # already in %

        mag = float(surprise) if pd.notna(surprise) else float(
            (actual - estimate) / abs(estimate) * 100 if estimate and estimate != 0 else 0
        )
        rows.append(make_event_row(
            date         = date,
            event_type   = "earnings",
            event_subtype= "eps_surprise",
            magnitude    = round(mag, 4),
            direction    = direction_from(mag),
            source       = "yfinance",
            description  = (
                f"AAPL Q earnings: EPS actual={actual:.2f} vs est={estimate:.2f} "
                f"({mag:+.1f}%)"
            ),
        ))
    print(f"  Collected {len(eps_df)} EPS quarters "
          f"({eps_df.index.min().date()} to {eps_df.index.max().date()})")
except Exception as e:
    print(f"  WARNING: EPS fetch failed: {e}")

# Revenue: quarterly_income_stmt gives last 5 quarters
print("\n  Fetching quarterly revenue ...")
try:
    qi = ticker.quarterly_income_stmt
    rev = qi.loc["Total Revenue"].sort_index()
    rev_df = rev.to_frame("revenue_actual")
    rev_df.index = pd.to_datetime(rev_df.index).normalize()

    # Revenue surprise proxy: QoQ change
    rev_df["rev_qoq"] = rev_df["revenue_actual"].pct_change() * 100

    for date, row in rev_df.iterrows():
        actual = row["revenue_actual"]
        qoq    = row["rev_qoq"]
        if pd.isna(actual):
            continue
        rows.append(make_event_row(
            date         = date,
            event_type   = "earnings",
            event_subtype= "revenue",
            magnitude    = round(float(qoq), 4) if pd.notna(qoq) else np.nan,
            direction    = direction_from(qoq) if pd.notna(qoq) else "neutral",
            source       = "yfinance",
            description  = (
                f"AAPL Q revenue: ${actual/1e9:.2f}B "
                f"(QoQ {qoq:+.1f}%)" if pd.notna(qoq) else
                f"AAPL Q revenue: ${actual/1e9:.2f}B"
            ),
        ))
    print(f"  Collected {len(rev_df)} revenue quarters "
          f"({rev_df.index.min().date()} to {rev_df.index.max().date()})")
except Exception as e:
    print(f"  WARNING: Revenue fetch failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Source 2 — FRED macroeconomic series
# ══════════════════════════════════════════════════════════════════════════════

section("SOURCE 2 — FRED Macroeconomic Data")

FRED_SERIES = {
    "FEDFUNDS": {
        "name"  : "Fed Funds Rate",
        "subtype": "fed_rate_change",
        "change": "mom",          # month-over-month change in rate level (bps)
        "unit"  : "bps",
        "positive_is_good": False, # rate hike = tighter = negative for stocks
        "desc"  : "Fed Funds Rate: {val:.2f}% (change {chg:+.0f}bps)",
    },
    "CPIAUCSL": {
        "name"  : "CPI",
        "subtype": "cpi_change",
        "change": "mom_pct",
        "unit"  : "%",
        "positive_is_good": False, # higher inflation = negative
        "desc"  : "CPI: {val:.3f} (MoM {chg:+.3f}%)",
    },
    "UNRATE": {
        "name"  : "Unemployment Rate",
        "subtype": "unemployment_change",
        "change": "mom",
        "unit"  : "pp",
        "positive_is_good": False,
        "desc"  : "Unemployment: {val:.1f}% (change {chg:+.2f}pp)",
    },
    "GDP": {
        "name"  : "GDP",
        "subtype": "gdp_change",
        "change": "qoq_pct",
        "unit"  : "%",
        "positive_is_good": True,
        "desc"  : "GDP: ${val:.0f}B (QoQ {chg:+.2f}%)",
    },
}

for series_id, cfg in FRED_SERIES.items():
    print(f"\n  Fetching {series_id} ({cfg['name']}) ...")
    try:
        s = fetch_fred(series_id)
        s = s[s.index >= FRED_START]

        if cfg["change"] == "mom":
            chg = (s - s.shift(1)) * 100   # level change in bps/pp
        elif cfg["change"] == "mom_pct":
            chg = s.pct_change() * 100
        elif cfg["change"] == "qoq_pct":
            chg = s.pct_change() * 100
        else:
            chg = s.diff()

        series_rows = 0
        for date, val in s.items():
            c = chg.get(date, np.nan)
            if pd.isna(c) and date == s.index[0]:
                continue    # skip first row with no change
            mag = round(float(c), 6) if pd.notna(c) else np.nan
            try:
                desc = cfg["desc"].format(val=val, chg=c if pd.notna(c) else 0)
            except Exception:
                desc = f"{cfg['name']}: {val}"

            rows.append(make_event_row(
                date         = date,
                event_type   = "macro",
                event_subtype= cfg["subtype"],
                magnitude    = mag,
                direction    = direction_from(c, cfg["positive_is_good"]) if pd.notna(c) else "neutral",
                source       = "fred",
                description  = desc,
            ))
            series_rows += 1

        print(f"  Collected {series_rows} {series_id} observations "
              f"({s.index.min().date()} to {s.index.max().date()})")
    except Exception as e:
        print(f"  WARNING: {series_id} fetch failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Source 3 — Apple product events (hardcoded)
# ══════════════════════════════════════════════════════════════════════════════

section("SOURCE 3 — Apple Product Events (hardcoded)")

APPLE_EVENTS = [
    # ── Stock splits ──────────────────────────────────────────────────────────
    ("2000-06-21", "split",   "stock_split",    2.0, "positive", "2-for-1 stock split"),
    ("2005-02-28", "split",   "stock_split",    2.0, "positive", "2-for-1 stock split"),
    ("2014-06-09", "split",   "stock_split",    7.0, "positive", "7-for-1 stock split"),
    ("2020-08-31", "split",   "stock_split",    4.0, "positive", "4-for-1 stock split"),

    # ── Major product launches ────────────────────────────────────────────────
    ("2001-10-23", "product", "ipod_launch",    None, "positive", "iPod launch"),
    ("2003-04-28", "product", "itunes_launch",  None, "positive", "iTunes Store launch"),
    ("2007-01-09", "product", "iphone_launch",  1,    "positive", "iPhone 1 announcement (Macworld)"),
    ("2008-07-11", "product", "iphone_launch",  2,    "positive", "iPhone 3G launch (App Store)"),
    ("2009-06-19", "product", "iphone_launch",  3,    "positive", "iPhone 3GS launch"),
    ("2010-01-27", "product", "ipad_launch",    1,    "positive", "iPad 1 announcement"),
    ("2010-06-24", "product", "iphone_launch",  4,    "positive", "iPhone 4 launch"),
    ("2011-10-14", "product", "iphone_launch",  4,    "positive", "iPhone 4S launch (first with Siri)"),
    ("2012-09-21", "product", "iphone_launch",  5,    "positive", "iPhone 5 launch"),
    ("2013-09-20", "product", "iphone_launch",  5,    "positive", "iPhone 5s/5c launch"),
    ("2014-09-19", "product", "iphone_launch",  6,    "positive", "iPhone 6/6 Plus launch"),
    ("2014-11-18", "product", "apple_watch_announce", None, "positive", "Apple Watch announcement"),
    ("2015-09-25", "product", "iphone_launch",  6,    "positive", "iPhone 6s/6s Plus launch"),
    ("2015-04-24", "product", "apple_watch_launch",   None, "positive", "Apple Watch launch"),
    ("2016-09-16", "product", "iphone_launch",  7,    "positive", "iPhone 7/7 Plus launch"),
    ("2016-09-07", "product", "airpods_announce", None, "positive", "AirPods announcement"),
    ("2016-12-13", "product", "airpods_launch", None, "positive", "AirPods launch"),
    ("2017-11-03", "product", "iphone_launch",  8,    "positive", "iPhone X launch"),
    ("2018-10-26", "product", "iphone_launch",  9,    "positive", "iPhone XS/XR launch"),
    ("2019-09-20", "product", "iphone_launch",  11,   "positive", "iPhone 11 launch"),
    ("2020-10-23", "product", "iphone_launch",  12,   "positive", "iPhone 12 launch (first 5G)"),
    ("2020-11-17", "product", "apple_silicon",  None, "positive", "Apple Silicon M1 launch (MacBook)"),
    ("2021-09-24", "product", "iphone_launch",  13,   "positive", "iPhone 13 launch"),
    ("2022-09-16", "product", "iphone_launch",  14,   "positive", "iPhone 14 launch"),
    ("2023-06-05", "product", "vision_pro_announce", None, "positive", "Apple Vision Pro announcement (WWDC)"),
    ("2023-09-22", "product", "iphone_launch",  15,   "positive", "iPhone 15 launch"),
    ("2024-02-02", "product", "vision_pro_launch", None, "positive", "Apple Vision Pro launch"),
    ("2024-09-20", "product", "iphone_launch",  16,   "positive", "iPhone 16 launch"),
    ("2025-09-19", "product", "iphone_launch",  17,   "positive", "iPhone 17 launch (expected)"),

    # ── WWDC (annual developer conference, usually June) ──────────────────────
    ("2000-05-15", "event",   "wwdc",           None, "neutral", "WWDC 2000"),
    ("2001-05-21", "event",   "wwdc",           None, "neutral", "WWDC 2001"),
    ("2002-05-06", "event",   "wwdc",           None, "neutral", "WWDC 2002"),
    ("2003-06-23", "event",   "wwdc",           None, "neutral", "WWDC 2003"),
    ("2004-06-28", "event",   "wwdc",           None, "neutral", "WWDC 2004"),
    ("2005-06-06", "event",   "wwdc",           None, "neutral", "WWDC 2005 — Intel transition announced"),
    ("2006-08-07", "event",   "wwdc",           None, "neutral", "WWDC 2006"),
    ("2007-06-11", "event",   "wwdc",           None, "neutral", "WWDC 2007 — iPhone SDK announced"),
    ("2008-06-09", "event",   "wwdc",           None, "neutral", "WWDC 2008 — App Store announced"),
    ("2009-06-08", "event",   "wwdc",           None, "neutral", "WWDC 2009"),
    ("2010-06-07", "event",   "wwdc",           None, "neutral", "WWDC 2010"),
    ("2011-06-06", "event",   "wwdc",           None, "neutral", "WWDC 2011 — iCloud announced"),
    ("2012-06-11", "event",   "wwdc",           None, "neutral", "WWDC 2012"),
    ("2013-06-10", "event",   "wwdc",           None, "neutral", "WWDC 2013"),
    ("2014-06-02", "event",   "wwdc",           None, "neutral", "WWDC 2014 — Swift announced"),
    ("2015-06-08", "event",   "wwdc",           None, "neutral", "WWDC 2015"),
    ("2016-06-13", "event",   "wwdc",           None, "neutral", "WWDC 2016"),
    ("2017-06-05", "event",   "wwdc",           None, "neutral", "WWDC 2017"),
    ("2018-06-04", "event",   "wwdc",           None, "neutral", "WWDC 2018"),
    ("2019-06-03", "event",   "wwdc",           None, "neutral", "WWDC 2019 — Mac Pro announced"),
    ("2020-06-22", "event",   "wwdc",           None, "neutral", "WWDC 2020 — Apple Silicon announced"),
    ("2021-06-07", "event",   "wwdc",           None, "neutral", "WWDC 2021"),
    ("2022-06-06", "event",   "wwdc",           None, "neutral", "WWDC 2022"),
    ("2023-06-05", "event",   "wwdc",           None, "positive","WWDC 2023 — Vision Pro + AI features"),
    ("2024-06-10", "event",   "wwdc",           None, "positive","WWDC 2024 — Apple Intelligence announced"),
    ("2025-06-09", "event",   "wwdc",           None, "neutral", "WWDC 2025 (expected)"),

    # ── CEO / leadership transitions ──────────────────────────────────────────
    ("2011-08-24", "event",   "ceo_transition", None, "negative", "Steve Jobs resigns as CEO; Tim Cook takes over"),
    ("2011-10-05", "event",   "ceo_transition", None, "negative", "Steve Jobs death"),

    # ── Services milestones ───────────────────────────────────────────────────
    ("2019-03-25", "event",   "services_launch", None, "positive", "Apple TV+ and Apple Arcade announced"),
    ("2019-11-01", "event",   "services_launch", None, "positive", "Apple TV+ launches"),
    ("2022-10-24", "event",   "services_milestone", None, "positive", "Apple Services revenue crosses $20B/quarter"),
]

apple_rows = []
for date_str, etype, esubtype, mag, dirn, desc in APPLE_EVENTS:
    apple_rows.append(make_event_row(
        date         = date_str,
        event_type   = etype,
        event_subtype= esubtype,
        magnitude    = float(mag) if mag is not None else np.nan,
        direction    = dirn,
        source       = "hardcoded",
        description  = desc,
    ))

rows.extend(apple_rows)
print(f"\n  Collected {len(apple_rows)} Apple product/event records")


# ══════════════════════════════════════════════════════════════════════════════
# Assemble and save
# ══════════════════════════════════════════════════════════════════════════════

section("ASSEMBLING UNIFIED EVENT TABLE")

events = pd.DataFrame(rows)
events["date"] = pd.to_datetime(events["date"])
events = events.sort_values("date").reset_index(drop=True)
events = events.set_index("date")
events.index.name = "date"

print(f"\n  Total events : {len(events):,}")
print(f"  Date range   : {events.index.min().date()} to {events.index.max().date()}")
print(f"  Columns      : {events.columns.tolist()}")

print(f"\n  Event counts by type:")
for etype, grp in events.groupby("event_type"):
    subtypes = grp["event_subtype"].value_counts()
    print(f"\n    {etype} ({len(grp)} total)")
    for sub, cnt in subtypes.items():
        print(f"      {sub:<30}  {cnt:>4} events")

print(f"\n  Direction distribution:")
for d, cnt in events["direction"].value_counts().items():
    print(f"    {d:<12}  {cnt:>4} ({cnt/len(events)*100:.1f}%)")

print(f"\n  Source distribution:")
for s, cnt in events["source"].value_counts().items():
    print(f"    {s:<12}  {cnt:>4} ({cnt/len(events)*100:.1f}%)")

# FRED coverage vs price data window (1995-2026)
print(f"\n  FRED coverage within price data window (1995-2026):")
price_window = events[
    (events.index >= "1995-01-01") &
    (events["source"] == "fred")
]
for sub, cnt in price_window["event_subtype"].value_counts().items():
    print(f"    {sub:<30}  {cnt:>4} observations")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
events.to_parquet(OUT_FILE)
print(f"\n  Saved -> {OUT_FILE}")
print(f"\nPhase 3 Step 1 complete. Run 09_event_features.py next.\n")
