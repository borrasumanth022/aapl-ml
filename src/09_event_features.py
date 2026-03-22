"""
Phase 3 - Step 2: Event Feature Engineering
============================================
Loads aapl_labeled.parquet + aapl_events.parquet and builds event-aware
features for every trading day.

Feature groups:
  A. Earnings       — proximity, surprise magnitude, streak
  B. Macro (FRED)   — rate level/changes, CPI YoY, unemployment
  C. Product cycle  — proximity to Apple events, iPhone cycle flag
  D. Regime         — rate environment, inflation regime (categorical → int)

All features are strictly backward-looking on each trading day (no lookahead).
FRED monthly data is forward-filled to daily — each day sees only the most
recently published observation.

Output: data/processed/aapl_with_events.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import warnings

import numpy as np
import pandas as pd
import requests
from config import paths as P, settings as S

warnings.filterwarnings("ignore")

LABELED_FILE  = P.DATA_LABELED
EVENTS_FILE   = P.DATA_EVENTS
OUT_FILE      = P.DATA_WITH_EVENTS

FRED_BASE     = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
CAP_EARNINGS  = S.CAP_EARNINGS
CAP_PRODUCT   = S.CAP_PRODUCT
IPHONE_WINDOW = S.IPHONE_WINDOW

INFLATION_HIGH   = S.INFLATION_HIGH
INFLATION_LOW    = S.INFLATION_LOW
RATE_RISING_BPS  = S.RATE_RISING_BPS
RATE_FALLING_BPS = S.RATE_FALLING_BPS


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


def fetch_fred_level(series_id, start="1993-01-01"):
    """Download a FRED series as a daily forward-filled Series (levels only)."""
    url = FRED_BASE.format(series_id)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text),
                     parse_dates=["observation_date"],
                     index_col="observation_date")
    df.columns = [series_id]
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    s = df[series_id].dropna().sort_index()
    return s[s.index >= start]


def _to_day_ints(dates) -> np.ndarray:
    """
    Convert any DatetimeIndex / array of Timestamps to integer days since
    1970-01-01. Normalises resolution differences (ms vs us vs ns) by casting
    to datetime64[D] first, so comparisons are always in whole-day units.
    """
    return pd.DatetimeIndex(dates).normalize().values.astype("datetime64[D]").astype("int64")


def days_to_next(trading_dates: pd.DatetimeIndex,
                 event_dates) -> np.ndarray:
    """
    For each date in trading_dates return calendar days to the next event date
    (strictly after). Returns np.nan where none exists.
    """
    trade_d = _to_day_ints(trading_dates)
    event_d = np.sort(_to_day_ints(event_dates))

    idx = np.searchsorted(event_d, trade_d, side="right")
    result = np.full(len(trade_d), np.nan)
    mask = idx < len(event_d)
    result[mask] = (event_d[idx[mask]] - trade_d[mask]).astype(float)
    return result


def days_since_last(trading_dates: pd.DatetimeIndex,
                    event_dates) -> np.ndarray:
    """
    For each date in trading_dates return calendar days since the most recent
    past event date (on or before that day). Returns np.nan where none exists.
    """
    trade_d = _to_day_ints(trading_dates)
    event_d = np.sort(_to_day_ints(event_dates))

    idx = np.searchsorted(event_d, trade_d, side="right") - 1
    result = np.full(len(trade_d), np.nan)
    mask = idx >= 0
    result[mask] = (trade_d[mask] - event_d[idx[mask]]).astype(float)
    return result


def align_monthly_to_daily(monthly: pd.Series,
                            daily_index: pd.DatetimeIndex) -> pd.Series:
    """
    Forward-fill a monthly series onto a daily index.
    Reindexes to daily and ffills — each trading day sees only the most recent
    monthly observation (published before or on that date).
    """
    combined = monthly.reindex(
        monthly.index.union(daily_index)
    ).sort_index().ffill()
    return combined.reindex(daily_index)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Load base data ─────────────────────────────────────────────────────────
    print("\nLoading base datasets ...")
    base    = pd.read_parquet(LABELED_FILE)
    events  = pd.read_parquet(EVENTS_FILE)

    base.index   = pd.to_datetime(base.index).normalize()
    events.index = pd.to_datetime(events.index).normalize()
    base = base.sort_index()

    trading_dates = base.index
    print(f"  Labeled rows  : {len(base):,}  ({trading_dates.min().date()} to {trading_dates.max().date()})")
    print(f"  Events rows   : {len(events):,}")

    feat = pd.DataFrame(index=trading_dates)   # all new features land here

    # ══════════════════════════════════════════════════════════════════════════
    # A. Earnings features
    # ══════════════════════════════════════════════════════════════════════════
    section("A. Earnings Features")

    eps_events = events[events["event_subtype"] == "eps_surprise"].copy()
    eps_events  = eps_events.sort_index()
    eps_dates   = eps_events.index.unique()

    print(f"  EPS events available: {len(eps_events)}  "
          f"({eps_dates.min().date()} to {eps_dates.max().date()})")

    # ── Days to / since earnings ───────────────────────────────────────────────
    feat["days_to_next_earnings"] = np.minimum(
        days_to_next(trading_dates, eps_dates.values), CAP_EARNINGS
    )
    feat["days_since_last_earnings"] = np.minimum(
        days_since_last(trading_dates, eps_dates.values), CAP_EARNINGS
    )

    # ── has_earnings_data flag ────────────────────────────────────────────────
    feat["has_earnings_data"] = (~np.isnan(
        days_since_last(trading_dates, eps_dates.values)
    )).astype(int)

    # ── last_eps_surprise_pct — forward-fill from last known event ────────────
    eps_surprise_daily = pd.Series(np.nan, index=trading_dates, name="last_eps_surprise_pct")
    for dt, row in eps_events.iterrows():
        if dt in eps_surprise_daily.index:
            eps_surprise_daily.loc[dt] = row["magnitude"]
    eps_surprise_daily = eps_surprise_daily.groupby(level=0).last()
    feat["last_eps_surprise_pct"] = eps_surprise_daily.ffill().fillna(0.0)

    # ── earnings_streak — consecutive beats(+1) / misses(-1) ─────────────────
    eps_sorted = eps_events.sort_index()
    streak_by_date = {}
    streak = 0
    for dt, row in eps_sorted.iterrows():
        mag = row["magnitude"]
        if pd.isna(mag) or mag == 0:
            streak = 0
        elif mag > 0:
            streak = max(streak, 0) + 1
        else:
            streak = min(streak, 0) - 1
        streak_by_date[dt] = streak

    streak_daily = pd.Series(np.nan, index=trading_dates)
    for dt, val in streak_by_date.items():
        if dt in streak_daily.index:
            streak_daily.loc[dt] = val
    streak_daily = streak_daily.groupby(level=0).last()
    feat["earnings_streak"] = streak_daily.ffill().fillna(0).astype(int)

    # Coverage report
    for col in ["days_to_next_earnings", "days_since_last_earnings",
                "last_eps_surprise_pct", "earnings_streak", "has_earnings_data"]:
        n_populated = feat[col].notna().sum()
        pct = n_populated / len(feat) * 100
        print(f"  {col:<35}  {n_populated:>5,} / {len(feat):,}  ({pct:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # B. Macro features (re-fetch FRED levels)
    # ══════════════════════════════════════════════════════════════════════════
    section("B. Macro Features (FRED)")

    print("  Fetching FRED series ...")

    # Fed funds rate
    print("    FEDFUNDS ...", end=" ", flush=True)
    fedfunds = fetch_fred_level("FEDFUNDS")
    ff_daily = align_monthly_to_daily(fedfunds, trading_dates)
    feat["fed_rate_level"]     = ff_daily.values
    feat["fed_rate_change_1m"] = (fedfunds - fedfunds.shift(1)).reindex(
        fedfunds.index.union(trading_dates)
    ).ffill().reindex(trading_dates).values * 100   # bps
    feat["fed_rate_change_3m"] = (fedfunds - fedfunds.shift(3)).reindex(
        fedfunds.index.union(trading_dates)
    ).ffill().reindex(trading_dates).values * 100   # bps
    print(f"OK  ({len(fedfunds)} monthly obs)")

    # CPI — compute YoY change (inflation rate)
    print("    CPIAUCSL ...", end=" ", flush=True)
    cpi = fetch_fred_level("CPIAUCSL")
    cpi_yoy = ((cpi - cpi.shift(12)) / cpi.shift(12) * 100).dropna()
    feat["cpi_yoy_change"] = align_monthly_to_daily(cpi_yoy, trading_dates).values
    print(f"OK  ({len(cpi)} monthly obs, YoY from {cpi_yoy.index.min().date()})")

    # Unemployment
    print("    UNRATE    ...", end=" ", flush=True)
    unrate = fetch_fred_level("UNRATE")
    feat["unemployment_level"]     = align_monthly_to_daily(unrate, trading_dates).values
    ur_change_3m = (unrate - unrate.shift(3))
    feat["unemployment_change_3m"] = align_monthly_to_daily(
        ur_change_3m, trading_dates
    ).values
    print(f"OK  ({len(unrate)} monthly obs)")

    # Coverage report
    for col in ["fed_rate_level", "fed_rate_change_1m", "fed_rate_change_3m",
                "cpi_yoy_change", "unemployment_level", "unemployment_change_3m"]:
        n = feat[col].notna().sum()
        print(f"  {col:<35}  {n:>5,} / {len(feat):,}  ({n/len(feat)*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # C. Product cycle features
    # ══════════════════════════════════════════════════════════════════════════
    section("C. Product Cycle Features")

    product_events = events[
        events["event_type"].isin(["product", "event", "split"])
    ].copy()
    prod_dates = product_events.index.unique().sort_values()

    iphone_events = events[events["event_subtype"] == "iphone_launch"].copy()
    iphone_dates  = iphone_events.index.unique().sort_values()

    print(f"  Product/event dates : {len(prod_dates)}")
    print(f"  iPhone launch dates : {len(iphone_dates)}")

    feat["days_to_next_product_event"] = np.minimum(
        days_to_next(trading_dates, prod_dates.values), CAP_PRODUCT
    )
    feat["days_since_last_product_event"] = np.minimum(
        days_since_last(trading_dates, prod_dates.values), CAP_PRODUCT
    )

    # is_iphone_cycle: within IPHONE_WINDOW days of any iPhone launch (either side)
    days_to_iphone   = days_to_next(trading_dates, iphone_dates.values)
    days_from_iphone = days_since_last(trading_dates, iphone_dates.values)

    iphone_near_before = np.where(~np.isnan(days_to_iphone),   days_to_iphone,   999)
    iphone_near_after  = np.where(~np.isnan(days_from_iphone), days_from_iphone, 999)
    feat["is_iphone_cycle"] = (
        (iphone_near_before <= IPHONE_WINDOW) |
        (iphone_near_after  <= IPHONE_WINDOW)
    ).astype(int)

    for col in ["days_to_next_product_event", "days_since_last_product_event",
                "is_iphone_cycle"]:
        n = feat[col].notna().sum()
        print(f"  {col:<35}  {n:>5,} / {len(feat):,}  ({n/len(feat)*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # D. Macro regime features
    # ══════════════════════════════════════════════════════════════════════════
    section("D. Macro Regime Features (derived)")

    rate_3m = feat["fed_rate_change_3m"]
    rate_env_cat = pd.Series("stable", index=trading_dates)
    rate_env_cat[rate_3m >  RATE_RISING_BPS]  = "rising"
    rate_env_cat[rate_3m <  RATE_FALLING_BPS] = "falling"
    rate_env_cat[rate_3m.isna()]               = "stable"
    rate_env_map = {"falling": -1, "stable": 0, "rising": 1}
    feat["rate_environment"] = rate_env_cat.map(rate_env_map).astype(int)

    cpi_yoy_s = feat["cpi_yoy_change"]
    inf_env_cat = pd.Series("normal", index=trading_dates)
    inf_env_cat[cpi_yoy_s >= INFLATION_HIGH] = "high"
    inf_env_cat[cpi_yoy_s <  INFLATION_LOW]  = "low"
    inf_env_cat[cpi_yoy_s.isna()]            = "normal"
    inf_env_map = {"low": -1, "normal": 0, "high": 1}
    feat["inflation_regime"] = inf_env_cat.map(inf_env_map).astype(int)

    rate_dist = rate_env_cat.value_counts()
    inf_dist  = inf_env_cat.value_counts()
    print(f"  rate_environment  : {dict(rate_dist)}")
    print(f"  inflation_regime  : {dict(inf_dist)}")

    for col in ["rate_environment", "inflation_regime"]:
        n = feat[col].notna().sum()
        print(f"  {col:<35}  {n:>5,} / {len(feat):,}  ({n/len(feat)*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # E. Merge into base and save
    # ══════════════════════════════════════════════════════════════════════════
    section("MERGING + SAVING")

    new_cols = feat.columns.tolist()
    print(f"\n  Event features engineered : {len(new_cols)}")

    combined = base.join(feat, how="left")

    print(f"  Combined shape            : {combined.shape}")
    print(f"  New columns added         :")
    for col in new_cols:
        n      = combined[col].notna().sum()
        pct    = n / len(combined) * 100
        sample = combined[col].dropna()
        if len(sample):
            mn, mx = sample.min(), sample.max()
            print(f"    {col:<35}  {pct:>6.1f}% populated  "
                  f"[{mn:.2f}, {mx:.2f}]")
        else:
            print(f"    {col:<35}  0.0% populated")

    print(f"\n  Overall feature coverage (% rows with non-null value):")
    coverage = (combined[new_cols].notna().mean() * 100).sort_values(ascending=False)
    fully_covered = (coverage == 100).sum()
    partial = ((coverage > 0) & (coverage < 100)).sum()
    zero_cov = (coverage == 0).sum()
    print(f"    100% coverage : {fully_covered} features")
    print(f"    Partial       : {partial} features")
    print(f"    0% coverage   : {zero_cov} features")

    pre_earnings_rows = (combined["has_earnings_data"] == 0).sum()
    print(f"\n  Rows without earnings history : {pre_earnings_rows:,} "
          f"({pre_earnings_rows/len(combined)*100:.1f}%)  -- pre-2005")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_FILE)
    print(f"\n  Saved -> {OUT_FILE}")
    print(f"  Final shape : {combined.shape[0]:,} rows x {combined.shape[1]} columns")
    print(f"\nPhase 3 Step 2 complete. Run src/10_retrain_with_events.py next.\n")
