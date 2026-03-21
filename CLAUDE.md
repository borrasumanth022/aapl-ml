# AAPL ML — Project Briefing for Claude Code

## What we're building
A machine learning system that:
1. Analyzes historical market data to recognize patterns
2. Links market behavior to real-world events (earnings, Fed decisions, macro news)
3. Eventually predicts future market reactions based on learned patterns

This is a phased project — we start simple and add complexity gradually.

## Current status
- Phase 1 (Data Foundation) — COMPLETE
  - Step 1 (fetch raw data): COMPLETE — 7,856 days of AAPL OHLCV data (1995–2026)
  - Step 2 (feature engineering): COMPLETE — 7,657 rows x 57 columns → aapl_features.parquet
  - Step 3 (labels): COMPLETE — 7,405 rows x 77 columns → aapl_labeled.parquet (20 label cols, 5 horizons)
- Phase 2 (Pattern Recognition) — COMPLETE
  - Step 1 (feature selection): COMPLETE — 36 features selected → models/feature_list.json
    - Started from 52 candidates (after dropping labels + OHLCV)
    - Dropped 12 price-level proxies (raw SMAs, EMAs, bb_upper/lower, atr_14, macd, log_close)
    - Dropped 4 correlation duplicates (log_return_1d, quarter, roc_5, williams_r; threshold |r|>0.95)
  - Step 2 (baseline XGBoost): COMPLETE — model saved → models/xgb_dir_1m.pkl
    - Target: dir_1m (3-class: Bear=-1, Sideways=0, Bull=+1)
    - Walk-forward validation: 5 expanding windows (1995–2025), ~1,225 test rows each
    - OOS accuracy: 45.89% | Naive baseline (always Bull): 52.90% — BELOW baseline by 7.00pp
    - Per-class OOS accuracy: Bear=28.5%, Sideways=2.5%, Bull=70.2%
    - Class distribution: Bull 51.9%, Bear 31.4%, Sideways 16.7% — model biased toward Bull
    - Model badly struggles with Sideways class (F1=0.045) — known issue for next iteration
    - Top features by gain: cross_50_200, month, hvol_63d, close_vs_sma200, atr_pct
    - Predictions saved → data/processed/aapl_predictions.parquet (6,125 OOS rows)
    - XGBoost params: n_estimators=300, max_depth=4, lr=0.05, subsample=0.8, min_child_weight=20

  - Step 3 (best model): COMPLETE — dir_1w + class-balanced weights → models/xgb_dir_1w_weighted.pkl
    - Experiment A (dir_1w, unweighted): OOS 38.71% — BEATS naive 37.50% by +1.21pp; Sideways recall 39%
    - Experiment B (dir_1m weighted): OOS 40.96% — below naive 52.90% by 11.93pp; accuracy loss > recall gain
    - Best model (dir_1w + weighted): OOS 38.35%, BEATS naive 37.50% by +0.85pp
    - Per-class OOS: Bear=23.1%, Sideways=45.5%, Bull=41.8%  (macro F1=0.367 vs baseline 0.314)
    - Weighting improves calibration but trades some Bull accuracy for Sideways; acceptable tradeoff
    - Predictions saved → data/processed/aapl_predictions_best.parquet (6,125 OOS rows)
    - SHAP analysis: shap_summary.csv saved; key findings below

  SHAP key findings (Phase 2 best model):
    - Top overall driver : atr_pct (dominates Sideways predictions especially)
    - Top Bear driver    : hvol_21d (21-day historical vol spike = bear signal)
    - Top Sideways driver: atr_pct (low ATR = range-bound = sideways)
    - Top Bull driver    : hvol_63d (longer-term vol context) + price_52w_pct
    - Class-specialised  : atr_pct/hvol most Bear/Sideways specific; price_52w_pct most Bull specific
    - Volatility features dominate — suggesting technical features mostly capture fear/uncertainty,
      not directional momentum. Strong motivation for Phase 3 event features.

- Phase 3 (Event Linkage) — COMPLETE
  - Step 1 (event data collection): COMPLETE — 1,474 events → data/processed/aapl_events.parquet
    - Source 1 (yfinance earnings): 85 EPS surprise quarters (2005–2026) + 5 revenue quarters
      - yfinance caps get_earnings_dates at 100 rows; revenue only last 5 quarters available
    - Source 2 (FRED macro): 1,320 monthly/quarterly observations (1993–2026)
      - FEDFUNDS: 397 obs | CPIAUCSL: 396 | UNRATE: 396 | GDP: 131
      - Downloaded keyless via fredgraph.csv endpoint (no API key needed)
      - MoM/QoQ changes computed as surprise proxies
    - Source 3 (hardcoded Apple events): 64 records
      - 19 iPhone launches (2007–2025), 26 WWDC dates, stock splits (2000/2005/2014/2020)
      - Major launches: iPod, iPad, Apple Watch, AirPods, Apple Silicon, Vision Pro
    - Schema: date, event_type, event_subtype, magnitude, direction, source, description
    - Coverage within price window (1995–2026): 374 FEDFUNDS + 373 CPI/UNRATE + 124 GDP

  - Step 2 (event feature engineering): COMPLETE — 16 new features → aapl_with_events.parquet
    - Output: 7,405 rows x 93 columns (77 original + 16 event features)
    - Key bug fixed: mixed datetime64[ms]/[us] resolutions required normalization to datetime64[D]
    - Earnings (5 features): days_to/since earnings, EPS surprise, streak, has_data flag
      - days_since_last_earnings: 68.6% populated (pre-2005 rows have no earnings history)
      - last_eps_surprise_pct: 100% (0.0 for pre-2005 rows), range [-10.1%, +52.9%]
      - earnings_streak: max +27 consecutive beats, min -2 misses in OOS history
    - Macro/FRED (6 features): rate level/change 1m+3m, CPI YoY, unemployment level+3m change
      - All 100% populated (FRED goes back to 1993, before our 1995 price data)
      - fed_rate_level: [0.05%, 6.54%] | cpi_yoy_change: [-1.96%, 8.98%]
    - Product cycle (3 features): days to/since Apple events, is_iphone_cycle flag
      - days_since_last_product_event: 84.4% (first Apple event in data is 2000-05-15)
      - is_iphone_cycle: [0, 1] — correctly flags ±60 day windows around launches
    - Regime (2 features): rate_environment (-1/0/1), inflation_regime (-1/0/1)
      - rate rising: 1,996 days | stable: 4,000 | falling: 1,409
      - inflation high (>4%): 837 days | normal: 5,035 | low (<1.5%): 1,533

  - Step 3 (retrain with events): COMPLETE — models/xgb_dir_1w_events.pkl
    - Features: 36 technical + 16 event = 52 total
    - NaN sentinels: days_since_last_earnings=90 (pre-2005), days_since_last_product_event=180 (pre-2000)

    RESULTS vs Phase 2 best model (xgb_dir_1w_weighted):
    Metric              Phase 2    Phase 3    Delta
    OOS Accuracy        38.35%     36.28%     -2.07pp
    Macro F1            0.367      0.354      -0.013
    Bear Recall         23.1%      28.3%      +5.2pp   <-- improved
    Sideways Recall     45.5%      35.9%      -9.6pp   <-- degraded
    Bull Recall         41.8%      42.3%      +0.5pp

    Key finding: event features HURT overall performance (Sideways recall -9.6pp)
    but specifically IMPROVED Bear recall (+5.2pp) — the macro/rate features help
    identify bear conditions but add noise to sideways detection.

    SHAP top event features:
    1. fed_rate_change_3m  (SHAP mean=0.056) — strongest for Bear AND Bull
    2. cpi_yoy_change       (SHAP mean=0.048) — strongest for Bear
    3. last_eps_surprise_pct(SHAP mean=0.045) — strongest for Bull (beat -> bull)
    4. fed_rate_change_1m   (SHAP mean=0.038) — strongest for Bull
    - rate_environment and has_earnings_data have near-zero SHAP despite high gain
      (gain = how often feature is used; SHAP = actual impact on prediction)
    - Earnings proximity (days_to/since_earnings) drives Sideways detection
    - Unemployment features strongest for Bear
    - iPhone cycle / product events help Bear (pre-launch uncertainty?)

    CONCLUSION: Phase 2 best model remains champion (38.35%, F1=0.367).
    Event features need better engineering before they add consistent value.
    Possible next steps: interaction features (earnings_in_5d × eps_surprise_pct),
    time-since-last-rate-hike as a regime indicator, or moving to a longer horizon
    (dir_1m) where macro context has more predictive power.

  - Step 4 (interaction features): COMPLETE — models/xgb_best_interactions.pkl
    - 5 interaction features added (57 total: 36 tech + 16 event + 5 interaction)
    - earnings_proximity_surprise = (90-days_to_next) * |last_eps_surprise_pct|
    - macro_stress_score = zscore(fed_change_3m) + zscore(cpi_yoy) + zscore(unrate_change)
    - vol_macro_interaction = hvol_21d * macro_stress_score
    - earnings_momentum = last_eps_surprise_pct * earnings_streak
    - rate_vol_regime = fed_rate_change_3m * hvol_63d

    FULL COMPARISON TABLE:
    Model                       Acc      F1     Bear   Side   Bull
    Phase 2 dir_1w (best)      38.35%  0.367   23.1%  45.5%  41.8%  <- previous champion
    Phase 3 dir_1w (events)    36.28%  0.354   28.3%  35.9%  42.3%
    Phase 3 dir_1w (interact)  38.30%  0.375   30.6%  39.9%  42.0%  <- NEW CHAMPION
    Phase 2 dir_1m (weighted)  40.96%  0.343   28.3%  19.5%  55.2%
    Phase 3 dir_1m (interact)  40.98%  0.345   38.4%  14.6%  51.3%

    WINNER: dir_1w + 57 features (interaction model)
    - Macro F1 0.375 vs Phase 2 best 0.367 (+0.008) — first time we beat Phase 2
    - Bear recall 30.6% vs 23.1% (+7.5pp) — biggest improvement across all experiments
    - Sideways recall 39.9% vs 45.5% (-5.6pp) — still a tradeoff but narrower
    - dir_1m Bear recall improved +10.1pp but Sideways too low (14.6%) to be useful

    SHAP interaction feature ranks (Model A winner):
    1. rate_vol_regime              rank #3   mean=0.0486  (Bear + Bull)
    2. earnings_proximity_surprise  rank #8   mean=0.0394  (Bull strongest)
    3. macro_stress_score           rank #19  mean=0.0287  (Bull)
    4. vol_macro_interaction        rank #38  mean=0.0142  (Bear)
    5. earnings_momentum            rank #45  mean=0.0107  (Bear)
    - rate_vol_regime is the standout: rate change × vol context captures regime transitions
    - earnings_proximity_surprise enters top 10 — confirms earnings timing matters

    PHASE 3 CONCLUSION:
    Best model = xgb_best_interactions.pkl (dir_1w, 57 features)
    OOS 38.30% | Macro F1 0.375 | Bear 30.6% | Sideways 39.9% | Bull 42.0%
    Beats Phase 2 champion on F1 and Bear recall. Ready for Phase 4.
    Champion checkpoint: models/xgb_phase3_champion.pkl
    Predictions checkpoint: data/processed/aapl_predictions_phase3_champion.parquet

- Phase 4 (Fusion Model) — IN PROGRESS
  Plan: Step 1 → LightGBM on same 57 features (Option A)
        Step 2 → Ensemble XGBoost + LightGBM proba outputs via meta-learner (Option C)
  Goal: Beat F1=0.375 (Phase 3 champion)

  - Step 1 (LightGBM): TODO
    - Same 57 features, same dir_1w target, same 5-fold walk-forward
    - Compare directly against XGBoost interaction model
    - Save lgbm_dir_1w.pkl

  - Step 2 (Ensemble): TODO
    - XGBoost + LightGBM probability outputs (6 proba columns each = 12 meta-features)
    - Meta-learner: LogisticRegression (simple, avoids overfitting on small meta-dataset)
    - Walk-forward: generate OOS probas in each fold → train meta on prior folds → predict next
    - Save ensemble_meta.pkl + xgb_phase3_champion.pkl (already saved)

## Immediate next task
Phase 4 Step 1 — LightGBM baseline on 57 features.

Note: Use Anaconda python directly:
  C:\Users\borra\anaconda3\python.exe pipeline\<script>.py

## Project roadmap (5 phases)
1. Data Foundation        — OHLCV data, technical features, forward return labels ← WE ARE HERE
2. Pattern Recognition    — Baseline models (XGBoost, LSTM) on price data alone
3. Event Linkage          — NLP layer linking news/macro events to market reactions
4. Fusion Model           — Combine price patterns + event signals
5. Prediction Engine      — Multi-horizon predictions (long → medium → short → intraday)

## Prediction targets (phased)
- First: Direction (up / sideways / down)
- Then: Magnitude (% return)
- Then: Volatility / risk level
- Eventually: Intraday signals

## Markets in scope
- Start: AAPL (single stock to prove the pipeline)
- Expand to: Other individual stocks, S&P 500, NASDAQ, Forex (EUR/USD, GBP/USD)

## Key design decisions already made
- Storage: Parquet files (not CSV, not SQL)
- Backtesting: Walk-forward validation only — no lookahead leakage
- Labels: 5 horizons (1w, 1m, 3m, 6m, 1y), both regression and classification
- Direction threshold: ±2% band for sideways label (avoids training on noise)
- Event framing: "When event type X occurs with surprise Y → reaction Z" (more tractable than raw price prediction)

## Environment
- OS: Windows
- Python: Anaconda base environment (Python 3.13.5)
- Editor: VS Code with Claude Code extension
- GPU: RTX 3060 (reserved for Phase 2 deep learning — not needed yet)

## Project structure
aapl_ml/
├── data/
│   ├── raw/                ← aapl_daily_raw.parquet (DONE)
│   └── processed/          ← aapl_features.parquet, aapl_labeled.parquet, aapl_predictions.parquet
├── pipeline/
│   ├── 01_fetch_data.py    ← DONE
│   ├── 02_features.py      ← DONE
│   ├── 03_labels.py        ← DONE
│   ├── 04_feature_selection.py  ← DONE
│   └── 05_train_baseline.py     ← DONE
├── models/
│   ├── feature_list.json        ← 36 selected features
│   ├── xgb_dir_1m.pkl           ← baseline model (dir_1m, unweighted)
│   ├── xgb_dir_1w.pkl           ← experiment A (dir_1w, unweighted)
│   ├── xgb_dir_1m_weighted.pkl  ← experiment B (dir_1m, weighted)
│   ├── xgb_dir_1w_weighted.pkl  ← PHASE 2 BEST MODEL (dir_1w + balanced weights)
│   └── shap_summary.csv         ← mean |SHAP| per feature per class
├── pipeline/
│   └── 08_events.py             ← Phase 3 Step 1: event data collection
├── notebooks/
│   └── 01_explore.ipynb    ← For data exploration after pipeline is complete
├── models/                 ← Saved models (Phase 2)
├── logs/                   ← Training logs
├── CLAUDE.md               ← This file
└── README.md

## Coding principles
- Keep each pipeline step as a standalone script (can be re-run independently)
- Print clear progress messages so we know what's happening
- Fail loudly with descriptive errors (don't silently continue on bad data)
- No lookahead bias — this is the #1 thing that kills ML trading models
- Prefer simple and working over complex and clever — prove value at each phase before adding complexity
