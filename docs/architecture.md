# Architecture — aapl_ml

## Goal
A machine learning system that learns to predict AAPL's weekly price direction (Bull/Sideways/Bear)
by combining technical indicators with macro and company-event signals.

## Data flow

```
                        External Sources
                        ─────────────────
                        yfinance (OHLCV)           ──→  src/01_fetch_data.py
                        yfinance (earnings)  ┐
                        FRED (macro)         ├──→  src/08_events.py
                        hardcoded events     ┘

Pipeline
─────────────────────────────────────────────────────────────────────
src/01_fetch_data.py        data/raw/aapl_daily_raw.parquet
        ↓
src/02_features.py          data/processed/aapl_features.parquet
        ↓                   (57 cols: OHLCV + 51 technical features)
src/03_labels.py            data/processed/aapl_labeled.parquet
        ↓                   (77 cols: +20 label cols across 5 horizons)
        ├─── src/04_feature_selection.py   models/feature_list.json
        │                                  (36 selected features)
        │
        │    models/feature_list.json
        │         ↓
        ├─── src/05_train_baseline.py      → Phase 2 baseline (dir_1m)
        ├─── src/06a_exp_dir1w.py          → Experiment A (dir_1w)
        ├─── src/06b_exp_weighted.py       → Experiment B (dir_1m weighted)
        └─── src/07_best_model.py          → Phase 2 champion (dir_1w weighted)

src/08_events.py            data/processed/aapl_events.parquet
        ↓                   (1,474 events: earnings, FRED macro, Apple events)

src/09_event_features.py    data/processed/aapl_with_events.parquet
(joins labeled + events)    (93 cols: 77 original + 16 event features)
        ↓
src/10_retrain_with_events.py  → Phase 3 Step 3 (52 features)
src/11_interaction_features.py → Phase 3 champion (57 features) ← CURRENT BEST

Phase 4 (in progress):
src/12_lgbm.py             → LightGBM baseline (57 features)
src/13_ensemble.py         → XGBoost + LightGBM meta-learner
```

## Key files

| File | Purpose |
|------|---------|
| `config/paths.py` | All canonical file paths |
| `config/settings.py` | Hyperparameters, label encoding, baselines |
| `models/feature_list.json` | 36 selected technical features |
| `data/processed/aapl_with_events.parquet` | Full dataset (57 potential features + labels) |
| `models/xgb_best_interactions.pkl` | Phase 3 champion model |
| `data/processed/aapl_predictions_interactions.parquet` | Phase 3 OOS predictions |

## Feature groups

### Technical (36 selected from 52 candidates)
- Price transforms: returns, volume z-score, 52-week range position
- Trend: close vs SMA (10/20/50/100/200), golden/death cross, MACD histogram
- Momentum: RSI 7/14, ROC 5/10/21, Stochastic K/D
- Volatility: Bollinger band width/pct, ATR%, historical vol (10/21/63 day)
- Microstructure: candle body, shadows, gap, H-L range
- Calendar: day of week, month, is_month/quarter_end

### Event (16)
- Earnings (5): days to/since, EPS surprise, streak, has_data flag
- Macro FRED (6): fed rate level/change 1m+3m, CPI YoY, unemployment level+3m
- Product cycle (3): days to/since Apple events, iPhone cycle flag
- Regime (2): rate environment, inflation regime

### Interaction (5)
- `earnings_proximity_surprise` = (90 - days_to_next) × |eps_surprise|
- `macro_stress_score` = zscore(fed_change_3m) + zscore(cpi_yoy) + zscore(unrate_change)
- `vol_macro_interaction` = hvol_21d × macro_stress_score
- `earnings_momentum` = last_eps_surprise × earnings_streak
- `rate_vol_regime` = fed_rate_change_3m × hvol_63d

## Model performance history

| Phase | Model | OOS Acc | Macro F1 | Bear | Side | Bull |
|-------|-------|---------|----------|------|------|------|
| 2 baseline | xgb dir_1m | 45.89% | — | 28.5% | 2.5% | 70.2% |
| 2 best | xgb dir_1w weighted | 38.35% | 0.367 | 23.1% | 45.5% | 41.8% |
| 3 events | xgb 52 feat | 36.28% | 0.354 | 28.3% | 35.9% | 42.3% |
| **3 champion** | **xgb 57 feat** | **38.30%** | **0.375** | **30.6%** | **39.9%** | **42.0%** |
| 4 target | beat F1=0.375 | — | — | — | — | — |

## Key design decisions
See `docs/decisions/` for full rationale on:
- `001` — Walk-forward validation only
- `002` — ±2% direction threshold for Sideways label
- `003` — Feature selection approach (drop price-level proxies + high-corr pairs)

## No-lookahead guarantee
Every feature is computed using only data available on or before the target date:
- Technical indicators: rolling windows on past prices only
- FRED macro: forward-filled monthly data (each day sees only the last published value)
- Earnings: forward-filled from last known event date
- Labels: computed as forward returns, excluded from feature set
