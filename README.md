# aapl_ml

**An end-to-end machine learning pipeline for equity price prediction** — from raw OHLCV data through feature engineering, labelling, and multi-horizon modelling.

Built on a strict no-lookahead principle: walk-forward validation throughout, no data leakage.

---

## Overview

aapl_ml is a phased research pipeline that progressively builds toward a full market prediction system:

1. Download and store historical price data
2. Engineer 50+ technical features across trend, momentum, volatility, and microstructure
3. Construct forward return labels at multiple time horizons
4. Train and evaluate baseline models (Phase 2)
5. Incorporate real-world event signals via NLP (Phase 3+)

Currently uses AAPL as the proving ground before expanding to other instruments.

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | **Data Foundation** — OHLCV ingestion, feature engineering, forward return labels | ✅ Complete |
| 2 | **Pattern Recognition** — Baseline models (XGBoost, LSTM) on price features | 🔲 Next |
| 3 | **Event Linkage** — NLP layer mapping news and macro events to market reactions | 🔲 Planned |
| 4 | **Fusion Model** — Combined price patterns + event signals | 🔲 Planned |
| 5 | **Prediction Engine** — Multi-horizon forecasts (1w → 1m → 3m → 6m → 1y) | 🔲 Planned |

---

## Phase 1 output

| Script | Output file | Rows | Columns |
|---|---|---|---|
| `01_fetch_data.py` | `data/raw/aapl_daily_raw.parquet` | 7,856 | 5 |
| `02_features.py` | `data/processed/aapl_features.parquet` | 7,657 | 57 |
| `03_labels.py` | `data/processed/aapl_labeled.parquet` | 7,405 | 77 |

Date range: **October 1995 – March 2026**

Labels are generated at 5 horizons (1w, 1m, 3m, 6m, 1y), each with three target types:
- **Regression** — raw % forward return
- **3-class direction** — +1 (up) / 0 (sideways) / −1 (down), with a ±2% noise band
- **Binary** — up (1) or down (0)

---

## Tech stack

| Library | Purpose |
|---|---|
| `yfinance` | Historical OHLCV data download |
| `pandas` / `numpy` | Feature engineering and label construction |
| `pyarrow` | Parquet serialisation |
| `scikit-learn` | Preprocessing, baseline models (Phase 2) |
| `xgboost` | Gradient boosting models (Phase 2) |
| `torch` | LSTM / deep learning (Phase 2) |

**Python 3.10+** — no GPU required for Phase 1.

Built with [Claude Code](https://claude.ai/claude-code) — AI-assisted development from architecture to deployment.

---

## Prerequisites

```bash
pip install yfinance pandas numpy pyarrow
```

For Phase 2, additionally install:

```bash
pip install scikit-learn xgboost torch
```

---

## Running the pipeline

Each script is standalone and can be re-run independently.

```bash
# Step 1 — download raw OHLCV data from Yahoo Finance
python pipeline/01_fetch_data.py

# Step 2 — engineer technical features
python pipeline/02_features.py

# Step 3 — build forward return labels
python pipeline/03_labels.py
```

Run in order on first setup. Outputs are written to `data/raw/` and `data/processed/`.

---

## Project structure

```
aapl_ml/
├── data/
│   ├── raw/                  # Raw OHLCV (downloaded)
│   └── processed/            # Feature matrix and labelled dataset
├── pipeline/
│   ├── 01_fetch_data.py      # Data ingestion
│   ├── 02_features.py        # Feature engineering
│   └── 03_labels.py          # Label construction
├── notebooks/
│   └── 01_explore.ipynb      # Exploratory data analysis
├── models/                   # Saved model artefacts (Phase 2+)
└── logs/                     # Training logs (Phase 2+)
```

---

## Feature groups

| Group | Examples |
|---|---|
| Price transforms | Log return, 1d/2d/5d return, 52-week range position |
| Trend | SMA 10/20/50/100/200, EMA 12/26, MACD, golden/death cross |
| Momentum | RSI 7/14, Stochastic %K/%D, Williams %R, Rate of Change |
| Volatility | Bollinger Bands, ATR, historical vol (10d/21d/63d) |
| Microstructure | Candle body ratio, upper/lower shadows, gap %, HL range |
| Calendar | Day of week, month, quarter, month-end/start flags |

---

## Label reference

| Column | Type | Description |
|---|---|---|
| `ret_1w` | float | % return over next 5 trading days |
| `ret_1m` | float | % return over next 21 trading days |
| `ret_3m` | float | % return over next 63 trading days |
| `ret_6m` | float | % return over next 126 trading days |
| `ret_1y` | float | % return over next 252 trading days |
| `dir_1w` | +1 / 0 / −1 | 3-class direction with ±2% sideways band |
| `bin_1w` | 1 / 0 | Binary up/down |
| `adj_ret_1w` | float | Return ÷ 21d realised volatility |

Each horizon has all four column types (`ret_`, `dir_`, `bin_`, `adj_ret_`).

---

## Design decisions

- **Parquet over CSV** — faster I/O and smaller files at scale
- **Walk-forward validation only** — no random train/test splits that leak future data
- **±2% sideways band** — avoids training models on returns that are indistinguishable from noise
- **Single instrument first** — validate the full pipeline end-to-end on AAPL before broadening scope

---

## Related

**[ManthIQ](https://github.com/borrasumanth022/ManthIQ)** — a React + FastAPI dashboard that reads the `aapl_features.parquet` output from this pipeline and visualises it with an interactive price chart, metric cards, and a Model Lab for prediction output.

---

## License

[MIT](LICENSE) © 2026 Sumanth Borra
