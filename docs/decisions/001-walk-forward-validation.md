# ADR 001 — Walk-forward (expanding window) validation only

**Status**: Accepted
**Date**: 2025 (Phase 2)

## Context

We need a train/test split strategy for time series data. Common options:
1. Random split (k-fold cross-validation)
2. Single train/test split at a fixed date
3. Walk-forward (expanding window) validation
4. Rolling window validation

## Decision

Use **walk-forward expanding window** validation with 5 folds via `sklearn.model_selection.TimeSeriesSplit`.

Each fold expands the training set to include all data up to the split point, and tests on the next chronological block. This mirrors how the model would be deployed: trained on all available history, evaluated on unseen future data.

## Consequences

**Positive:**
- No lookahead bias — test data is always strictly after training data
- Mirrors real deployment conditions
- Provides 5 independent estimates of OOS performance
- Larger training sets in later folds reduce variance

**Negative:**
- Earlier folds have small training sets (fold 1 has ~20% of data as training)
- Cannot parallelise folds (order matters)
- More expensive than a single split

**Why not random k-fold?**
A random split would allow the model to train on 2025 data and be tested on 2005 data, which is impossible in deployment. It systematically overstates performance.

**Why not a single 80/20 split?**
A single split gives only one estimate of OOS performance and doesn't capture how the model performs across different market regimes (dot-com, GFC, COVID, rate hike cycle).

## Implementation
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    ...
```
All OOS predictions are collected across folds and evaluated together.
