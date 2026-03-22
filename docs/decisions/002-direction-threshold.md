# ADR 002 — ±2% direction threshold for Sideways label

**Status**: Accepted
**Date**: 2025 (Phase 1)

## Context

We need to label each trading day as Bull (+1), Sideways (0), or Bear (-1) based on the forward return over a given horizon. The threshold that separates "directional" moves from "noise" is a design choice.

## Decision

Use **±2%** as the Sideways band: returns within [-2%, +2%] are labelled 0 (Sideways).

```python
DIRECTION_THRESHOLD = 0.02   # in config/settings.py
```

Applied in `src/03_labels.py` to all 5 horizons.

## Consequences

**Class distribution for dir_1w (used by champion model):**
- Bull:     ~37–40% of days
- Bear:     ~28–32% of days
- Sideways: ~28–35% of days

This gives a more balanced 3-class problem than a simple binary up/down split.

**Why ±2%?**
- At the 1-week horizon, daily AAPL volatility (hvol_21d) averages ~25% annualised → ~1.6% per week
- ±2% captures roughly ±1.25σ of weekly moves — beyond this the move is "meaningful"
- Smaller threshold (e.g. ±1%) creates too many Sideways labels, reducing signal
- Larger threshold (e.g. ±3%) makes Sideways too rare to learn from

**Why a 3-class label vs binary up/down?**
A binary classifier ignores that a flat week has a different trading implication than a strong directional week. The ±2% band forces the model to predict with some conviction before calling a direction.

**Known issue:**
At longer horizons (dir_1m, dir_3m), the ±2% band becomes less meaningful — monthly returns exceeding ±2% are the norm rather than the exception, causing the Sideways class to collapse. This is why dir_1w performs better than dir_1m for this model.

## Future consideration
Consider a horizon-adaptive threshold (e.g., ±volatility × √horizon) rather than a fixed ±2%.
