# Skill: model-training

Reusable workflow for training a new model and comparing it against the current champion.

## When to use
- Training a new model type (LightGBM, CatBoost, LSTM, ensemble)
- Tuning hyperparameters on an existing model
- Experimenting with a new target horizon or feature set

## Current champion
- **Model**: `xgb_best_interactions.pkl` (Phase 3 champion)
- **Target**: `dir_1w` (weekly direction)
- **Features**: 57 (36 tech + 16 event + 5 interaction)
- **OOS Accuracy**: 38.30% | **Macro F1**: 0.375
- See `config/settings.py:PHASE3_CHAMPION` for full metrics

## Workflow

### 1. Name the script
Follow the numbering convention: `src/12_<name>.py`, `src/13_<name>.py`, etc.

### 2. Load data and features
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import paths as P, settings as S
import pandas as pd, json

df = pd.read_parquet(P.DATA_WITH_EVENTS)
with open(P.FEATURE_LIST) as f:
    tech_features = json.load(f)["features"]
all_features = tech_features + S.EVENT_FEATURES + S.INTERACTION_FEATURES
```

### 3. Walk-forward validation (mandatory)
Use `sklearn.model_selection.TimeSeriesSplit(n_splits=S.N_SPLITS)`.
- **Never use random splits** — this is a time series
- Train on strictly past data only
- Collect OOS predictions across all folds before computing metrics

### 4. Evaluate
Always report in this order so results are comparable across phases:
1. OOS Accuracy vs naive baseline (always-majority-class)
2. Macro F1 (primary metric — balanced across 3 classes)
3. Per-class recall: Bear, Sideways, Bull
4. Confusion matrix

Compare against `S.PHASE3_CHAMPION` metrics.

### 5. Save artefacts
- Model: `P.MODEL_<NAME>` (add path to `config/paths.py` first)
- Predictions: `P.PRED_<NAME>` (same)
- SHAP: `P.SHAP_<NAME>` (same)

### 6. Update champion if improved
If Macro F1 > 0.375:
- Copy model to `models/xgb_phase4_champion.pkl` (or appropriate name)
- Update `config/settings.py:PHASE4_CHAMPION`
- Update `CLAUDE.md` with results

## Evaluation standard
A model beats the champion only if **Macro F1 improves** — not just accuracy.
Accuracy can be misleading on imbalanced 3-class problems.
Bear recall ≥ 25% and Sideways recall ≥ 30% are secondary requirements.
