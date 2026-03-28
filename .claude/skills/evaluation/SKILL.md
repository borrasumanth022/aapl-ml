# Skill: evaluation

Standard evaluation protocol for all aapl_ml models. Run this after every training run.

## Metrics hierarchy
1. **Macro F1** — primary. Averages F1 across all 3 classes equally. Penalises poor Sideways recall.
2. **OOS Accuracy** — secondary. Must beat naive baseline (always-majority-class).
3. **Per-class recall** — Bear ≥ 25%, Sideways ≥ 30%, Bull ≥ 40% as soft targets.
4. **Beat naive baseline** — hard requirement. Never accept a model that doesn't beat always-Bull.

## Naive baselines by target
| Target | Naive accuracy | Naive strategy |
|--------|---------------|----------------|
| dir_1w | 37.50%        | Always Bull    |
| dir_1m | 52.90%        | Always Bull    |

## Standard evaluation block
Copy this into every training script:

```python
from sklearn.metrics import accuracy_score, classification_report, f1_score
from config import settings as S

oos_acc  = accuracy_score(all_actual, all_predicted)
macro_f1 = f1_score(all_actual, all_predicted, labels=S.CLASSES, average="macro")

per_class_recall = {
    cls: accuracy_score(all_actual[all_actual == cls], all_predicted[all_actual == cls])
    if (all_actual == cls).sum() > 0 else float("nan")
    for cls in S.CLASSES
}

# Compare vs champion
champ = S.PHASE3_CHAMPION
delta_f1  = macro_f1 - champ["f1"]
print(f"Macro F1: {macro_f1:.4f}  (champion: {champ['f1']:.3f}, delta: {delta_f1:+.3f})")
print(f"Bear recall: {per_class_recall[-1]*100:.1f}%  (champion: {champ['recall']['Bear']*100:.1f}%)")
```

## SHAP analysis (run after training final model)
```python
import shap, numpy as np

explainer = shap.TreeExplainer(final_model)
shap_raw  = explainer.shap_values(X)

# Handle SHAP 0.51+ 3D array format
shap_arr = np.array(shap_raw)
if shap_arr.ndim == 3 and shap_arr.shape[0] == len(X):
    shap_values = [shap_arr[:, :, i] for i in range(shap_arr.shape[2])]
elif shap_arr.ndim == 3 and shap_arr.shape[0] == 3:
    shap_values = [shap_arr[i] for i in range(3)]
else:
    shap_values = shap_raw
```

## Walk-forward validation rule
All evaluation must be on **OOS data only** — data the model never saw during training.
- Collect predictions across all 5 folds before computing aggregate metrics
- Never compute metrics on training data and report them as OOS
- See `docs/decisions/001-walk-forward-validation.md` for rationale

## Comparison table format
Always include this table in script output and CLAUDE.md updates:
```
Model                       Acc      F1     Bear   Side   Bull
Phase 3 champion (xgb)     38.30%  0.375   30.6%  39.9%  42.0%  <- beat this
<New model>                 XX.XX%  X.XXX   XX.X%  XX.X%  XX.X%
```
