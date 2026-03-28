# Skill: git-workflow

Standard git workflow for the aapl_ml project.

## What to commit vs ignore
**Commit**: pipeline scripts (`src/`), config, docs, notebooks, README
**Never commit**: data files (`data/`), model files (`models/`), logs — these are gitignored
**Never commit**: `CLAUDE.md` — internal briefing, gitignored

## Commit message convention
```
Phase N complete — <what was done>, <key metric>

Examples:
  Phase 2 complete — XGBoost baseline, SHAP analysis, walk-forward validation
  Phase 3 complete — event features, interaction features, F1=0.375
  Phase 4 Step 1 — LightGBM baseline, F1=X.XXX
  Model Lab updated with Phase N predictions
```

## Standard workflow after completing a phase step

### 1. Verify nothing sensitive is staged
```bash
git status
git diff --staged
```
Check: no `.env`, no data files, no model `.pkl` files.

### 2. Stage only source files
```bash
git add src/ config/ docs/ .claude/
git add README.md .gitignore
```

### 3. Commit with Co-Author tag
```bash
git commit -m "$(cat <<'EOF'
<commit message>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

### 4. Push
```bash
git push origin main
```

## Checkpoint pattern (for model artefacts)
Before starting a new experiment that might overwrite the champion:
```bash
cp models/xgb_best_X.pkl models/xgb_phaseN_champion.pkl
cp data/processed/aapl_predictions_X.parquet data/processed/aapl_predictions_phaseN_champion.parquet
```
These get gitignored automatically but persist locally as a safety net.

## Two-repo structure
This project has two GitHub repos:
- `aapl-ml` — ML pipeline (this repo)
- `ManthIQ` — FastAPI + React dashboard that consumes predictions

After updating `data/processed/aapl_predictions_*.parquet`, remember to:
1. Update `ManthIQ/backend/main.py:PRED_PATH` to point to the new parquet
2. Update `ManthIQ/frontend/src/pages/ModelLab.jsx` accuracy stats
3. Commit and push ManthIQ separately
