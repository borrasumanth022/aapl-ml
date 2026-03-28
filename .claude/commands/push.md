Commit and push the current aapl_ml changes to GitHub for $ARGUMENTS (default: current branch).

Steps:
1. Run `git status` — show what has changed.
2. Block if any of these are staged:
   - `data/` directory (any parquet)
   - `models/` directory (any .pkl or .pt file)
   - `CLAUDE.md` or `CLAUDE.local.md`
   - `.claude/settings.local.json`
   - `*.log`, `*.ipynb`, `shap_summary*.csv`
3. Stage only safe files:
   - `src/` — pipeline scripts
   - `config/` — any config files
   - `.claude/` — except settings.local.json (gitignored)
   - `docs/`, `requirements.txt`, `.gitignore`, `README.md`
4. Show `git diff --stat`.
5. Propose a commit message in the aapl_ml convention:
   - Format: `Phase N complete — <description>, <metric>` for major milestones
   - Format: `<type>(<scope>): <description>` for incremental changes
   - Include metric delta: `feat(features): add rate_vol_regime interaction — F1 0.375 (+0.008)`
6. Commit with Co-Authored-By tag.
7. Push to current branch. Never push to main directly.

If adding a new champion model, also run:
```bash
git tag model/xgb-phase3-F1-0.375
git push origin model/xgb-phase3-F1-0.375
```

If $ARGUMENTS contains a message in quotes, use that message verbatim.
