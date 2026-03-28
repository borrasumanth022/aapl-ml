# /project:review -- Code review for ML correctness and lookahead bias

**Usage:**
- /project:review -- review files changed in last commit
- /project:review src/11_interaction_features.py -- specific file
- /project:review src/ -- all .py files in src/

## Instructions

Adopt the persona from .claude/agents/code-reviewer.md.

1. Determine files to review:
   - If path given, read that file/directory
   - If no path, run: git diff HEAD~1 --name-only

2. For each Python file, run the full checklist:

   Lookahead bias:
   - [ ] No shift(-N) on price/return columns as features
   - [ ] No rolling(..., center=True) on feature columns
   - [ ] No target column or derivative used as a feature

   Walk-forward validation:
   - [ ] Uses TimeSeriesSplit, not KFold(shuffle=True)
   - [ ] Train rows always earlier in time than test rows

   Holdout:
   - [ ] Holdout cutoff is 2024-01-01
   - [ ] Holdout not touched until final model comparison

   AAPL-specific:
   - [ ] NaN sentinels: 90 for pre-2005 earnings, 180 for pre-2000 product events
   - [ ] SHAP saved to CSV after each new champion
   - [ ] Per-class recall reported (not just accuracy)

3. For each FAIL: file, line number, code, why wrong, correct fix.

4. Final verdict: CLEAN or ISSUES FOUND (N issues).

