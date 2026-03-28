# /project:push -- Stage, commit, and push code to GitHub

**Usage:**
- /project:push "Phase 5 Step 3: Transformer on raw OHLCV" -- with message
- /project:push -- prompts for message

## Instructions

1. Run git status to see all changes.

2. Stage ONLY these paths:
     git add src/
     git add config/
     git add .claude/commands/
     git add .claude/rules/
     git add .claude/agents/
     git add .claude/skills/
     git add .claude/hooks/
     git add .claude/settings.json
     git add requirements.txt
     git add .gitignore
     git add README.md
     git add CLAUDE.md
     git add docs/

3. Scan staged diff for:
   - Any path under data/ or models/ -> STOP and unstage
   - Any personal machine path in staged file -> STOP and warn
   - CLAUDE.local.md or settings.local.json in staged set -> STOP and unstage

4. If on main branch, stop. Push to: feat/, fix/, or phase{N}/ branch.

5. Commit format:
     {message}

     Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

6. Push and print the branch URL.

