"""
.claude/hooks/check-lookahead.py — Lookahead bias guardrail

Reads tool input JSON from stdin (Claude Code PreToolUse hook format).
Scans Python source files being written or edited for patterns that could
introduce lookahead bias into the ML pipeline.

Exit 0 = OK (allow the tool call)
Exit 2 = WARNING (block with message to Claude)

Can also be run standalone:
    python .claude/hooks/check-lookahead.py src/some_script.py
"""

import json
import re
import sys
from pathlib import Path

# ── Patterns that indicate lookahead ─────────────────────────────────────────
LOOKAHEAD_PATTERNS = [
    # Forward shifts on series (e.g., df["close"].shift(-5))
    (r'\.shift\s*\(\s*-\s*\d+', "Forward shift detected: .shift(-N) reads future values"),
    # Column names with forward-looking keywords
    (r'["\'](?:future|fwd|forward|next_day|tomorrow|lead_)[^"\']*["\']',
     "Forward-looking column name detected"),
    # Merge/join that sorts ascending then uses last rows as test
    (r'sort_values.*ascending=False.*head\(',
     "Suspicious sort+head — verify no lookahead in test selection"),
    # Label columns used as features (catches accidental ret_/dir_/bin_ in feature list)
    (r'features\s*=\s*.*(?:ret_|dir_|bin_|adj_ret_)',
     "Label prefix (ret_/dir_/bin_) found in feature list — potential target leakage"),
]

# ── Patterns that are safe despite matching the above (allow-list) ────────────
SAFE_PATTERNS = [
    r'forward_return',          # function name, not usage
    r'#.*shift\(-',             # commented out
    r'DIRECTION_THRESHOLD',     # config constant
    r'fwd_keywords\s*=',        # the check list itself in 04_feature_selection.py
    r'"fwd"',                   # keyword in a list, not a column
]


def scan_content(content: str, filepath: str) -> list[str]:
    """Return list of warning messages for any lookahead patterns found."""
    warnings = []
    lines = content.splitlines()

    for i, line in enumerate(lines, 1):
        # Skip lines that match the safe allow-list
        if any(re.search(p, line) for p in SAFE_PATTERNS):
            continue
        for pattern, message in LOOKAHEAD_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                warnings.append(f"  Line {i:>4}: {message}\n           {line.strip()}")

    return warnings


def main():
    # ── Standalone mode: python check-lookahead.py <filepath> ────────────────
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        try:
            content = Path(filepath).read_text(encoding="utf-8")
        except Exception as e:
            print(f"Could not read {filepath}: {e}")
            sys.exit(0)

        if not filepath.endswith(".py"):
            sys.exit(0)

        warnings = scan_content(content, filepath)
        if warnings:
            print(f"LOOKAHEAD CHECK — {filepath}")
            for w in warnings:
                print(w)
            print(f"\n{len(warnings)} potential lookahead issue(s) found.")
            sys.exit(2)
        else:
            print(f"LOOKAHEAD CHECK — {filepath}: OK")
            sys.exit(0)

    # ── Hook mode: reads JSON from stdin ─────────────────────────────────────
    try:
        tool_input = json.loads(sys.stdin.read())
    except Exception:
        sys.exit(0)  # not JSON — don't block

    tool_name = tool_input.get("tool_name", "")
    tool_args = tool_input.get("tool_input", {})

    if tool_name not in ("Write", "Edit"):
        sys.exit(0)

    filepath = tool_args.get("file_path", "")
    if not filepath.endswith(".py"):
        sys.exit(0)

    # Get content being written/edited
    content = tool_args.get("content", "") or tool_args.get("new_string", "")
    if not content:
        sys.exit(0)

    warnings = scan_content(content, filepath)
    if warnings:
        msg = f"LOOKAHEAD BIAS CHECK FAILED for {filepath}:\n"
        msg += "\n".join(warnings)
        msg += "\n\nVerify these are not introducing future data into features."
        print(msg)
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
