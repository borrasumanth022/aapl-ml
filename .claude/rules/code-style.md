# Code Style Rules — aapl_ml

## File paths
- Always use `pathlib.Path` — never raw Windows strings like `"C:\\Users\\..."` in src/
- Import path constants from `config/paths.py` where available

## Output
- All `print()` output must be ASCII-only — no Unicode arrows (→), box chars (│), or emoji
- Use `[INFO]`, `[WARN]`, `[ERROR]` prefixes for log lines

## Function signatures
- Type hints required on all function signatures
- Return type annotation required on all functions that return a value

## File writes
- Skip-if-exists pattern on all output files:
  ```python
  if output_path.exists():
      print(f"[INFO] {output_path.name} already exists — skipping")
  else:
      df.to_parquet(output_path)
  ```

## Imports
- Standard library → third-party → local, separated by blank lines
- No wildcard imports (`from module import *`)

## Error handling
- `sys.exit(1)` on bad/missing data — never silent pass
- Validate required columns exist after loading parquet; print missing columns before exiting
