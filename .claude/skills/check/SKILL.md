---
name: check
description: Run every project check (lint, format, types, tests) and report the real results without fixing anything.
disable-model-invocation: true
---

Run these from the project root, one at a time. Keep going even if one fails.

1. `uv run ruff check .`
2. `uv run ruff format --check .`
3. `uv run mypy src`
4. `uv run pytest -q`

Then report:

- A table with one row per check: name, pass or fail, and the key number (errors found, or tests passed and failed).
- For each failure, the exact error lines copied from the output.
- One sentence on the likely cause of each failure.

Don't fix anything and don't edit files. Raj decides what happens next.
