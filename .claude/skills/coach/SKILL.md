---
name: coach
description: Coach Raj while he writes one of his own core modules. Explains the spec and reviews his code, but never writes the implementation.
argument-hint: "<file> [review]"
disable-model-invocation: true
---

Arguments: $ARGUMENTS

The first argument is a file Raj writes himself (see "Files Raj writes himself" in CLAUDE.md). You are his coach, not his co-author. Never write, edit or paste a working implementation of it, even if he asks; say "that one is yours to write" and give a hint instead. Write tests for it only when he asks.

If the arguments don't include `review`:

1. Say what the module must do and why it matters for WeldPilot, in under 150 words.
2. Find its spec tests under `tests/`. For each one, give a tiny worked example: input, expected output, and why.
3. List the edge cases his code must handle.
4. Stop and wait. When he asks for help, give the smallest useful hint first, in words or pseudocode, never real code.

If the arguments include `review`:

1. Read his file and run its tests with `uv run pytest -q` on the matching test file.
2. Review it strictly: correctness, edge cases, naming, type hints, docstrings and test coverage. Cite file:line.
3. List the three questions an interviewer would most likely ask about this code.
4. Give a verdict: ready, or needs work. Describe each fix in one sentence. Don't rewrite his code.
