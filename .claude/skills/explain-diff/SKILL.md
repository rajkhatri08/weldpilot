---
name: explain-diff
description: Explain the uncommitted changes in plain words, check them against CLAUDE.md, and suggest a commit message. Read-only.
disable-model-invocation: true
---

Run `git status`, `git diff` and `git diff --staged`. Read any new untracked files. Then explain:

1. **What changed**: one line per file.
2. **Why**: the reason for each change, in plain words.
3. **Rule check**: anything that breaks CLAUDE.md (data rules, files Raj writes himself, hard-coded paths, missing tests). Say "none found" if clean.
4. **Risks**: what could break, and which test would catch it. Name the missing test if there isn't one.
5. **Commit message**: Conventional Commits style (`feat:`, `fix:`, `test:`, `docs:`, `chore:`), a subject under 72 characters, and a short body.

Don't edit files, and don't run `git add` or `git commit`.
