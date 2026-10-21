# WeldPilot

Real-time TIG weld monitoring from camera frames: detect a fault early, name its likely cause, suggest an action. Advisory only; it never controls the welder. The edge target is a TI AM62A board (INT8).

Raj is building this to learn Claude Code and to show it in interviews. Explain your reasoning as you work.

## Commands

- Sync dependencies: `uv sync`
- Tests: `uv run pytest`
- Lint and autofix: `uv run ruff check . --fix`
- Format: `uv run ruff format .`
- Type check: `uv run mypy src`
- All checks: `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest`
- CLI: `uv run weldpilot --help`

## Layout

- `src/weldpilot/`: `data/`, `detect/`, `cause/`, `act/`, `edge/`, `eval/`, `report/`, `cli.py`
- `tests/` mirrors `src/weldpilot/`
- `configs/`: YAML configs
- `docs/`: `datasheet.md`, `analysis_plan.md`, decision records in `docs/adr/NNNN-title.md`
- `data/` and `reports/` are gitignored, except `data/README.md` and `data/annotations/`

## Data rules (never break these)

1. Split by trial (one welding run), never by frame.
2. The official test trials stay locked until gate G4 (Jan 24, 2027). Don't open, display, featurize or train on test images. Counting files, indexing paths and labels, and perceptual hashes for the leakage audit are the only exceptions.
3. Every analysis gets its trials from `weldpilot.data.splits`. No ad-hoc splits.
4. Thresholds and calibration come from held-out training trials only. Never use Anomalib's Folder default `val_split_mode=FROM_TEST`.
5. Every reported number carries its unit, n (trials and frames), a 95% interval and, for alarm rates, the exposure in hours.
6. Never modify, move or delete anything in `data/raw/`.
7. Fix seeds, and record the git SHA and config hash with every result.

## Files Raj writes himself

Don't create or edit these. Explain, hint when asked, and review. If a task needs a change in one of them, stop and tell Raj what and why.

- `src/weldpilot/data/splits.py`
- `tests/test_leakage.py`
- `src/weldpilot/eval/metrics.py`
- `src/weldpilot/eval/stats.py`
- `src/weldpilot/detect/cusum.py`
- `src/weldpilot/detect/arc_state.py`
- `src/weldpilot/cause/abstain.py`

## How to work with Raj

- He is new to Claude Code. Say why, not only what.
- Plan first for anything that touches more than two files, and wait for approval.
- Keep diffs small and on task. Ask before adding a dependency.
- Run the checks and show the real output before you say something works.
- When unsure, ask. Don't guess.
- End each task with: files changed, how Raj can check it by hand, and one idea worth understanding.
- Never run `git commit` or `git push`. Suggest a commit message instead.
- Don't edit `CLAUDE.md`, `.claude/` or `.github/` unless asked.

## Code style

- Python 3.12, type hints everywhere, `pathlib`, no bare `except`.
- Pure functions; file and network I/O at the edges.
- Paths come from config (`WELDPILOT_DATA`, default `./data`), never hard-coded.
- Portable code: no OS-specific paths, and project tasks as Typer CLI commands rather than shell scripts.
- Tests are fast, use tiny synthetic fixtures, and never read real data.

## Environment

- Raj works on a MacBook Air M1 (Apple Silicon, macOS). CI runs on Ubuntu.
- PyTorch code uses the `mps` device when available, else CPU. Never assume CUDA.
- TI's model compiler for the AM62A (edgeai-tidl-tools) runs only on x86 Linux, so edge compilation happens off the Mac.

## Domain facts

- Main dataset: TIG welding of aluminium 5083 (Bacioiu et al., Journal of Manufacturing Processes 45, 2019). 33,254 frames from 60 trials, split by trial into train and test (about 75/25).
- Control settings: 90 A and 35 cm/min travel speed. Every trial starts at the control settings before a fault is introduced.
- Labels come from an operator, so treat them as noisy and audit them.
- Frame rate: assume 55 fps (the camera rate reported for the SS304 set) and state the assumption wherever frames become seconds.
- Second dataset, used later: SS304, 45,058 frames, 1208×700, Xiris XVC-1000 camera.
- Dataset layout and label format: (Raj adds these after Session 2)
