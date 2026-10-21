# WeldPilot

Real-time TIG weld monitoring from camera frames. WeldPilot aims to detect a weld fault early, name its likely cause and suggest an action to the operator. It is advisory only and never controls the welder. The edge target is a TI AM62A board running an INT8 model.

**Status: early build.** The project skeleton is in place. Detection, cause and action code is not written yet.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 for you if needed.

```bash
uv sync                      # create .venv and install locked dependencies
uv run weldpilot info        # version, Python, git commit, data root
uv run pytest                # tests
uv run pre-commit install    # run lint and format checks on every commit
```

Datasets live in `./data` by default. Set `WELDPILOT_DATA` to use another folder. See `data/README.md` for the layout.

## How this project is built

I use [Claude Code](https://claude.com/claude-code) as a pair programmer for scaffolding, tooling, tests and reviews. I write the core files myself: the trial splits, the leakage test, the evaluation metrics and statistics, the CUSUM detector, the arc-state logic and the abstention rule. `CLAUDE.md` lists them. Design decisions are recorded in `docs/adr/`.

## Licence

Apache-2.0. See `LICENSE`.
