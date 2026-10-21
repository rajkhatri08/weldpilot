# 0001: Python 3.12, uv, ruff, mypy and Typer

- Status: accepted
- Date: 2026-10-21

## Context

WeldPilot needs one reproducible Python environment that behaves the same on a MacBook Air M1 and on the Ubuntu CI runner. Evaluation code computes the numbers we report, so bugs there are costly and hard to spot. Project tasks must run on any OS, without shell scripts.

## Decision

### Python 3.12

- The newest NumPy (2.5) and SciPy (1.18) require Python 3.12 or later (PyPI `requires_python`, checked 2026-10-21). On 3.11 we would be stuck on older releases.
- Python 3.11 stops getting security fixes in October 2027; 3.12 is supported until October 2028 (python.org release-cycle data).
- TI's model compiler for the AM62A (edgeai-tidl-tools) runs only on x86 Linux, in its own environment, so it doesn't constrain the Python version of this one.

### uv

One fast tool installs Python, creates the virtual environment and writes a cross-platform lockfile (`uv.lock`). CI runs `uv sync --locked`, so it fails if the lockfile is out of date and otherwise installs exactly what we tested locally.

### ruff

One tool for linting and formatting, replacing flake8, isort, pyupgrade and black. It is fast enough to run on every commit through pre-commit.

### mypy (strict)

Strict mode requires types on every function and rejects implicit `Any`. That catches `None` handling mistakes and wrong argument types in evaluation code before they change a reported number. It checks `src/` only; tests stay quick to write.

### Typer

The CLI is built from type-hinted functions, so the types document the commands. Project tasks become `weldpilot <command>` and run on macOS, Linux and Windows alike, instead of shell scripts that only work on one OS.

## Consequences

- Contributors need only uv; it installs Python 3.12 itself.
- pandas, SciPy, PyYAML and pyarrow ship without type information. When `src/` first imports one, we decide between stub packages and mypy overrides (deferred until then).
- The ruff version in `uv.lock` and the `rev` in `.pre-commit-config.yaml` must be bumped together.

## Alternatives considered

- **Python 3.11:** would cap NumPy and SciPy at older releases, and support ends a year sooner.
- **pip + venv + pip-tools, or Poetry:** slower, and Python versions are installed separately.
- **flake8 + black + isort:** three tools and three configs for what ruff does alone.
- **argparse / Click:** argparse needs more boilerplate. Typer is built on Click and adds type hints.
