# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Project skeleton: uv package with a src layout for Python 3.12 and the subpackages `data`, `detect`, `cause`, `act`, `edge`, `eval` and `report`.
- `weldpilot info` command: package version, Python version, git commit (with `-dirty` for uncommitted changes) and data root.
- Settings from the environment: `WELDPILOT_DATA` sets the data root (default `./data`).
- Ruff, mypy (strict) and pytest config; pre-commit hooks; GitHub Actions CI.
- README, ADR template, ADR 0001 (Python and tooling), datasheet outline and the Apache-2.0 licence.
