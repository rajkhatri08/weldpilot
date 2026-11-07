"""Provenance: which version of the code produced a result (data rule 8)."""

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent

NO_CONFIG = "no config"
"""Recorded where a config hash would go: no command reads a config file yet (data rule 8)."""

PROVENANCE_DIR = "provenance"
"""In the reports folder: one JSON file per command, rewritten by each run."""


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def git_sha(start: Path | None = None) -> str:
    """Return the commit SHA of the git repo that contains `start`.

    `start` defaults to this package's folder, so the SHA describes the code that is
    running, whatever the current directory. Adds "-dirty" when the repo has uncommitted
    changes, untracked files included. Returns "unknown" when git is missing or `start`
    is not inside a repo with at least one commit.
    """
    folder = _PACKAGE_DIR if start is None else start
    try:
        sha = _git(folder, "rev-parse", "HEAD")
        dirty = _git(folder, "status", "--porcelain") != ""
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def provenance_json(
    command: str, outputs: Sequence[str], code_sha: str, dataset_fingerprint: str
) -> str:
    """One run's provenance as JSON text: the command, the outputs it wrote (relative to the
    reports folder), the git SHA of the code, the dataset fingerprint and the config.

    Keys are sorted and nothing depends on the clock, so the same code run on the same data
    writes the same bytes.
    """
    record: dict[str, str | list[str]] = {
        "command": command,
        "outputs": list(outputs),
        "git_sha": code_sha,
        "dataset_fingerprint": dataset_fingerprint,
        "config": NO_CONFIG,
    }
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def write_provenance(reports_root: Path, name: str, text: str) -> Path:
    """Write `text` to <reports_root>/provenance/<name>.json and return that path."""
    path = reports_root / PROVENANCE_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return path
