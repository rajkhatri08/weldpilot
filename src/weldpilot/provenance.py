"""Provenance: which version of the code produced a result (data rule 8)."""

import subprocess
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


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
