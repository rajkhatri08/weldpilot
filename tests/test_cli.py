import platform
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from weldpilot import __version__
from weldpilot.cli import app, git_sha

runner = CliRunner()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A throwaway repo with one commit. Identity and signing are set locally so the
    commit works on any machine, including a CI runner with no git identity."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "WeldPilot Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "weld.txt").write_text("bead\n")
    _git(repo, "add", "weld.txt")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "--no-verify", "-m", "first")
    return repo


def test_info_prints_versions_and_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WELDPILOT_DATA", str(tmp_path))
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output
    assert platform.python_version() in result.output
    assert str(tmp_path) in result.output


def test_git_sha_clean_repo_returns_head(git_repo: Path) -> None:
    assert git_sha(git_repo) == _git(git_repo, "rev-parse", "HEAD")


def test_git_sha_uncommitted_change_adds_dirty(git_repo: Path) -> None:
    head = _git(git_repo, "rev-parse", "HEAD")
    (git_repo / "weld.txt").write_text("porosity\n")
    assert git_sha(git_repo) == f"{head}-dirty"


def test_git_sha_outside_repo_is_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    # Stop git from searching above tmp_path, in case the temp folder sits inside a repo.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert git_sha(plain) == "unknown"
