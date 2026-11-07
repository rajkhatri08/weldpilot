import json
import subprocess
from pathlib import Path

import pytest

from weldpilot.provenance import PROVENANCE_DIR, git_sha, provenance_json, write_provenance


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


def test_a_provenance_record_names_the_command_its_outputs_the_code_and_the_data() -> None:
    outputs = ["audit/duplicates.csv", "audit/duplicate_thumbnails/"]
    text = provenance_json("weldpilot audit duplicates", outputs, "abc123", "f" * 64)
    assert json.loads(text) == {
        "command": "weldpilot audit duplicates",
        "outputs": outputs,
        "git_sha": "abc123",
        "dataset_fingerprint": "f" * 64,
        "config": "no config",  # where a config hash will go once commands read configs
    }


def test_a_provenance_record_has_sorted_keys_and_keeps_the_order_of_its_outputs() -> None:
    text = provenance_json("weldpilot onset sheets", ["b/", "a/"], "abc123", "f" * 64)
    record = json.loads(text)
    assert list(record) == sorted(record)
    assert record["outputs"] == ["b/", "a/"]
    assert text.endswith("}\n")


def test_write_provenance_keeps_one_file_per_name_in_the_reports_folder(tmp_path: Path) -> None:
    path = write_provenance(tmp_path / "reports", "audit_labels", "{}\n")
    assert path == tmp_path / "reports" / PROVENANCE_DIR / "audit_labels.json"
    assert path.read_text(encoding="utf-8") == "{}\n"
    assert write_provenance(tmp_path / "reports", "audit_labels", "[]\n") == path
    assert path.read_text(encoding="utf-8") == "[]\n"  # the next run overwrites it
