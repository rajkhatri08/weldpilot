"""More tests of `weldpilot audit ...`, on a release whose session ids are invented.

tests/data/test_audit_commands.py stays closed until onset pass 2, because its fixtures copy
real sessions that are marked blind. Tests added meanwhile live here; merge them into it
after pass 2.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from tests.data.invented import SESSION_A, Invented, point_into_test, write_marks
from weldpilot.cli import app
from weldpilot.data import audit_commands, phash
from weldpilot.data.duplicate_audit import THUMBNAIL_DIR
from weldpilot.data.index import FRAMES_FILE
from weldpilot.data.label_audit import Timeline

runner = CliRunner()

FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"


def _logging_git_sha(events: list[str]) -> Any:
    def fake_git_sha(start: Path | None = None) -> str:
        events.append("sha")
        return FAKE_SHA

    return fake_git_sha


# --- duplicates: the image guard --------------------------------------------------------


def test_duplicates_refuse_to_draw_a_training_frame_whose_path_is_in_test(
    invented: Invented,
) -> None:
    write_marks(invented, [SESSION_A])
    # Session A's frames now hold test/ files, so they pair with the test session at 0 bits
    # and get a thumbnail, while their split still says train.
    point_into_test(invented, SESSION_A)
    result = runner.invoke(app, ["audit", "duplicates", "--workers", "1"])
    assert result.exit_code == 1, result.output
    assert f"session {SESSION_A} frame" in result.output
    assert "is not a training frame" in result.output
    assert not (invented.root / "reports" / "audit" / THUMBNAIL_DIR).exists()


# --- the git SHA is read before any long step -------------------------------------------


def test_labels_read_the_git_sha_before_building_the_timelines(
    invented: Invented, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_marks(invented, [SESSION_A])
    events: list[str] = []
    real_timelines = audit_commands.timelines

    def building(frames: pd.DataFrame, session_ids: Sequence[str]) -> list[Timeline]:
        events.append("timelines")
        return real_timelines(frames, session_ids)

    monkeypatch.setattr(audit_commands, "git_sha", _logging_git_sha(events))
    monkeypatch.setattr(audit_commands, "timelines", building)
    result = runner.invoke(app, ["audit", "labels", "--unblind"])  # synthetic data only
    assert result.exit_code == 0, result.output
    assert events == ["sha", "timelines"]


def test_duplicates_read_the_git_sha_before_hashing(
    invented: Invented, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_marks(invented, [SESSION_A])
    events: list[str] = []
    real_refresh = phash.refresh_hashes

    def hashing(*args: Any, **kwargs: Any) -> phash.HashTable:
        events.append("hash")
        return real_refresh(*args, **kwargs)

    monkeypatch.setattr(audit_commands, "git_sha", _logging_git_sha(events))
    monkeypatch.setattr(phash, "refresh_hashes", hashing)
    result = runner.invoke(app, ["audit", "duplicates", "--workers", "1"])
    assert result.exit_code == 0, result.output
    assert events == ["sha", "hash"]


# --- what each output records -----------------------------------------------------------


def _fixed_git_sha(start: Path | None = None) -> str:
    return FAKE_SHA


def _record(invented: Invented, name: str) -> Any:
    path = invented.root / "reports" / "provenance" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_labels_record_the_git_sha_the_dataset_and_no_config(
    invented: Invented, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_marks(invented, [SESSION_A])
    monkeypatch.setattr(audit_commands, "git_sha", _fixed_git_sha)
    result = runner.invoke(app, ["audit", "labels", "--unblind"])  # synthetic data only
    assert result.exit_code == 0, result.output
    assert _record(invented, "audit_labels") == {
        "command": "weldpilot audit labels",
        "outputs": ["audit/label_timelines.png", "audit/labels_summary.csv", "audit/labels.md"],
        "git_sha": FAKE_SHA,
        "dataset_fingerprint": invented.fingerprint,
        "config": "no config",
    }


def test_duplicates_record_the_git_sha_the_dataset_and_no_config(
    invented: Invented, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_marks(invented, [SESSION_A])
    monkeypatch.setattr(audit_commands, "git_sha", _fixed_git_sha)
    result = runner.invoke(app, ["audit", "duplicates", "--workers", "1"])
    assert result.exit_code == 0, result.output
    assert _record(invented, "audit_duplicates") == {
        "command": "weldpilot audit duplicates",
        # The thumbnails' folder, not their names: those name sessions.
        "outputs": ["audit/duplicates.csv", "audit/duplicates.md", f"audit/{THUMBNAIL_DIR}/"],
        "git_sha": FAKE_SHA,
        "dataset_fingerprint": invented.fingerprint,
        "config": "no config",
    }


def test_duplicates_md_and_the_hash_cache_name_the_code_and_the_index(
    invented: Invented, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_marks(invented, [SESSION_A])
    monkeypatch.setattr(audit_commands, "git_sha", _fixed_git_sha)
    result = runner.invoke(app, ["audit", "duplicates", "--workers", "1"])
    assert result.exit_code == 0, result.output
    report = (invented.root / "reports" / "audit" / "duplicates.md").read_text(encoding="utf-8")
    index_sha = pd.read_parquet(invented.derived / FRAMES_FILE).attrs["git_sha"]
    assert f"- Code: git `{FAKE_SHA}`." in report
    assert (
        f"- Index: built at git `{index_sha}` from the dataset with fingerprint "
        f"`{invented.fingerprint}`."
    ) in report
    cache = pd.read_parquet(invented.derived / phash.PHASH_FILE)
    assert cache.attrs["git_sha"] == FAKE_SHA
    assert cache.attrs["dataset_fingerprint"] == invented.fingerprint


def test_a_second_duplicates_run_writes_the_same_record(
    invented: Invented, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_marks(invented, [SESSION_A])
    monkeypatch.setattr(audit_commands, "git_sha", _fixed_git_sha)
    path = invented.root / "reports" / "provenance" / "audit_duplicates.json"
    records = []
    for _ in range(2):  # the second run takes every hash from the cache
        assert runner.invoke(app, ["audit", "duplicates", "--workers", "1"]).exit_code == 0
        records.append(path.read_bytes())
    assert records[0] == records[1]


@pytest.mark.parametrize(
    "args", [["audit", "labels", "--unblind"], ["audit", "duplicates", "--workers", "1"]]
)
def test_audits_go_to_the_reports_folder_that_weldpilot_reports_names(
    invented: Invented, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    write_marks(invented, [SESSION_A])
    elsewhere = invented.root / "elsewhere"
    monkeypatch.setenv("WELDPILOT_REPORTS", str(elsewhere))
    result = runner.invoke(app, args)  # labels on synthetic data only
    assert result.exit_code == 0, result.output
    assert (elsewhere / "audit").is_dir()
    assert len(list((elsewhere / "provenance").glob("*.json"))) == 1
    assert not (invented.root / "reports").exists()
