"""More tests of `weldpilot onset ...`, on a release whose session ids are invented.

tests/data/test_onset_commands.py stays closed until onset pass 2, because its fixtures copy
real sessions that are marked blind. Tests added meanwhile live here; merge them into it
after pass 2.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from tests.data.invented import SESSION_A, SESSION_B, Invented, point_into_test, write_marks
from weldpilot.cli import app
from weldpilot.data import onset_commands
from weldpilot.data.index import FRAMES_FILE
from weldpilot.data.onset import MARKS_FILE_NAME, marks_csv

runner = CliRunner()

FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"


def _marks(invented: Invented) -> Path:
    return invented.root / "annotations" / MARKS_FILE_NAME


# --- pick: the git SHA it prints --------------------------------------------------------


def test_pick_reads_the_git_sha_before_it_writes_the_marks_file(
    invented: Invented, monkeypatch: pytest.MonkeyPatch
) -> None:
    # git tracks the marks file, so a SHA read after writing it would always end in -dirty.
    marks = _marks(invented)
    existed: list[bool] = []

    def fake_git_sha(start: Path | None = None) -> str:
        existed.append(marks.exists())
        return FAKE_SHA

    monkeypatch.setattr(onset_commands, "git_sha", fake_git_sha)
    result = runner.invoke(app, ["onset", "pick"])
    assert result.exit_code == 0, result.output
    assert existed == [False]
    assert f"git {FAKE_SHA})" in result.output


def test_pick_with_force_reads_the_git_sha_before_it_overwrites_the_marks_file(
    invented: Invented, monkeypatch: pytest.MonkeyPatch
) -> None:
    marks = _marks(invented)
    marks.parent.mkdir(parents=True, exist_ok=True)
    before = marks_csv([])  # a marks file with the header only
    marks.write_text(before, encoding="utf-8")
    seen: list[str] = []

    def fake_git_sha(start: Path | None = None) -> str:
        seen.append(marks.read_text(encoding="utf-8"))
        return FAKE_SHA

    monkeypatch.setattr(onset_commands, "git_sha", fake_git_sha)
    result = runner.invoke(app, ["onset", "pick", "--force"])
    assert result.exit_code == 0, result.output
    assert seen == [before]
    assert marks.read_text(encoding="utf-8") != before  # and it did overwrite the file


# --- sheets: the image guard ------------------------------------------------------------


def test_sheets_refuse_a_training_frame_whose_path_is_in_test(
    invented: Invented, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runner.invoke(app, ["onset", "pick"]).exit_code == 0
    point_into_test(invented, SESSION_A)  # split still says train; the paths say test/
    drawn: list[str] = []

    def spy(session_id: str, paths: Mapping[int, Path]) -> Image.Image:
        drawn.append(session_id)
        return Image.new("L", (1, 1))

    monkeypatch.setattr(onset_commands, "make_sheet", spy)
    result = runner.invoke(app, ["onset", "sheets"])
    assert result.exit_code == 1, result.output
    assert f"session {SESSION_A}" in result.output
    assert "outside train/" in result.output
    assert drawn == []  # every session is checked before any sheet is drawn
    assert not (invented.root / "reports" / "onset_sheets").exists()


# --- sheets: provenance -----------------------------------------------------------------


def test_sheets_read_the_git_sha_before_drawing(
    invented: Invented, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runner.invoke(app, ["onset", "pick"]).exit_code == 0
    events: list[str] = []
    real_make_sheet = onset_commands.make_sheet

    def fake_git_sha(start: Path | None = None) -> str:
        events.append("sha")
        return FAKE_SHA

    def drawing(session_id: str, paths: Mapping[int, Path]) -> Image.Image:
        events.append("draw")
        return real_make_sheet(session_id, paths)

    monkeypatch.setattr(onset_commands, "git_sha", fake_git_sha)
    monkeypatch.setattr(onset_commands, "make_sheet", drawing)
    result = runner.invoke(app, ["onset", "sheets"])
    assert result.exit_code == 0, result.output
    assert events == ["sha", "draw", "draw"]


def _fixed_git_sha(start: Path | None = None) -> str:
    return FAKE_SHA


def _sheets_record(invented: Invented) -> Path:
    return invented.root / "reports" / "provenance" / "onset_sheets.json"


def test_sheets_record_the_git_sha_the_dataset_and_no_config(
    invented: Invented, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runner.invoke(app, ["onset", "pick"]).exit_code == 0
    monkeypatch.setattr(onset_commands, "git_sha", _fixed_git_sha)
    result = runner.invoke(app, ["onset", "sheets"])
    assert result.exit_code == 0, result.output
    assert json.loads(_sheets_record(invented).read_text(encoding="utf-8")) == {
        "command": "weldpilot onset sheets",
        "outputs": ["onset_sheets/"],  # the folder: sheet file names are session ids
        "git_sha": FAKE_SHA,
        "dataset_fingerprint": invented.fingerprint,
        "config": "no config",
    }
    assert f"wrote {Path('reports', 'provenance', 'onset_sheets.json')}" in result.output


def test_two_sheet_runs_write_the_same_record(
    invented: Invented, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runner.invoke(app, ["onset", "pick"]).exit_code == 0
    monkeypatch.setattr(onset_commands, "git_sha", _fixed_git_sha)
    records = []
    for _ in range(2):
        assert runner.invoke(app, ["onset", "sheets"]).exit_code == 0
        records.append(_sheets_record(invented).read_bytes())
    assert records[0] == records[1]


def test_sheets_go_to_the_reports_folder_that_weldpilot_reports_names(
    invented: Invented, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runner.invoke(app, ["onset", "pick"]).exit_code == 0
    elsewhere = invented.root / "elsewhere"
    monkeypatch.setenv("WELDPILOT_REPORTS", str(elsewhere))
    result = runner.invoke(app, ["onset", "sheets"])
    assert result.exit_code == 0, result.output
    drawn = sorted(p.name for p in (elsewhere / "onset_sheets").glob("*.png"))
    assert drawn == [f"{SESSION_A}.png", f"{SESSION_B}.png"]
    assert (elsewhere / "provenance" / "onset_sheets.json").is_file()
    assert not (invented.root / "reports").exists()


# --- sheets: inputs that are empty or missing -------------------------------------------


def test_sheets_with_a_marks_file_that_has_only_its_header_draw_nothing(
    invented: Invented,
) -> None:
    write_marks(invented, [])
    result = runner.invoke(app, ["onset", "sheets"])
    assert result.exit_code == 0, result.output
    assert "no sessions listed in" in result.output
    assert not (invented.root / "reports").exists()  # no sheet, and no provenance file


def test_sheets_without_frames_parquet_name_the_file_and_the_fix(invented: Invented) -> None:
    assert runner.invoke(app, ["onset", "pick"]).exit_code == 0
    (invented.derived / FRAMES_FILE).unlink()
    result = runner.invoke(app, ["onset", "sheets"])
    assert result.exit_code == 1
    assert f"no {FRAMES_FILE} in" in result.output
    assert "run `weldpilot data index` first" in result.output
    assert not (invented.root / "reports").exists()
