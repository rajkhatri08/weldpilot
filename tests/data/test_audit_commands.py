"""CLI tests for `weldpilot audit labels`, on the synthetic release indexed by the real writer.
Never the real data."""

import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner, Result

from tests.data.synth import DEFAULT_FOLDERS, SynthFolder, write_dataset
from weldpilot.cli import app
from weldpilot.data.al5083 import CLASS_NAMES
from weldpilot.data.index import FRAMES_FILE, build_index, write_index
from weldpilot.provenance import git_sha

runner = CliRunner()

TRAIN_A = "170904-141730"  # from DEFAULT_FOLDERS: good to fault at 3, and at 10 after a gap
TRAIN_B = "170907-100000"  # added below: good 0-6, frames 7-18 missing, class 3 from 19
CROSSING = "170904-150144"  # from DEFAULT_FOLDERS: never a training session
TEST_ONLY = "170905-112213"  # from DEFAULT_FOLDERS

AUDIT_FOLDERS = (
    *DEFAULT_FOLDERS,
    SynthFolder("train", f"{TRAIN_B}-Al 2mm-part1", 0, tuple(range(7))),
    SynthFolder("train", f"{TRAIN_B}-Al 2mm-part2", 3, tuple(range(19, 40))),
)

INDEX_PROVENANCE = {"git_sha": "1dx5ha", "dataset_fingerprint": "f1n9e4"}

MARKS = (
    "session_id,pass,onset_frame,marked_on,note\n"
    f"{TRAIN_A},,5,2026-11-03,by Claude\n"  # a blank pass is pass 1; operator onset 3: +2
    f"{TRAIN_B},1,22,2026-11-03,by Claude\n"  # operator onset 19: +3
    f"{TRAIN_A},2,2,2026-11-11,\n"  # -1
    f"{TRAIN_B},2,,,\n"  # pass 2, not marked yet
)

OUT = Path("reports") / "audit"
FILES = ["label_timelines.png", "labels.md", "labels_summary.csv"]


def _write_marks(data_root: Path, text: str) -> Path:
    path = data_root / "annotations" / "onset_marks.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The indexed synthetic release and its marks, with WELDPILOT_DATA and the working
    directory in tmp_path, so nothing can touch the real data/ or reports/."""
    monkeypatch.setenv("WELDPILOT_DATA", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    release = write_dataset(tmp_path / "raw" / "tig-al5083", AUDIT_FOLDERS)
    write_index(build_index(release), tmp_path / "derived", INDEX_PROVENANCE)
    _write_marks(tmp_path, MARKS)
    return tmp_path


def _audit(*args: str) -> Result:
    return runner.invoke(app, ["audit", "labels", *args])


def _written(data_root: Path) -> list[str]:
    folder = data_root / OUT
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def _report(data_root: Path) -> str:
    return (data_root / OUT / "labels.md").read_text(encoding="utf-8")


def test_without_unblind_it_refuses_before_reading_anything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No index and no marks: had it read anything, it would fail on those instead.
    monkeypatch.setenv("WELDPILOT_DATA", str(tmp_path / "nowhere"))
    monkeypatch.chdir(tmp_path)
    result = _audit()
    assert result.exit_code == 1
    assert "after onset pass 2 is done, with --unblind" in result.output
    assert _written(tmp_path) == []


def test_it_writes_the_three_files_for_training_sessions_only(data_root: Path) -> None:
    result = _audit("--unblind")
    assert result.exit_code == 0, result.output
    assert _written(data_root) == FILES
    assert (data_root / OUT / "labels_summary.csv").read_text(encoding="utf-8") == (
        "session_id,n_frames,labels_in_time_order,n_label_changes,long_gaps\n"
        "170815-133921,3,3,0,\n"
        f"{TRAIN_A},10,0>2>0>4>0,4,\n"
        f"{TRAIN_B},28,0>3,1,7..18\n"
        "170913-152931,4,2>0,1,\n"
    )
    report = _report(data_root)
    assert CROSSING not in report
    assert TEST_ONLY not in report


def test_the_terminal_shows_counts_and_paths_only(data_root: Path) -> None:
    result = _audit("--unblind")
    assert result.exit_code == 0, result.output
    assert "audited  4 training sessions (45 frames)" in result.output
    assert "marks    3 onset marks (passes: 1, 2)" in result.output
    assert [name for name in CLASS_NAMES.values() if name in result.output] == []


def test_the_report_counts_label_changes_and_gaps_before_onsets(data_root: Path) -> None:
    assert _audit("--unblind").exit_code == 0
    report = _report(data_root)
    assert "change label at least once: 3 of the 4 training sessions (45 frames)" in report
    # TRAIN_A goes good to fault at 3 and at 10 (frame 9 missing); TRAIN_B at 19 (7-18).
    assert "(good weld to any fault): 3, in 2 of the 4 sessions (38 frames)" in report
    assert "and the first fault frame: 2 of the 3." in report
    assert "with a gap longer than 10 frames: 1." in report
    assert f"| {TRAIN_A} | 8 | 10 | 4 misalignment | 1 | 0.02 |" in report
    assert f"| {TRAIN_B} | 6 | 19 | 3 lack of fusion | 12 | 0.22 |" in report


def test_the_report_compares_each_pass_with_the_operator_onset(data_root: Path) -> None:
    assert _audit("--unblind").exit_code == 0
    report = _report(data_root)
    # Pass, session, mark, operator onset, delta in frames and in s, frames missing before it.
    assert f"| 1 | {TRAIN_A} | 5 | 3 | +2 | +0.04 | 0 |" in report
    assert f"| 1 | {TRAIN_B} | 22 | 19 | +3 | +0.05 | 12 |" in report
    assert f"| 2 | {TRAIN_A} | 2 | 3 | -1 | -0.02 | 0 |" in report
    # Pass 1: two sessions (10 + 28 frames), median of +2 and +3. Pass 2: one, one unmarked.
    assert "| 1 | 2 | 38 | +2.5 | +0.05 | 0 | 0 |" in report
    assert "| 2 | 1 | 10 | -1 | -0.02 | 1 | 0 |" in report
    assert "more than one good-to-fault change: 1 of the 2. Δ uses the first." in report


def test_the_outputs_record_where_their_numbers_come_from(data_root: Path) -> None:
    assert _audit("--unblind").exit_code == 0
    report = _report(data_root)
    marks = (data_root / "annotations" / "onset_marks.csv").read_bytes()
    assert f"git `{git_sha()}`" in report
    assert "built at git `1dx5ha` from the dataset with fingerprint `f1n9e4`" in report
    assert f"SHA-256 `{hashlib.sha256(marks).hexdigest()}`" in report
    with Image.open(data_root / OUT / "label_timelines.png") as image:
        assert image.info["Source"] == f"weldpilot audit labels, git {git_sha()}"


def test_a_marks_file_saved_with_a_byte_order_mark_is_read(data_root: Path) -> None:
    # Excel and Numbers can start a CSV with one; read naively, the first column would be
    # "﻿session_id" and look missing.
    path = data_root / "annotations" / "onset_marks.csv"
    path.write_bytes(b"\xef\xbb\xbf" + MARKS.encode("utf-8"))
    result = _audit("--unblind")
    assert result.exit_code == 0, result.output
    assert f"| 1 | {TRAIN_A} | 5 | 3 | +2 | +0.04 | 0 |" in _report(data_root)


@pytest.mark.parametrize(
    "bad_id", [CROSSING, TEST_ONLY, "170101-000000"], ids=["crossing", "test-only", "unknown"]
)
def test_marks_of_a_session_that_isnt_a_training_session_are_refused(
    data_root: Path, bad_id: str
) -> None:
    _write_marks(data_root, MARKS + f"{bad_id},1,8,,\n")
    result = _audit("--unblind")
    assert result.exit_code == 1
    assert f"not training sessions, so nothing was written: {bad_id!r}" in result.output
    assert _written(data_root) == []


def test_a_mark_outside_its_session_is_refused_before_anything_is_written(
    data_root: Path,
) -> None:
    _write_marks(data_root, MARKS + f"{TRAIN_B},3,99,,\n")  # TRAIN_B ends at frame 39
    result = _audit("--unblind")
    assert result.exit_code == 1
    assert f"line 6: the onset_frame of session {TRAIN_B} is outside its frames" in result.output
    assert _written(data_root) == []


@pytest.mark.parametrize(
    ("missing", "advice"),
    [
        ("annotations/onset_marks.csv", "run `weldpilot onset pick` first"),
        ("derived/frames.parquet", "run `weldpilot data index` first"),
        ("derived/sessions.parquet", "run `weldpilot data index` first"),
    ],
    ids=["marks", "frames", "sessions"],
)
def test_a_missing_input_is_a_clear_error(data_root: Path, missing: str, advice: str) -> None:
    (data_root / missing).unlink()
    result = _audit("--unblind")
    assert result.exit_code == 1
    assert advice in result.output
    assert _written(data_root) == []


def test_a_frames_table_that_disagrees_with_the_sessions_table_is_refused(
    data_root: Path,
) -> None:
    # A stale or edited frames.parquet that puts a training session's frame in test/.
    path = data_root / "derived" / FRAMES_FILE
    frames = pd.read_parquet(path)
    first = frames.index[frames["session_id"] == TRAIN_A][0]
    frames.loc[first, "split"] = "test"
    frames.to_parquet(path, index=False)
    result = _audit("--unblind")
    assert result.exit_code == 1
    assert f"frames.parquet doesn't match sessions.parquet for {TRAIN_A}" in result.output
    assert _written(data_root) == []


def test_no_image_is_opened(data_root: Path) -> None:
    # Everything comes from the index: with the release deleted, the audit still runs.
    shutil.rmtree(data_root / "raw")
    result = _audit("--unblind")
    assert result.exit_code == 0, result.output


def test_two_runs_write_the_same_files(data_root: Path) -> None:
    def run() -> tuple[bytes, bytes, tuple[int, int], bytes]:
        assert _audit("--unblind").exit_code == 0
        with Image.open(data_root / OUT / "label_timelines.png") as image:
            size, pixels = image.size, image.tobytes()
        csv = (data_root / OUT / "labels_summary.csv").read_bytes()
        md = (data_root / OUT / "labels.md").read_bytes()
        return csv, md, size, pixels

    assert run() == run()
