"""CLI tests for `weldpilot onset pick` and `weldpilot onset sheets`, on the synthetic release
indexed by the real writer. Never the real data."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner

from tests.data.synth import DEFAULT_FOLDERS, SynthFolder, write_dataset
from weldpilot.cli import app
from weldpilot.data import onset_commands
from weldpilot.data.al5083 import CLASS_NAMES
from weldpilot.data.index import FRAMES_FILE, build_index, write_index

runner = CliRunner()

TRAIN_A = "170904-141730"  # train-only with good-to-fault changes, from DEFAULT_FOLDERS
TRAIN_B = "170907-100000"  # train-only with a good-to-fault change, added below
CROSSING = "170904-150144"  # its good-to-fault change crosses from train/ into test/
TEST_ONLY = "170906-153326"  # test-only with a good-to-fault change, added below

# Both CROSSING and TEST_ONLY would qualify if the split rules were skipped.
ONSET_FOLDERS = (
    *DEFAULT_FOLDERS,
    SynthFolder("train", f"{TRAIN_B}-Al 2mm-part1", 0, (0, 1, 2, 3, 4, 5, 6)),
    SynthFolder("train", f"{TRAIN_B}-Al 2mm-part2", 3, (9, 10, 11)),
    SynthFolder("test", f"{TEST_ONLY}-Al 2mm-part1", 0, (0, 1)),
    SynthFolder("test", f"{TEST_ONLY}-Al 2mm-part2", 1, (2, 3)),
)


def _index(data_root: Path, folders: Sequence[SynthFolder]) -> None:
    """Write the synthetic release under data_root/raw and its index to data_root/derived."""
    release = write_dataset(data_root / "raw" / "tig-al5083", folders)
    write_index(build_index(release), data_root / "derived", {})


@pytest.fixture
def data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The indexed synthetic release, with WELDPILOT_DATA and the working directory in
    tmp_path, so nothing can touch the real data/ or reports/."""
    monkeypatch.setenv("WELDPILOT_DATA", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _index(tmp_path, ONSET_FOLDERS)
    return tmp_path


def _marks(data_root: Path) -> Path:
    return data_root / "annotations" / "onset_marks.csv"


def _write_marks(data_root: Path, *session_ids: str) -> None:
    path = _marks(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"{session_id},,,,\n" for session_id in session_ids)
    path.write_text(f"session_id,pass,onset_frame,marked_on,note\n{rows}")


def _sheets(data_root: Path) -> list[str]:
    return sorted(p.name for p in (data_root / "reports" / "onset_sheets").glob("*.png"))


def test_pick_writes_the_marks_file_and_prints_only_counts(data_root: Path) -> None:
    result = runner.invoke(app, ["onset", "pick"])
    assert result.exit_code == 0, result.output
    assert _marks(data_root).read_text() == (
        f"session_id,pass,onset_frame,marked_on,note\n{TRAIN_A},,,,\n{TRAIN_B},,,,\n"
    )
    assert "qualified  2 training sessions" in result.output
    assert "picked     2 (seed 0" in result.output
    # Blind: no class name reaches the terminal.
    assert [name for name in CLASS_NAMES.values() if name in result.output] == []


def test_crossing_and_test_only_sessions_are_never_picked(data_root: Path) -> None:
    result = runner.invoke(app, ["onset", "pick", "--n", "100"])
    assert result.exit_code == 0, result.output
    picked = _marks(data_root).read_text()
    assert CROSSING not in picked
    assert TEST_ONLY not in picked
    assert "picked     2 " in result.output


def test_pick_wont_overwrite_the_marks_file_without_force(data_root: Path) -> None:
    marks = _marks(data_root)
    marks.parent.mkdir(parents=True)
    marks.write_text(f"session_id,pass,onset_frame,marked_on,note\n{TRAIN_A},1,85,2026-11-04,\n")
    hand_marks = marks.read_bytes()

    result = runner.invoke(app, ["onset", "pick"])
    assert result.exit_code == 1
    assert "--force" in result.output
    assert marks.read_bytes() == hand_marks

    result = runner.invoke(app, ["onset", "pick", "--force"])
    assert result.exit_code == 0, result.output
    assert marks.read_text() == (
        f"session_id,pass,onset_frame,marked_on,note\n{TRAIN_A},,,,\n{TRAIN_B},,,,\n"
    )


@pytest.mark.parametrize(
    "bad_id", [CROSSING, TEST_ONLY, "170101-000000"], ids=["crossing", "test-only", "unknown"]
)
def test_sheets_refuse_a_session_that_is_not_a_training_session(
    data_root: Path, bad_id: str
) -> None:
    _write_marks(data_root, TRAIN_A, bad_id)
    result = runner.invoke(app, ["onset", "sheets"])
    assert result.exit_code == 1
    # The refusal's own message: the frames guard behind it would also exit 1 and name the
    # session, and without this line the test would pass on that alone.
    assert f"not training sessions, so no sheets were drawn: {bad_id!r}" in result.output
    assert _sheets(data_root) == []  # not even the training session's sheet


def test_sheets_draw_one_png_per_marked_session(data_root: Path) -> None:
    _write_marks(data_root, TRAIN_B, TRAIN_A, TRAIN_A)  # TRAIN_A again, as a second pass
    result = runner.invoke(app, ["onset", "sheets"])
    assert result.exit_code == 0, result.output
    assert _sheets(data_root) == [f"{TRAIN_A}.png", f"{TRAIN_B}.png"]


def test_the_sheet_code_receives_frame_numbers_and_paths_only(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    draw = onset_commands.make_sheet

    def spy(*args: Any, **kwargs: Any) -> Image.Image:
        calls.append((args, kwargs))
        return draw(*args, **kwargs)

    monkeypatch.setattr(onset_commands, "make_sheet", spy)
    _write_marks(data_root, TRAIN_A, TRAIN_B)
    result = runner.invoke(app, ["onset", "sheets"])
    assert result.exit_code == 0, result.output

    assert [args[0] for args, _kwargs in calls] == [TRAIN_A, TRAIN_B]
    for args, kwargs in calls:
        session_id, frames = args  # exactly two arguments: nothing else rides along
        assert kwargs == {}
        assert isinstance(session_id, str)
        assert isinstance(frames, dict)
        assert all(type(k) is int and isinstance(v, Path) for k, v in frames.items())


def test_sheets_are_identical_when_every_label_changes(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same release with every folder's class changed. A sheet that showed or used the
    # labels in any way would come out different.
    relabelled = data_root / "relabelled"
    _index(relabelled, [replace(f, label=(f.label + 1) % 6) for f in ONSET_FOLDERS])

    pixels: list[list[tuple[tuple[int, int], bytes]]] = []
    for root in (data_root, relabelled):
        monkeypatch.setenv("WELDPILOT_DATA", str(root))
        monkeypatch.chdir(root)
        _write_marks(root, TRAIN_A, TRAIN_B)
        result = runner.invoke(app, ["onset", "sheets"])
        assert result.exit_code == 0, result.output
        drawn = []
        for session_id in (TRAIN_A, TRAIN_B):
            with Image.open(root / "reports" / "onset_sheets" / f"{session_id}.png") as image:
                drawn.append((image.size, image.tobytes()))
        pixels.append(drawn)
    assert pixels[0] == pixels[1]


def test_sheets_refuse_a_training_session_whose_frames_say_test(data_root: Path) -> None:
    # A stale or edited frames.parquet that disagrees with sessions.parquet. Nothing is
    # drawn, and no image of the session is opened (data rule 2).
    path = data_root / "derived" / FRAMES_FILE
    frames = pd.read_parquet(path)
    first = frames.index[frames["session_id"] == TRAIN_A][0]
    frames.loc[first, "split"] = "test"
    frames.to_parquet(path, index=False)

    _write_marks(data_root, TRAIN_A)
    result = runner.invoke(app, ["onset", "sheets"])
    assert result.exit_code == 1
    assert TRAIN_A in result.output
    assert _sheets(data_root) == []
