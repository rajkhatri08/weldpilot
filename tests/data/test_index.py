import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from tests.data.synth import DEFAULT_FOLDERS, SynthFolder, write_png
from weldpilot.data.al5083 import ASSUMED_FPS
from weldpilot.data.index import (
    FRAMES_FILE,
    SEGMENTS_FILE,
    SESSIONS_FILE,
    DatasetError,
    DatasetIndex,
    Frame,
    Segment,
    Session,
    build_index,
    parse_labels,
    write_index,
)

MakeDataset = Callable[..., Path]


def _frames(index: DatasetIndex, session_id: str) -> list[Frame]:
    return [f for f in index.frames if f.session_id == session_id]


def _segment(index: DatasetIndex, folder: str) -> Segment:
    return next(s for s in index.segments if s.folder == folder)


def _session(index: DatasetIndex, session_id: str) -> Session:
    return next(s for s in index.sessions if s.session_id == session_id)


def test_interleaved_parts_follow_frame_numbers(make_dataset: MakeDataset) -> None:
    frames = _frames(build_index(make_dataset()), "170904-141730")
    assert [f.frame_number for f in frames] == [0, 1, 2, 3, 4, 7, 8, 10, 11, 15]
    assert [f.part for f in frames] == [1, 1, 1, 2, 2, 1, 1, 3, 3, 1]
    assert [f.gap_before for f in frames] == [None, 0, 0, 0, 0, 2, 0, 1, 0, 3]


def test_part_numbers_need_not_follow_time(make_dataset: MakeDataset) -> None:
    frames = _frames(build_index(make_dataset()), "170913-152931")
    assert [(f.part, f.frame_number) for f in frames] == [(2, 1), (2, 2), (1, 5), (1, 6)]


def test_frames_carry_paths_labels_and_time(make_dataset: MakeDataset) -> None:
    first = _frames(build_index(make_dataset()), "170815-133921")[0]
    assert first.rel_path == "train/170815-133921-Al 2mm/frame_00020.png"
    assert (first.part, first.label, first.gap_before) == (None, 3, None)
    assert first.t_s == 20 / ASSUMED_FPS


def test_n_missing_counts_numbers_absent_from_the_release(make_dataset: MakeDataset) -> None:
    index = build_index(make_dataset())
    part1 = _segment(index, "170904-141730-Al 2mm-part1")
    # Range 0..15 holds 16 numbers: 6 in part1, 4 in parts 2 and 3 (present, not missing)
    # and 6 absent from the release (5, 6, 9, 12, 13, 14).
    assert (part1.first_frame, part1.last_frame, part1.n_frames, part1.n_missing) == (0, 15, 6, 6)
    assert _segment(index, "170904-141730-Al 2mm-part2").n_missing == 0
    assert _segment(index, "170815-133921-Al 2mm").n_missing == 1


def test_a_session_with_folders_in_both_splits_crosses(make_dataset: MakeDataset) -> None:
    index = build_index(make_dataset())
    flags = {s.session_id: (s.in_train, s.in_test, s.crosses_split) for s in index.sessions}
    assert flags["170904-150144"] == (True, True, True)
    assert flags["170904-141730"] == (True, False, False)
    assert flags["170905-112213"] == (False, True, False)
    assert {f.split for f in _frames(index, "170904-150144")} == {"train", "test"}


def test_good_to_fault_onsets_follow_the_timeline(make_dataset: MakeDataset) -> None:
    index = build_index(make_dataset())
    interleaved = _session(index, "170904-141730")
    assert interleaved.labels == (0, 2, 4)
    assert (interleaved.has_good, interleaved.n_good_to_fault, interleaved.max_gap) == (True, 2, 3)
    assert (interleaved.n_folders, interleaved.n_frames) == (3, 10)
    assert _session(index, "170904-150144").n_good_to_fault == 1  # good in train/, fault in test/
    assert _session(index, "170913-152931").n_good_to_fault == 0  # the fault comes first
    no_good = _session(index, "170815-133921")
    assert (no_good.has_good, no_good.n_good_to_fault, no_good.max_gap) == (False, 0, 1)


def test_a_one_frame_session_has_no_gap(make_dataset: MakeDataset) -> None:
    index = build_index(make_dataset([SynthFolder("train", "170815-134756-Al 2mm", 0, (5,))]))
    assert index.sessions[0].max_gap is None
    assert index.frames[0].gap_before is None


def test_headers_are_read_for_train_frames_only(make_dataset: MakeDataset) -> None:
    index = build_index(make_dataset())
    for f in index.frames:
        expected = (8, 8, "L") if f.split == "train" else (None, None, None)
        assert (f.width, f.height, f.mode) == expected


def test_test_images_are_never_opened(
    make_dataset: MakeDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset()
    opened: list[Path] = []
    real_open = Image.open

    def spy(fp: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(Path(fp))
        return real_open(fp, *args, **kwargs)

    monkeypatch.setattr(Image, "open", spy)
    build_index(root)
    # Every training header is read, and nothing else: no test image is opened.
    assert set(opened) == set((root / "train").glob("*/frame_*.png"))


@pytest.mark.parametrize(
    ("split", "folder"), [("train", "170815-133921-Al 2mm"), ("test", "170905-112213-Al 2mm")]
)
def test_a_png_without_a_label_is_an_error(
    make_dataset: MakeDataset, split: str, folder: str
) -> None:
    root = make_dataset()
    write_png(root / split / folder / "frame_00099.png")
    with pytest.raises(DatasetError, match="PNGs with no label") as err:
        build_index(root)
    assert f"{split}/{folder}/frame_00099.png" in str(err.value)


@pytest.mark.parametrize(
    ("split", "folder"), [("train", "170815-133921-Al 2mm"), ("test", "170905-112213-Al 2mm")]
)
def test_a_label_without_a_png_is_an_error(
    make_dataset: MakeDataset, split: str, folder: str
) -> None:
    root = make_dataset()
    path = root / split / f"{split}.json"
    labels = json.loads(path.read_text())
    labels[f"{folder}/frame_00099.png"] = 0
    path.write_text(json.dumps(labels))
    with pytest.raises(DatasetError, match="labels with no PNG") as err:
        build_index(root)
    assert f"{split}/{folder}/frame_00099.png" in str(err.value)


def test_a_folder_with_two_classes_is_an_error(make_dataset: MakeDataset) -> None:
    root = make_dataset()
    path = root / "train" / "train.json"
    labels = json.loads(path.read_text())
    labels["170904-141730-Al 2mm-part1/frame_00007.png"] = 3
    path.write_text(json.dumps(labels))
    with pytest.raises(DatasetError, match="folders holding more than one class") as err:
        build_index(root)
    assert "train/170904-141730-Al 2mm-part1 (classes 0, 3)" in str(err.value)


def test_a_frame_number_twice_in_one_session_is_an_error(make_dataset: MakeDataset) -> None:
    # Frame 7 of session 170904-141730 is already in part1 (train/).
    extra = SynthFolder("test", "170904-141730-Al 2mm-part4", 1, (7,))
    with pytest.raises(DatasetError, match="appear twice in one session") as err:
        build_index(make_dataset([*DEFAULT_FOLDERS, extra]))
    assert "session 170904-141730 frame 7" in str(err.value)


def test_a_bad_folder_name_is_an_error(make_dataset: MakeDataset) -> None:
    extra = SynthFolder("train", "170904-141730-Al 2mm copy", 0, (40,))
    with pytest.raises(DatasetError, match="folder names") as err:
        build_index(make_dataset([*DEFAULT_FOLDERS, extra]))
    assert "train/170904-141730-Al 2mm copy" in str(err.value)


def test_an_unexpected_file_is_an_error(make_dataset: MakeDataset) -> None:
    root = make_dataset()
    (root / "train" / "170815-133921-Al 2mm" / "notes.txt").write_text("a stray file")
    with pytest.raises(DatasetError, match="unexpected files or folders") as err:
        build_index(root)
    assert "train/170815-133921-Al 2mm/notes.txt" in str(err.value)


def test_ds_store_files_are_ignored(make_dataset: MakeDataset) -> None:
    root = make_dataset()
    (root / "train" / ".DS_Store").write_bytes(b"Finder state")
    (root / "train" / "170815-133921-Al 2mm" / ".DS_Store").write_bytes(b"Finder state")
    assert len(build_index(root).frames) == 24


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"a/frame_00000.png": 0, "a/frame_00000.png": 1}', "keys that appear more than once"),
        ('{"a/frame_00000.png": 7}', "not a class 0-5"),
        ('{"a/frame_00000.png": true}', "not a class 0-5"),  # bool is a subclass of int
        ('{"a/frame_00000.png": "0"}', "not a class 0-5"),
        ("[0, 1]", "must hold a JSON object"),
        ('{"a/frame_00000.png": ', "not valid JSON"),
    ],
)
def test_label_files_are_read_strictly(text: str, message: str) -> None:
    _, problems = parse_labels(text, "train/train.json")
    assert any(message in p for p in problems), problems


def test_parquet_round_trip_keeps_columns_types_and_provenance(
    make_dataset: MakeDataset, tmp_path: Path
) -> None:
    index = build_index(make_dataset())
    provenance = {"git_sha": "abc123", "dataset_fingerprint": "f" * 64, "assumed_fps": 55.0}
    write_index(index, tmp_path / "derived", provenance)
    frames = pd.read_parquet(tmp_path / "derived" / FRAMES_FILE)
    segments = pd.read_parquet(tmp_path / "derived" / SEGMENTS_FILE)
    sessions = pd.read_parquet(tmp_path / "derived" / SESSIONS_FILE)

    # Column order as specified for each file.
    assert list(frames.columns) == [
        "session_id",
        "part",
        "folder",
        "split",
        "frame_number",
        "rel_path",
        "label",
        "t_s",
        "gap_before",
        "width",
        "height",
        "mode",
    ]
    assert list(segments.columns) == [
        "folder",
        "session_id",
        "part",
        "split",
        "label",
        "n_frames",
        "first_frame",
        "last_frame",
        "n_missing",
    ]
    assert list(sessions.columns) == [
        "session_id",
        "in_train",
        "in_test",
        "crosses_split",
        "n_folders",
        "n_frames",
        "labels",
        "has_good",
        "n_good_to_fault",
        "first_frame",
        "last_frame",
        "max_gap",
    ]
    assert (len(frames), len(segments), len(sessions)) == (24, 9, 5)
    for table in (frames, segments, sessions):
        assert table.attrs == provenance
    assert str(frames["part"].dtype) == "Int64"
    assert frames["part"].isna().sum() == 5  # the two folders without a part suffix
    assert frames.loc[frames["split"] == "test", "width"].isna().all()
    interleaved = sessions.set_index("session_id").loc["170904-141730"]
    assert list(interleaved["labels"]) == [0, 2, 4]
