import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.data.synth import README_TEXT
from weldpilot.data.al5083 import FINGERPRINT_PARTS, IGNORED_NAMES
from weldpilot.data.fingerprint import (
    FileDigest,
    FingerprintRecord,
    PartCounts,
    combine,
    compare,
    dump_record,
    fingerprint_dataset,
    parse_record,
)

MakeDataset = Callable[..., Path]


def _fingerprint(root: Path, workers: int = 4) -> FingerprintRecord:
    return fingerprint_dataset(root, FINGERPRINT_PARTS, IGNORED_NAMES, workers=workers)


@pytest.mark.parametrize(
    "rel_path", ["README", "train/170904-141730-Al 2mm-part1/frame_00000.png", "test/test.json"]
)
def test_one_changed_byte_changes_the_fingerprint(make_dataset: MakeDataset, rel_path: str) -> None:
    root = make_dataset()
    before = _fingerprint(root)
    path = root / rel_path
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))
    after = _fingerprint(root)
    assert after.fingerprint != before.fingerprint
    assert after.parts == before.parts  # same counts and sizes: only the content changed


def test_ds_store_files_and_the_al5083_copy_are_ignored(make_dataset: MakeDataset) -> None:
    root = make_dataset()
    before = _fingerprint(root)
    folder = root / "train" / "170904-141730-Al 2mm-part1"
    for where in (root, root / "train", root / "test", folder):
        (where / ".DS_Store").write_bytes(b"Finder state")
    (root / "al5083" / "train").mkdir(parents=True)
    (root / "al5083" / "train" / "train.json").write_text("{}")
    (root / "al5083" / "README").write_text(README_TEXT)
    assert _fingerprint(root) == before


def test_counts_cover_each_part(make_dataset: MakeDataset) -> None:
    record = _fingerprint(make_dataset())
    assert record.dataset == "tig-al5083"
    assert (record.parts["train"].png, record.parts["train"].other) == (20, 1)
    assert (record.parts["test"].png, record.parts["test"].other) == (4, 1)
    assert record.parts["README"] == PartCounts(png=0, other=1, bytes=len(README_TEXT))


def test_thread_count_does_not_change_the_result(make_dataset: MakeDataset) -> None:
    root = make_dataset()
    assert _fingerprint(root, workers=1) == _fingerprint(root, workers=4)


def test_a_missing_part_is_an_error(make_dataset: MakeDataset) -> None:
    root = make_dataset()
    (root / "README").unlink()
    with pytest.raises(FileNotFoundError, match="README"):
        _fingerprint(root)


def test_combine_depends_on_paths_not_on_order() -> None:
    a = FileDigest("train/a.png", 3, "aa")
    b = FileDigest("train/b.png", 3, "bb")
    assert combine([a, b]) == combine([b, a])
    assert combine([a, b]) != combine([FileDigest("train/c.png", 3, "aa"), b])


def test_combine_hashes_the_documented_manifest() -> None:
    # The format is documented in configs/dataset.yaml, so anyone can recompute it.
    files = [FileDigest("b", 2, "y"), FileDigest("a", 1, "x")]
    assert combine(files) == hashlib.sha256(b"a\t1\tx\nb\t2\ty\n").hexdigest()


def test_record_round_trips_through_yaml(make_dataset: MakeDataset) -> None:
    record = _fingerprint(make_dataset())
    text = dump_record(record, IGNORED_NAMES)
    assert text.startswith("# Written by `weldpilot data verify --record`")
    assert "Files named .DS_Store are skipped." in text
    assert parse_record(text) == record


@pytest.mark.parametrize(
    "text",
    [
        "fingerprint: [",  # not YAML
        "dataset: tig-al5083\nfingerprint: abc\nparts: {}\n",  # not 64 hex digits
        f"dataset: tig-al5083\nfingerprint: {'0' * 64}\nparts: {{}}\nextra: 1\n",  # unknown key
    ],
)
def test_parse_record_rejects_invalid_files(text: str) -> None:
    with pytest.raises(ValueError):
        parse_record(text)


def test_compare_lists_count_changes_before_the_fingerprint(make_dataset: MakeDataset) -> None:
    root = make_dataset()
    recorded = _fingerprint(root)
    assert compare(recorded, recorded) == []
    (root / "train" / "notes.txt").write_text("a stray file")
    problems = compare(recorded, _fingerprint(root))
    assert len(problems) == 2
    assert problems[0].startswith("train: recorded 20 PNG + 1 other files")
    assert problems[1].startswith(f"fingerprint: recorded {recorded.fingerprint}, now ")
