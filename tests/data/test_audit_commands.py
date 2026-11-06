"""CLI tests for `weldpilot audit labels` and `weldpilot audit duplicates`, on the synthetic
release indexed by the real writer. Never the real data."""

import hashlib
import itertools
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner, Result

from tests.data.synth import DEFAULT_FOLDERS, SynthFolder, write_dataset
from weldpilot.cli import app
from weldpilot.data import audit_commands, phash
from weldpilot.data.al5083 import CLASS_NAMES
from weldpilot.data.fingerprint import FileDigest
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


# --- weldpilot audit duplicates ---------------------------------------------------------

TRAIN_C = "170815-133921"  # from DEFAULT_FOLDERS: class 3, frames 20, 21 and 23
TRAIN_D = "170913-152931"  # from DEFAULT_FOLDERS: class 0 (frames 5, 6), class 2 (1, 2)
TEST_ONLY_2 = "170906-090000"  # added below: a second test-only session

DUP_FOLDERS = (*DEFAULT_FOLDERS, SynthFolder("test", f"{TEST_ONLY_2}-Al 2mm", 4, (0, 1)))
N_DUP_FRAMES = 26

DUP_MARKS = f"session_id,pass,onset_frame,marked_on,note\n{TRAIN_A},,,,\n"  # TRAIN_A: blind


def _png(folder: str, number: int) -> str:
    """A frame of DUP_FOLDERS, as its path relative to the release."""
    split = next(f.split for f in DUP_FOLDERS if f.name == folder)
    return f"{split}/{folder}/frame_{number:05d}.png"


# Every frame is unique noise, except these pairs: the second frame is a byte copy of the
# first, so the two hashes are 0 bits apart.
PLANTED = {
    "training_vs_test": (_png(f"{TRAIN_C}-Al 2mm", 20), _png(f"{TEST_ONLY}-Al 2mm", 0)),
    "crossing_same_session": (
        _png(f"{CROSSING}-Al 2mm-part1", 7),
        _png(f"{CROSSING}-Al 2mm-part2", 8),
    ),
    "crossing_vs_other_test": (_png(f"{CROSSING}-Al 2mm-part1", 5), _png(f"{TEST_ONLY}-Al 2mm", 1)),
    "training_vs_training, with the marked session": (
        _png(f"{TRAIN_A}-Al 2mm-part1", 0),
        _png(f"{TRAIN_D}-Al 2mm-part1", 5),
    ),
    "training_vs_training": (_png(f"{TRAIN_D}-Al 2mm-part2", 1), _png(f"{TRAIN_C}-Al 2mm", 21)),
    # Never reported:
    "one training session": (
        _png(f"{TRAIN_A}-Al 2mm-part1", 1),
        _png(f"{TRAIN_A}-Al 2mm-part2", 3),
    ),
    "a crossing train/ frame with a training frame": (
        _png(f"{CROSSING}-Al 2mm-part1", 6),
        _png(f"{TRAIN_C}-Al 2mm", 23),
    ),
    "two test sessions": (_png(f"{CROSSING}-Al 2mm-part2", 9), _png(f"{TEST_ONLY_2}-Al 2mm", 0)),
}

# Same pixels, other bytes: saved again without compression, so its hash is 0 bits from its
# source's but its SHA-256 differs. A training_vs_test pair that is not an exact copy.
REENCODED = (_png(f"{TRAIN_D}-Al 2mm-part1", 6), _png(f"{TEST_ONLY_2}-Al 2mm", 1))


@pytest.fixture
def dup_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The synthetic release painted with noise and planted pairs, indexed, with TRAIN_A
    marked blind. WELDPILOT_DATA and the working directory are in tmp_path."""
    monkeypatch.setenv("WELDPILOT_DATA", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    release = write_dataset(tmp_path / "raw" / "tig-al5083", DUP_FOLDERS)
    frames = sorted(p.relative_to(release).as_posix() for p in release.rglob("*.png"))
    assert len(frames) == N_DUP_FRAMES
    for seed, rel in enumerate(frames):
        noise = np.random.default_rng(seed).integers(0, 256, (32, 32), dtype=np.uint8)
        Image.fromarray(noise).save(release / rel)
    for source, copy in PLANTED.values():
        shutil.copyfile(release / source, release / copy)
    source, copy = REENCODED
    with Image.open(release / source) as image:
        image.save(release / copy, compress_level=0)
    # The fixture itself: the planted pairs, and no others, are within 4 bits, and the
    # re-encoded frame has other bytes.
    assert (release / source).read_bytes() != (release / copy).read_bytes()
    hashes = {rel: phash.phash_file(release / rel) for rel in frames}
    close = {
        frozenset((a, b))
        for a, b in itertools.combinations(frames, 2)
        if phash.hamming(hashes[a], hashes[b]) <= 4
    }
    assert close == {frozenset(pair) for pair in [*PLANTED.values(), REENCODED]}
    write_index(build_index(release), tmp_path / "derived", INDEX_PROVENANCE)
    _write_marks(tmp_path, DUP_MARKS)
    return tmp_path


def _duplicates(*args: str) -> Result:
    return runner.invoke(app, ["audit", "duplicates", "--workers", "1", *args])


def _dup_report(root: Path) -> str:
    return (root / OUT / "duplicates.md").read_text(encoding="utf-8")


def test_duplicates_writes_its_outputs_and_the_cache(dup_root: Path) -> None:
    result = _duplicates()
    assert result.exit_code == 0, result.output
    assert _written(dup_root) == ["duplicate_thumbnails", "duplicates.csv", "duplicates.md"]
    cache = pd.read_parquet(dup_root / "derived" / "phash.parquet")
    assert len(cache) == N_DUP_FRAMES
    assert cache.attrs["phash_algorithm"] == phash.PHASH_ALGORITHM
    assert cache.attrs["use"] == "the leakage audit only (data rule 2)"


def test_the_csv_lists_every_pair_that_matters_and_no_label(dup_root: Path) -> None:
    assert _duplicates().exit_code == 0
    thumbs = "duplicate_thumbnails"
    assert (dup_root / OUT / "duplicates.csv").read_text(encoding="utf-8") == (
        "kind,distance,a_split,a_session,a_frame,b_split,b_session,b_frame,thumbnail\n"
        f"training_vs_test,0,train,{TRAIN_C},20,test,{TEST_ONLY},0,"
        f"{thumbs}/training_vs_test__{TRAIN_C}_f00020__{TEST_ONLY}_f00000.png\n"
        f"training_vs_test,0,train,{TRAIN_D},6,test,{TEST_ONLY_2},1,"
        f"{thumbs}/training_vs_test__{TRAIN_D}_f00006__{TEST_ONLY_2}_f00001.png\n"
        f"crossing_same_session,0,train,{CROSSING},7,test,{CROSSING},8,\n"
        f"crossing_vs_other_test,0,train,{CROSSING},5,test,{TEST_ONLY},1,\n"
        f"training_vs_training,0,train,{TRAIN_C},21,train,{TRAIN_D},1,"
        f"{thumbs}/training_vs_training__{TRAIN_C}_f00021__{TRAIN_D}_f00001.png\n"
        f"training_vs_training,0,train,{TRAIN_A},0,train,{TRAIN_D},5,"
        f"{thumbs}/training_vs_training__{TRAIN_A}_f00000__{TRAIN_D}_f00005.png\n"
    )


def test_the_report_counts_pairs_by_kind(dup_root: Path) -> None:
    assert _duplicates().exit_code == 0
    report = _dup_report(dup_root)
    assert f"hashed: {N_DUP_FRAMES} frames from 6 sessions, 20 in train/ and 6 in test/" in report
    assert "Every hash has 32 bits set, so every distance is even." in report
    # Pairs, d = 0, 2 and 4, then frames and sessions on sides a and b.
    assert "| `training_vs_test` | 2 | 2 | 0 | 0 | 2 | 2 | 2 | 2 |" in report
    assert "| `crossing_same_session` | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 1 |" in report
    assert "| `training_vs_training` | 2 | 2 | 0 | 0 | 2 | 2 | 2 | 1 |" in report
    assert f"would take {N_DUP_FRAMES * (N_DUP_FRAMES - 1) // 2} checks" in report


def test_the_class_table_leaves_out_the_marked_session(dup_root: Path) -> None:
    assert _duplicates().exit_code == 0
    report = _dup_report(dup_root)
    assert "1 of the 4 `training_vs_test` and `training_vs_training` pairs" in report
    # TRAIN_A (marked) is left out of the counts and of the totals: with it, the first row
    # would count its frame 0 and its class 0 frames in the last column.
    assert "| 0 good weld | 1 of 2 | 0 of 0 | 0 of 2 |" in report
    assert "| 2 contamination | 0 of 2 | 0 of 0 | 1 of 2 |" in report
    assert "| 3 lack of fusion | 1 of 3 | 0 of 0 | 1 of 3 |" in report
    assert "| 4 misalignment | 0 of 0 | 1 of 2 | 0 of 0 |" in report
    assert "| 5 lack of penetration | 0 of 0 | 1 of 2 | 0 of 0 |" in report
    assert "in the 2 training sessions not marked blind (7 frames)" in report
    assert "or in test/ (6 frames, 3 sessions)" in report
    assert (
        "different classes: 2 of 2 `training_vs_test` and 1 of 1 `training_vs_training`" in report
    )


def test_the_report_lists_the_exact_copy_and_not_the_reencoded_frame(dup_root: Path) -> None:
    assert _duplicates().exit_code == 0
    report = _dup_report(dup_root)
    assert "Their files, 2 in train/ and 2 in test/, were compared by SHA-256" in report
    assert "1 of the 2 pairs are byte-identical." in report
    train, test = PLANTED["training_vs_test"]
    assert f"| `{train}` | `{test}` |" in report
    assert REENCODED[1] not in report  # same pixels, other bytes: not a byte copy


def test_only_the_files_of_pairs_0_bits_apart_are_checksummed(
    dup_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = dup_root / "raw" / "tig-al5083"
    summed: list[str] = []
    real_digest = audit_commands.digest_file

    def spy(path: Path, root: Path) -> FileDigest:
        summed.append(path.relative_to(release).as_posix())
        return real_digest(path, root)

    monkeypatch.setattr(audit_commands, "digest_file", spy)
    result = _duplicates()
    assert result.exit_code == 0, result.output
    assert (
        "copies    1 byte-identical of the 2 training_vs_test pairs 0 bits apart" in result.output
    )
    assert sorted(summed) == sorted([*PLANTED["training_vs_test"], *REENCODED])


def test_duplicates_shows_counts_and_paths_only_in_the_terminal(dup_root: Path) -> None:
    result = _duplicates()
    assert result.exit_code == 0, result.output
    assert f"hashed    {N_DUP_FRAMES} frames: {N_DUP_FRAMES} new, 0 from the cache" in result.output
    assert "pairs     training_vs_training             2" in result.output
    assert "(4 thumbnails)" in result.output
    assert [name for name in CLASS_NAMES.values() if name in result.output] == []


def test_a_second_run_takes_every_hash_from_the_cache_and_writes_the_same_files(
    dup_root: Path,
) -> None:
    def run() -> tuple[str, bytes, bytes]:
        result = _duplicates()
        assert result.exit_code == 0, result.output
        csv = (dup_root / OUT / "duplicates.csv").read_bytes()
        return result.output, csv, (dup_root / OUT / "duplicates.md").read_bytes()

    _first_output, *first_files = run()
    second_output, *second_files = run()
    assert f"0 new, {N_DUP_FRAMES} from the cache" in second_output
    assert first_files == second_files


def test_thumbnails_show_training_frames_only(
    dup_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _duplicates().exit_code == 0  # fills the cache, so the next run decodes no frame
    release = dup_root / "raw" / "tig-al5083"
    opened: list[str] = []
    real_open = Image.open

    def spy(fp: Any, *args: Any, **kwargs: Any) -> Image.Image:
        opened.append(Path(fp).relative_to(release).as_posix())
        return real_open(fp, *args, **kwargs)

    monkeypatch.setattr(Image, "open", spy)
    result = _duplicates()
    assert result.exit_code == 0, result.output
    assert "0 new" in result.output
    assert sorted(opened) == sorted(
        [
            _png(f"{TRAIN_C}-Al 2mm", 20),  # training_vs_test: its test frame isn't drawn
            _png(f"{TRAIN_D}-Al 2mm-part1", 6),  # the same
            _png(f"{TRAIN_C}-Al 2mm", 21),
            _png(f"{TRAIN_D}-Al 2mm-part2", 1),
            _png(f"{TRAIN_A}-Al 2mm-part1", 0),
            _png(f"{TRAIN_D}-Al 2mm-part1", 5),
        ]
    )
    monkeypatch.undo()
    folder = dup_root / OUT / "duplicate_thumbnails"
    sizes = {}
    for path in sorted(folder.iterdir()):
        with Image.open(path) as image:
            sizes[path.name.split("__")[0]] = image.size
    assert sizes == {"training_vs_test": (32, 32), "training_vs_training": (72, 32)}


def test_without_the_marks_file_nothing_is_hashed(dup_root: Path) -> None:
    (dup_root / "annotations" / "onset_marks.csv").unlink()
    result = _duplicates()
    assert result.exit_code == 1
    assert "so the sessions marked blind are unknown" in result.output
    assert not (dup_root / "derived" / "phash.parquet").exists()
    assert _written(dup_root) == []


def test_marks_of_a_session_that_isnt_a_training_session_stop_it(dup_root: Path) -> None:
    _write_marks(dup_root, DUP_MARKS + f"{CROSSING},1,,,\n")
    result = _duplicates()
    assert result.exit_code == 1
    assert f"not training sessions, so nothing was written: {CROSSING!r}" in result.output
    assert not (dup_root / "derived" / "phash.parquet").exists()


@pytest.mark.parametrize("missing", ["frames.parquet", "sessions.parquet"])
def test_a_missing_index_table_is_a_clear_error(dup_root: Path, missing: str) -> None:
    (dup_root / "derived" / missing).unlink()
    result = _duplicates()
    assert result.exit_code == 1
    assert "run `weldpilot data index` first" in result.output
    assert _written(dup_root) == []


def test_a_frames_table_that_disagrees_stops_it_before_any_image_is_opened(
    dup_root: Path,
) -> None:
    path = dup_root / "derived" / FRAMES_FILE
    frames = pd.read_parquet(path)
    first = frames.index[frames["session_id"] == TRAIN_C][0]
    frames.loc[first, "split"] = "test"
    frames.to_parquet(path, index=False)
    result = _duplicates()
    assert result.exit_code == 1
    assert f"training sessions with frames in test/: {TRAIN_C}" in result.output
    assert not (dup_root / "derived" / "phash.parquet").exists()
