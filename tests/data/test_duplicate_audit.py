"""Tests for the near-duplicate audit's pure functions, on tiny synthetic tables."""

import numpy as np
import pandas as pd
import pytest

from weldpilot.data import duplicate_audit as audit
from weldpilot.data.phash import NearPairs

T1, T2 = "170801-100000", "170802-100000"  # training sessions
X = "170803-100000"  # crossing: frames in train/ and test/
Y = "170804-100000"  # test-only
TRAINING, TEST = [T1, T2], [X, Y]

# Out of order on purpose, so orientation can't come from position: a test frame comes
# before training frames, and T2 before T1.
FRAMES = pd.DataFrame(
    [
        (Y, "test", 1),  # 0
        (T2, "train", 5),  # 1
        (T1, "train", 1),  # 2
        (T1, "train", 2),  # 3
        (X, "train", 1),  # 4
        (X, "test", 2),  # 5
        (Y, "test", 2),  # 6
    ],
    columns=["session_id", "split", "frame_number"],
)


def near(*pairs: tuple[int, int, int]) -> NearPairs:
    """NearPairs from (first, second, distance) triples."""
    first, second, distance = (
        np.array(column, dtype=np.int64) for column in zip(*pairs, strict=True)
    )
    return NearPairs(first, second, distance, n_candidates=len(pairs), chunk_widths=(64,))


def rows(pairs: pd.DataFrame) -> list[tuple[object, ...]]:
    return list(pairs.itertuples(index=False, name=None))


def test_each_kind_is_found_and_oriented() -> None:
    found = near((0, 2, 2), (3, 6, 0), (1, 3, 2), (4, 5, 0), (4, 6, 4))
    assert rows(audit.classify_pairs(FRAMES, found, TRAINING)) == [
        # The train/ frame first, closest pair first.
        ("training_vs_test", 0, "train", T1, 2, "test", Y, 2),
        ("training_vs_test", 2, "train", T1, 1, "test", Y, 1),
        ("crossing_same_session", 0, "train", X, 1, "test", X, 2),
        ("crossing_vs_other_test", 4, "train", X, 1, "test", Y, 2),
        # The lower session id first, though T2's frame comes first in the table.
        ("training_vs_training", 2, "train", T1, 2, "train", T2, 5),
    ]


@pytest.mark.parametrize(
    "pair",
    [(2, 3, 0), (1, 4, 2), (0, 5, 0), (0, 6, 2)],
    ids=[
        "one-training-session",
        "crossing-train-frame-with-training-frame",
        "two-test-sessions",
        "one-test-session",
    ],
)
def test_pairs_that_dont_matter_are_dropped(pair: tuple[int, int, int]) -> None:
    pairs = audit.classify_pairs(FRAMES, near(pair), TRAINING)
    assert pairs.empty
    assert list(pairs.columns) == list(audit.PAIR_COLUMNS)


def test_the_frames_table_passes_the_check() -> None:
    audit.check_frames(FRAMES, TRAINING, TEST)


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        (FRAMES.drop(columns="split"), "missing columns: split"),
        (pd.concat([FRAMES, pd.DataFrame([(T1, "val", 9)], columns=FRAMES.columns)]), "val"),
        (
            pd.concat(
                [FRAMES, pd.DataFrame([("170805-100000", "train", 1)], columns=FRAMES.columns)]
            ),
            "neither training nor test sessions: 170805-100000",
        ),
        (
            pd.concat([FRAMES, pd.DataFrame([(T1, "test", 9)], columns=FRAMES.columns)]),
            f"training sessions with frames in test/: {T1}",
        ),
    ],
    ids=["missing-column", "odd-split", "unknown-session", "training-frame-in-test"],
)
def test_a_frames_table_that_disagrees_is_refused(frames: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        audit.check_frames(frames, TRAINING, TEST)


def pair_table(*pairs: tuple[object, ...]) -> pd.DataFrame:
    return pd.DataFrame(list(pairs), columns=list(audit.PAIR_COLUMNS))


PAIRS = pair_table(
    ("training_vs_test", 2, "train", T1, 1, "test", Y, 1),
    ("training_vs_test", 2, "train", T1, 2, "test", Y, 2),
    ("training_vs_test", 4, "train", T1, 3, "test", Y, 1),
    ("training_vs_test", 4, "train", T2, 5, "test", Y, 2),
    ("crossing_same_session", 0, "train", X, 1, "test", X, 2),
    ("training_vs_training", 2, "train", T1, 2, "train", T2, 5),
)


def test_the_closest_pair_of_each_pair_of_sessions_gets_a_thumbnail() -> None:
    names = audit.with_thumbnails(PAIRS)["thumbnail"].tolist()
    folder = audit.THUMBNAIL_DIR
    assert names == [
        f"{folder}/training_vs_test__{T1}_f00001__{Y}_f00001.png",  # 2 bits, lower a frame
        "",
        "",
        f"{folder}/training_vs_test__{T2}_f00005__{Y}_f00002.png",  # its own session pair
        "",  # crossing kinds get none
        f"{folder}/training_vs_training__{T1}_f00002__{T2}_f00005.png",
    ]


def test_a_thumbnail_never_shows_a_test_frame() -> None:
    assert audit.thumbnail_frames("training_vs_test", T1, 1, Y, 1) == [(T1, 1)]
    assert audit.thumbnail_frames("training_vs_training", T1, 2, T2, 5) == [(T1, 2), (T2, 5)]


@pytest.mark.parametrize("kind", ["crossing_same_session", "crossing_vs_other_test"])
def test_crossing_kinds_have_no_thumbnail_frames(kind: str) -> None:
    with pytest.raises(ValueError, match=f"{kind} pairs get no thumbnail"):
        audit.thumbnail_frames(kind, X, 1, Y, 2)


def test_the_csv_has_no_labels() -> None:
    text = audit.duplicates_csv(audit.with_thumbnails(PAIRS))
    header, first, *_ = text.splitlines()
    assert header == "kind,distance,a_split,a_session,a_frame,b_split,b_session,b_frame,thumbnail"
    assert first == (
        f"training_vs_test,2,train,{T1},1,test,{Y},1,"
        f"duplicate_thumbnails/training_vs_test__{T1}_f00001__{Y}_f00001.png"
    )
    assert "label" not in header


# Classes: T1's frames 1 and 2 are classes 0 and 3, T2 (marked) is absent, X's are class 1,
# and Y's frames 1 and 2 are classes 5 and 3.
LABELS = pd.DataFrame(
    [
        (T1, "train", 1, 0),
        (T1, "train", 2, 3),
        (X, "train", 1, 1),
        (X, "test", 2, 1),
        (Y, "test", 1, 5),
        (Y, "test", 2, 3),
    ],
    columns=list(audit.LABEL_COLUMNS),
)


LABELLED_PAIRS = pair_table(
    ("training_vs_test", 2, "train", T1, 1, "test", Y, 1),  # classes 0 and 5
    ("training_vs_test", 0, "train", T1, 2, "test", Y, 2),  # classes 3 and 3
    ("crossing_same_session", 0, "train", X, 1, "test", X, 2),  # not a class kind
    ("training_vs_training", 2, "train", T1, 2, "train", T2, 5),  # T2 is marked
)
"""Every frame of these has a row in LABELS, except T2's, which is marked."""


def test_a_flagged_frame_without_a_label_is_refused() -> None:
    # T1's frame 3 isn't in LABELS: the index and the pairs disagree.
    with pytest.raises(ValueError, match="a flagged frame has no row in the labels table"):
        audit.class_table(PAIRS, LABELS, TRAINING, marked=[T2])


def test_the_class_table_leaves_out_pairs_with_a_marked_session() -> None:
    table = audit.class_table(LABELLED_PAIRS, LABELS, TRAINING, marked=[T2])
    assert table.n_left_out == 1
    assert table.counted == {"training_vs_test": 2, "training_vs_training": 0}
    assert table.mixed == {"training_vs_test": 1, "training_vs_training": 0}
    # T1 alone counts among the training sessions; test/ holds X's frame 2 and Y's frames.
    assert (table.n_training_sessions, table.n_training_frames) == (1, 2)
    assert (table.n_test_sessions, table.n_test_frames) == (2, 3)
    by_class = {row.label: row for row in table.rows}
    assert list(by_class) == [0, 1, 2, 3, 4, 5]
    assert by_class[0] == audit.ClassRow(0, 1, 0, 0, training_frames=1, test_frames=0)
    assert by_class[1] == audit.ClassRow(1, 0, 0, 0, training_frames=0, test_frames=1)
    assert by_class[3] == audit.ClassRow(3, 1, 1, 0, training_frames=1, test_frames=1)
    assert by_class[5] == audit.ClassRow(5, 0, 1, 0, training_frames=0, test_frames=1)


def test_labels_of_a_marked_session_are_refused() -> None:
    with_marked = pd.concat([LABELS, pd.DataFrame([(T2, "train", 5, 2)], columns=LABELS.columns)])
    with pytest.raises(ValueError, match=f"holds sessions marked blind: {T2}"):
        audit.class_table(PAIRS, with_marked, TRAINING, marked=[T2])


def scope(fewest_bits: int, most_bits: int) -> audit.Scope:
    return audit.Scope(
        n_frames=7,
        n_sessions=4,
        n_train_frames=4,
        n_test_frames=3,
        n_training_sessions=2,
        n_test_sessions=2,
        n_crossing_sessions=1,
        n_marked=1,
        fewest_bits=fewest_bits,
        most_bits=most_bits,
        max_distance=4,
        n_candidates=9,
        chunk_widths=(13, 13, 13, 13, 12),
        n_thumbnails=2,
    )


PROVENANCE = audit.Provenance("c0de", "1dx", "f1n9", "dct-phash-v1")

TRAIN_PNG = f"train/{T1}-Al 2mm/frame_00002.png"
TEST_PNG = f"test/{Y}-Al 2mm/frame_00002.png"
NO_COPIES = audit.ExactCopies(n_checked=1, n_train_files=1, n_test_files=1, identical=())


def test_same_hash_pairs_are_training_vs_test_pairs_0_bits_apart() -> None:
    pairs = pair_table(
        ("training_vs_test", 0, "train", T1, 2, "test", Y, 2),  # this one
        ("training_vs_test", 2, "train", T1, 1, "test", Y, 1),
        ("crossing_same_session", 0, "train", X, 1, "test", X, 2),
        ("training_vs_training", 0, "train", T1, 2, "train", T2, 5),
    )
    assert rows(audit.same_hash_pairs(pairs)) == [rows(pairs)[0]]


def test_exact_copies_are_the_pairs_whose_files_have_one_checksum() -> None:
    other_train, other_test = (
        f"train/{T2}-Al 2mm/frame_00005.png",
        f"test/{Y}-Al 2mm/frame_00001.png",
    )
    paths = [(TRAIN_PNG, TEST_PNG), (other_train, other_test), (TRAIN_PNG, other_test)]
    sha256 = {TRAIN_PNG: "aa", TEST_PNG: "aa", other_train: "bb", other_test: "cc"}
    assert audit.exact_copies(paths, sha256) == audit.ExactCopies(
        n_checked=3, n_train_files=2, n_test_files=2, identical=((TRAIN_PNG, TEST_PNG),)
    )


def test_the_report_lists_each_exact_copy_with_its_paths() -> None:
    table = audit.class_table(LABELLED_PAIRS, LABELS, TRAINING, marked=[T2])
    copies = audit.ExactCopies(
        n_checked=1, n_train_files=1, n_test_files=1, identical=((TRAIN_PNG, TEST_PNG),)
    )
    report = audit.duplicates_markdown(LABELLED_PAIRS, table, copies, scope(32, 32), PROVENANCE)
    assert "Their files, 1 in train/ and 1 in test/, were compared by SHA-256" in report
    assert "1 of the 1 pairs are byte-identical." in report
    assert f"| `{TRAIN_PNG}` | `{TEST_PNG}` |" in report


def test_the_report_says_when_no_file_is_an_exact_copy() -> None:
    table = audit.class_table(LABELLED_PAIRS, LABELS, TRAINING, marked=[T2])
    report = audit.duplicates_markdown(LABELLED_PAIRS, table, NO_COPIES, scope(32, 32), PROVENANCE)
    assert "0 of the 1 pairs are byte-identical." in report
    assert "No test file is a byte copy of a training file." in report
    assert "| train/ file | test/ file |" not in report


@pytest.mark.parametrize(
    ("bits", "columns"),
    [
        ((32, 32), "| d = 0 | d = 2 | d = 4 |"),
        ((31, 33), "| d = 0 | d = 1 | d = 2 | d = 3 | d = 4 |"),
    ],
    ids=["equal-bit-counts", "unequal-bit-counts"],
)
def test_odd_distances_are_shown_only_when_they_can_occur(
    bits: tuple[int, int], columns: str
) -> None:
    table = audit.class_table(LABELLED_PAIRS, LABELS, TRAINING, marked=[T2])
    report = audit.duplicates_markdown(LABELLED_PAIRS, table, NO_COPIES, scope(*bits), PROVENANCE)
    assert f"| Kind | Pairs {columns} Frames a |" in report


def test_the_report_states_the_search_cost_and_provenance() -> None:
    table = audit.class_table(LABELLED_PAIRS, LABELS, TRAINING, marked=[T2])
    report = audit.duplicates_markdown(LABELLED_PAIRS, table, NO_COPIES, scope(32, 32), PROVENANCE)
    assert "Every hash has 32 bits set, so every distance is even." in report
    assert "would take 21 checks" in report  # 7 * 6 / 2
    assert "That took 9 checks (42.86% of all pairs)" in report
    assert "- Code: git `c0de`." in report
    assert "built at git `1dx` from the dataset with fingerprint `f1n9`" in report
