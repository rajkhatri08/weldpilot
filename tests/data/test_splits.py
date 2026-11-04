"""Spec tests for `weldpilot.data.splits`, written before the module.

Every analysis gets its sessions from these functions (data rule 4), so they are where data
rule 3 is enforced: a crossing session is never a training session. Each test writes a tiny
sessions.parquet to tmp_path; none reads real data. The `grouped_folds` tests at the end were
written before that function.
"""

import os
import random
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pytest

# The module, not its functions: pytest collects every function named test_* in this file,
# so an imported `test_sessions` would be run as a test.
from weldpilot.data import splits
from weldpilot.data.index import SESSIONS_FILE, build_index, write_index

MakeDataset = Callable[..., Path]
Selector = Callable[[pd.DataFrame], list[str]]
# session_id, in_train, in_test, crosses_split. None stands for a null.
Row = tuple[str | None, bool | None, bool | None, bool | None]

COLUMNS = {"session_id": "str", "in_train": "bool", "in_test": "bool", "crosses_split": "bool"}
"""The columns load_sessions needs, with the types `weldpilot data index` writes."""

# Session ids from tests/data/synth.py.
TRAIN_ONLY: Row = ("170815-133921", True, False, False)
TEST_ONLY: Row = ("170905-112213", False, True, False)
CROSSING: Row = ("170904-150144", True, True, True)
BAD_ID = "170913-152931"  # the offending session in the error tests

# Names, not the functions: each is looked up when its test runs, so while one selector is
# unwritten only its own tests fail, and the rest of the file still runs.
SELECTORS = ["training_sessions", "test_sessions"]


def _table(rows: Sequence[Row]) -> pd.DataFrame:
    """Rows as a sessions table, typed as `weldpilot data index` writes it."""
    table = pd.DataFrame(list(rows), columns=list(COLUMNS))
    # pandas infers those types from the values. A table with no rows has none, so it is
    # cast; the others aren't, because a cast to bool would turn a None into False.
    return table.astype(COLUMNS) if table.empty else table


def _write(tmp_path: Path, table: pd.DataFrame) -> Path:
    """Write `table` as sessions.parquet in tmp_path/derived and return that folder."""
    derived = tmp_path / "derived"
    derived.mkdir(exist_ok=True)
    table.to_parquet(derived / SESSIONS_FILE, index=False)
    return derived


def _load(tmp_path: Path, rows: Sequence[Row]) -> pd.DataFrame:
    return splits.load_sessions(_write(tmp_path, _table(rows)))


@pytest.mark.parametrize(
    "missing",
    [
        ("session_id",),
        ("in_train",),
        ("in_test",),
        ("crosses_split",),
        ("in_train", "crosses_split"),  # every missing column is named, not only the first
    ],
    ids="+".join,
)
def test_load_sessions_names_every_missing_column(tmp_path: Path, missing: tuple[str, ...]) -> None:
    table = _table([TRAIN_ONLY, TEST_ONLY, CROSSING]).drop(columns=list(missing))
    derived = _write(tmp_path, table)
    with pytest.raises(ValueError) as err:
        splits.load_sessions(derived)
    for column in missing:
        assert column in str(err.value)


# Never indexed, or indexed and the file since deleted.
@pytest.mark.parametrize(
    "make_folder", [False, True], ids=["no-derived-folder", "empty-derived-folder"]
)
def test_a_missing_sessions_parquet_is_a_file_not_found_error(
    tmp_path: Path, make_folder: bool
) -> None:
    derived = tmp_path / "derived"
    if make_folder:
        derived.mkdir()
    with pytest.raises(FileNotFoundError):
        splits.load_sessions(derived)


def test_an_empty_table_has_no_sessions(tmp_path: Path) -> None:
    sessions = _load(tmp_path, [])
    assert sessions.empty
    assert splits.training_sessions(sessions) == []
    assert splits.test_sessions(sessions) == []


def test_a_crossing_session_is_a_test_session_only(tmp_path: Path) -> None:
    sessions = _load(tmp_path, [TRAIN_ONLY, TEST_ONLY, CROSSING])
    assert splits.training_sessions(sessions) == ["170815-133921"]  # data rule 3
    assert splits.test_sessions(sessions) == ["170904-150144", "170905-112213"]


def test_ids_come_back_sorted_in_a_plain_list(tmp_path: Path) -> None:
    # Each split's ids are in neither sorted nor reverse order in the table.
    rows: list[Row] = [
        ("170913-152931", True, False, False),
        ("170905-112213", False, True, False),
        ("170815-133921", True, False, False),
        ("170906-153326", True, True, True),
        ("170904-141730", True, False, False),
        ("170904-150144", True, True, True),
    ]
    sessions = _load(tmp_path, rows)
    train_ids = splits.training_sessions(sessions)
    test_ids = splits.test_sessions(sessions)
    assert isinstance(train_ids, list)  # not a pandas Series or a numpy array
    assert isinstance(test_ids, list)
    assert train_ids == ["170815-133921", "170904-141730", "170913-152931"]
    assert test_ids == ["170904-150144", "170905-112213", "170906-153326"]


@pytest.mark.parametrize(
    "crosses_split", [False, True], ids=["not-marked-crossing", "marked-crossing"]
)
@pytest.mark.parametrize("selector", SELECTORS)
def test_a_session_in_neither_split_is_an_error(
    tmp_path: Path, selector: str, crosses_split: bool
) -> None:
    select: Selector = getattr(splits, selector)
    # Loaded before pytest.raises, so the error must come from the selector itself.
    sessions = _load(tmp_path, [TRAIN_ONLY, (BAD_ID, False, False, crosses_split), TEST_ONLY])
    with pytest.raises(ValueError, match=BAD_ID):
        select(sessions)


@pytest.mark.parametrize(
    "flags",  # in_train, in_test, crosses_split
    [
        pytest.param((True, False, True), id="train-only-marked-crossing"),
        pytest.param((False, True, True), id="test-only-marked-crossing"),
        # The dangerous one: a selector that trusted crosses_split would train on test folders.
        pytest.param((True, True, False), id="in-both-not-marked-crossing"),
    ],
)
@pytest.mark.parametrize("selector", SELECTORS)
def test_a_wrong_crosses_split_is_an_error(
    tmp_path: Path, selector: str, flags: tuple[bool, bool, bool]
) -> None:
    select: Selector = getattr(splits, selector)
    sessions = _load(tmp_path, [TRAIN_ONLY, (BAD_ID, *flags), TEST_ONLY])
    with pytest.raises(ValueError, match=BAD_ID):
        select(sessions)


@pytest.mark.parametrize(
    "repeat",
    [
        pytest.param((BAD_ID, True, False, False), id="identical-rows"),
        # The dangerous one: each row is valid alone, but together they put the session in
        # both splits, and neither row marks it crossing.
        pytest.param((BAD_ID, False, True, False), id="train-only-and-test-only-rows"),
    ],
)
@pytest.mark.parametrize("selector", SELECTORS)
def test_a_repeated_session_id_is_an_error(tmp_path: Path, selector: str, repeat: Row) -> None:
    select: Selector = getattr(splits, selector)
    # The repeat isn't next to the first row, so comparing neighbours won't find it.
    sessions = _load(tmp_path, [(BAD_ID, True, False, False), TRAIN_ONLY, repeat, TEST_ONLY])
    with pytest.raises(ValueError, match=BAD_ID):
        select(sessions)


# For each flag, a row with a null there. Read as False, each null would leave a valid row,
# so only a check for nulls catches it.
NULL_ROWS: dict[str, Row] = {
    "in_train": (BAD_ID, None, True, False),
    "in_test": (BAD_ID, True, None, False),
    "crosses_split": (BAD_ID, True, False, None),
}


# pandas holds a missing bool as None in an object column, or as pd.NA in its nullable
# boolean type. The selectors must catch both.
@pytest.mark.parametrize("dtype", ["object", "boolean"], ids=["None", "pd.NA"])
@pytest.mark.parametrize("column", list(NULL_ROWS))
@pytest.mark.parametrize("selector", SELECTORS)
def test_a_null_flag_is_an_error(tmp_path: Path, selector: str, column: str, dtype: str) -> None:
    select: Selector = getattr(splits, selector)
    table = _table([TRAIN_ONLY, NULL_ROWS[column], TEST_ONLY]).astype({column: dtype})
    sessions = splits.load_sessions(_write(tmp_path, table))
    assert sessions[column].isna().any()  # the null survived load_sessions
    with pytest.raises(ValueError, match=BAD_ID):
        select(sessions)


@pytest.mark.parametrize("selector", SELECTORS)
def test_a_row_without_a_session_id_is_an_error(tmp_path: Path, selector: str) -> None:
    select: Selector = getattr(splits, selector)
    # Valid flags, so only a check for the missing id catches the middle row.
    sessions = _load(tmp_path, [TRAIN_ONLY, (None, True, False, False), TEST_ONLY])
    assert sessions["session_id"].isna().tolist() == [False, True, False]  # kept by load_sessions
    # With no id to name, the message names the row: 1 is the middle one, counting from 0.
    with pytest.raises(ValueError, match="row 1"):
        select(sessions)


def test_load_sessions_reads_the_table_the_index_writes(
    make_dataset: MakeDataset, tmp_path: Path
) -> None:
    # Written by the code behind `weldpilot data index`, from the synthetic release in
    # tests/data/synth.py: a column renamed in index.py fails here, not on real data.
    derived = tmp_path / "derived"
    write_index(build_index(make_dataset()), derived, {})
    sessions = splits.load_sessions(derived)
    assert splits.training_sessions(sessions) == ["170815-133921", "170904-141730", "170913-152931"]
    assert splits.test_sessions(sessions) == ["170904-150144", "170905-112213"]


# grouped_folds(sessions, k=5, seed=0): k folds of (training ids, validation ids), both
# sorted. The validation folds split the training sessions, stratified by each session's
# rarest class. Every call is `splits.grouped_folds`, looked up when the test runs, so while
# the function is unwritten only these tests fail.

FoldRow = tuple[str | None, bool | None, bool | None, bool | None, list[int]]
"""Row's four columns, then labels: the session's classes, as `weldpilot data index` writes
them."""

BURN_THROUGH = "170904-160401"

# A training set shaped so that breaking any rule changes the folds. Class: the training
# sessions that hold it. 0: 10, 1: 1 (burn through, as in the real release), 2: 3, 3: 3,
# 4: 2, 5: 4. The comments give each session's rarest class.
FOLD_ROWS: list[FoldRow] = [
    (*TRAIN_ONLY, [3]),  # 3, its only class
    ("170904-141730", True, False, False, [0, 2, 4]),  # 4 (2 sessions) before 2 (3)
    (BURN_THROUGH, True, False, False, [0, 1]),  # 1
    ("170905-101500", True, False, False, [0, 5]),  # 5
    ("170905-113000", True, False, False, [0, 5]),  # 5
    ("170905-124500", True, False, False, [0, 4, 5]),  # 4 (2) before 5 (4)
    ("170906-091200", True, False, False, [0, 2, 3]),  # 2 and 3 tie at 3: the lower, 2
    ("170906-102400", True, False, False, [0]),  # 0: good weld only
    ("170907-100000", True, False, False, [0, 3, 5]),  # 3 (3) before 5 (4)
    ("170908-143000", True, False, False, [0, 2]),  # 2
    ("170914-100000", True, False, False, [0]),  # 0
    # Never in a fold, and their classes don't count. If they did, class 4 would have four
    # sessions and 170904-141730 would move to stratum 2.
    (*CROSSING, [0, 4]),
    (*TEST_ONLY, [0, 4]),
]
TRAINING_IDS = sorted(str(row[0]) for row in FOLD_ROWS if row[1] and not row[2])

# The strata, worked out by hand from the counts above, in dealing order: rarest class first,
# ties to the lower class. Rarity is the class's number of training sessions, not the
# stratum's size: five strata hold two sessions each, yet 0 goes last.
STRATA: dict[int, set[str]] = {
    1: {BURN_THROUGH},
    4: {"170904-141730", "170905-124500"},
    2: {"170906-091200", "170908-143000"},  # 2 and 3 tie at 3 sessions: 2 first
    3: {"170815-133921", "170907-100000"},
    5: {"170905-101500", "170905-113000"},
    0: {"170906-102400", "170914-100000"},
}


def _fold_table(rows: Sequence[FoldRow]) -> pd.DataFrame:
    """Rows as a sessions table with a labels column of class lists."""
    return pd.DataFrame(list(rows), columns=[*COLUMNS, "labels"])


def _load_folds(tmp_path: Path, rows: Sequence[FoldRow]) -> pd.DataFrame:
    """Write the rows as sessions.parquet and read them back, as `_load` does."""
    return splits.load_sessions(_write(tmp_path, _fold_table(rows)))


def _shuffled(rows: Sequence[FoldRow], seed: int) -> list[FoldRow]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _dealt(k: int) -> dict[int, list[int]]:
    """The validation folds each stratum must fill, sorted.

    One counter deals the sessions round-robin from fold 0, through the strata in order, so
    each stratum takes the next folds in turn. Which of its sessions gets which of those
    folds is up to the shuffle.
    """
    folds: dict[int, list[int]] = {}
    counter = 0
    for stratum, members in STRATA.items():
        folds[stratum] = sorted((counter + i) % k for i in range(len(members)))
        counter += len(members)
    return folds


def _validation_fold(folds: Sequence[tuple[list[str], list[str]]]) -> dict[str, int]:
    """Session id -> the index of the fold that validates it."""
    return {sid: i for i, (_train, validation) in enumerate(folds) for sid in validation}


def test_the_fold_table_holds_the_counts_its_strata_come_from(tmp_path: Path) -> None:
    # A check of the fixture, not of grouped_folds: STRATA and _dealt are worked out by hand.
    sessions = _load_folds(tmp_path, FOLD_ROWS)
    training = splits.training_sessions(sessions)
    assert training == TRAINING_IDS
    labels = dict(zip(sessions["session_id"], sessions["labels"], strict=True))
    counts = Counter(int(c) for sid in training for c in labels[sid])
    assert counts == {0: 10, 1: 1, 2: 3, 3: 3, 4: 2, 5: 4}
    assert sorted(sid for members in STRATA.values() for sid in members) == TRAINING_IDS
    assert _dealt(5) == {1: [0], 4: [1, 2], 2: [3, 4], 3: [0, 1], 5: [2, 3], 0: [0, 4]}


@pytest.mark.parametrize("k", [2, 3, 5, 11])
def test_every_training_session_is_validated_in_exactly_one_fold(tmp_path: Path, k: int) -> None:
    sessions = _load_folds(tmp_path, FOLD_ROWS)
    folds = splits.grouped_folds(sessions, k=k, seed=0)
    assert len(folds) == k
    validated = [sid for _train, validation in folds for sid in validation]
    assert sorted(validated) == TRAINING_IDS  # each once: none twice, none left out
    for train, validation in folds:
        assert set(train).isdisjoint(validation)
        assert sorted(train + validation) == TRAINING_IDS


def test_each_fold_is_a_pair_of_sorted_lists_of_ids(tmp_path: Path) -> None:
    sessions = _load_folds(tmp_path, FOLD_ROWS)
    folds = splits.grouped_folds(sessions, k=3, seed=0)
    assert isinstance(folds, list)
    for fold in folds:
        assert isinstance(fold, tuple)
        assert len(fold) == 2
        for ids in fold:
            assert isinstance(ids, list)  # not a set, a pandas Series or a numpy array
            assert all(isinstance(sid, str) for sid in ids)
            assert ids == sorted(ids)


def test_crossing_and_test_only_sessions_are_in_no_fold(tmp_path: Path) -> None:
    sessions = _load_folds(tmp_path, FOLD_ROWS)
    for train, validation in splits.grouped_folds(sessions, k=5, seed=0):
        assert {CROSSING[0], TEST_ONLY[0]}.isdisjoint(train + validation)  # data rule 3


@pytest.mark.parametrize(
    "bad_rows",
    [
        pytest.param([(BAD_ID, False, False, False, [0])], id="in-neither-split"),
        pytest.param([(BAD_ID, True, True, False, [0, 2])], id="in-both-not-marked-crossing"),
        pytest.param(
            [(BAD_ID, True, False, False, [0]), (BAD_ID, False, True, False, [0])],
            id="repeated-id",
        ),
    ],
)
def test_a_table_training_sessions_rejects_is_rejected(
    tmp_path: Path, bad_rows: list[FoldRow]
) -> None:
    # The first bad row goes before the others and any repeat after them.
    sessions = _load_folds(tmp_path, [bad_rows[0], *FOLD_ROWS, *bad_rows[1:]])
    with pytest.raises(ValueError, match=BAD_ID):
        splits.training_sessions(sessions)  # the fixture: training_sessions rejects it
    with pytest.raises(ValueError, match=BAD_ID):
        splits.grouped_folds(sessions, k=2, seed=0)


@pytest.mark.parametrize(
    "k", [-1, 0, 1, 12, 13], ids=["k=-1", "k=0", "k=1", "k=12-above-11", "k=13-every-row"]
)
def test_k_outside_2_to_the_number_of_training_sessions_is_an_error(tmp_path: Path, k: int) -> None:
    # 11 training sessions in 13 rows: k counts training sessions, not rows.
    sessions = _load_folds(tmp_path, FOLD_ROWS)
    with pytest.raises(ValueError, match=r"\bk\b"):  # the message names k
        splits.grouped_folds(sessions, k=k, seed=0)


def test_k_can_be_2_or_the_number_of_training_sessions(tmp_path: Path) -> None:
    sessions = _load_folds(tmp_path, FOLD_ROWS)
    assert len(splits.grouped_folds(sessions, k=2, seed=0)) == 2
    one_each = splits.grouped_folds(sessions, k=len(TRAINING_IDS), seed=0)
    assert [len(validation) for _train, validation in one_each] == [1] * len(TRAINING_IDS)


def test_a_table_without_training_sessions_allows_no_k(tmp_path: Path) -> None:
    sessions = _load_folds(tmp_path, [(*CROSSING, [0, 4]), (*TEST_ONLY, [0, 4])])
    assert splits.training_sessions(sessions) == []
    with pytest.raises(ValueError, match=r"\bk\b"):
        splits.grouped_folds(sessions, k=2, seed=0)


def test_a_training_session_without_labels_is_named(tmp_path: Path) -> None:
    sessions = _load_folds(tmp_path, [*FOLD_ROWS, (BAD_ID, True, False, False, [])])
    labels = dict(zip(sessions["session_id"], sessions["labels"], strict=True))
    assert len(labels[BAD_ID]) == 0  # still an empty list after the parquet round trip
    with pytest.raises(ValueError, match=BAD_ID):
        splits.grouped_folds(sessions, k=2, seed=0)


@pytest.mark.parametrize("k", [2, 3, 5])
def test_strata_are_dealt_rarest_first_with_one_counter(tmp_path: Path, k: int) -> None:
    # Each stratum takes the next folds in turn, from where the one before stopped. That pins
    # down each session's stratum (its rarest class, ties to the lower class, counted over
    # training sessions only), the order of the strata and the single counter. Ten seeds, so
    # a wrong stratum can't pass by the luck of one shuffle.
    sessions = _load_folds(tmp_path, FOLD_ROWS)
    expected = _dealt(k)
    for seed in range(10):
        fold_of = _validation_fold(splits.grouped_folds(sessions, k=k, seed=seed))
        for stratum, members in STRATA.items():
            assert sorted(fold_of[sid] for sid in members) == expected[stratum], (seed, stratum)


@pytest.mark.parametrize("k", [2, 3, 4, 5, 11])
def test_fold_sizes_and_each_strata_counts_differ_by_at_most_one(tmp_path: Path, k: int) -> None:
    sessions = _load_folds(tmp_path, FOLD_ROWS)
    for seed in range(10):
        folds = splits.grouped_folds(sessions, k=k, seed=seed)
        sizes = [len(validation) for _train, validation in folds]
        assert max(sizes) - min(sizes) <= 1, seed
        for stratum, members in STRATA.items():
            counts = [len(members.intersection(validation)) for _train, validation in folds]
            assert max(counts) - min(counts) <= 1, (seed, stratum)


@pytest.mark.parametrize("k", [2, 3, 5])
def test_a_class_with_one_training_session_is_not_trained_on_where_it_is_validated(
    tmp_path: Path, k: int
) -> None:
    # Documented behaviour, not a wish: burn through has one training session in this
    # release, so the one fold that validates it has no burn through to train on.
    sessions = _load_folds(tmp_path, FOLD_ROWS)
    labels = dict(zip(sessions["session_id"], sessions["labels"], strict=True))
    holders = {sid for sid in splits.training_sessions(sessions) if 1 in labels[sid]}
    assert holders == {BURN_THROUGH}  # the fixture: one training session has class 1
    for seed in range(5):
        folds = splits.grouped_folds(sessions, k=k, seed=seed)
        validating = [train for train, validation in folds if holders.intersection(validation)]
        assert len(validating) == 1, seed
        assert holders.isdisjoint(validating[0]), seed


def test_the_same_seed_gives_the_same_folds_whatever_the_row_order(tmp_path: Path) -> None:
    orders = [FOLD_ROWS, FOLD_ROWS[::-1], _shuffled(FOLD_ROWS, 1), _shuffled(FOLD_ROWS, 2)]
    assert len({tuple(row[0] for row in order) for order in orders}) == 4  # all different
    tables = [_load_folds(tmp_path, order) for order in orders]
    for seed in (0, 7):
        folds = [splits.grouped_folds(table, k=3, seed=seed) for table in tables]
        assert all(other == folds[0] for other in folds[1:]), seed


def test_the_same_seed_gives_the_same_folds_in_a_new_process(tmp_path: Path) -> None:
    # Python hashes strings differently in every process unless PYTHONHASHSEED is fixed, so
    # folds that leaned on set order would pass the test above and still change from one
    # run to the next.
    derived = _write(tmp_path, _fold_table(FOLD_ROWS))
    expected = splits.grouped_folds(splits.load_sessions(derived), k=3, seed=0)
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from weldpilot.data import splits\n"
        "print(splits.grouped_folds(splits.load_sessions(Path(sys.argv[1])), k=3, seed=0))\n"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code, str(derived)],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for hash_seed in ("1", "2")
    }
    assert outputs == {f"{expected}\n"}


def test_the_seed_changes_the_folds(tmp_path: Path) -> None:
    # Five strata hold two sessions each, so the shuffle has something to change.
    sessions = _load_folds(tmp_path, FOLD_ROWS)
    seen = {repr(splits.grouped_folds(sessions, k=3, seed=seed)) for seed in range(20)}
    assert len(seen) > 1


def test_k_defaults_to_5_and_seed_to_0(tmp_path: Path) -> None:
    sessions = _load_folds(tmp_path, FOLD_ROWS)
    folds = splits.grouped_folds(sessions)
    assert len(folds) == 5
    assert folds == splits.grouped_folds(sessions, k=5, seed=0)


def test_grouped_folds_read_the_table_the_index_writes(
    make_dataset: MakeDataset, tmp_path: Path
) -> None:
    # labels as `weldpilot data index` writes them: a parquet list column, which comes back
    # as numpy arrays, not lists. Three training sessions in tests/data/synth.py.
    derived = tmp_path / "derived"
    write_index(build_index(make_dataset()), derived, {})
    folds = splits.grouped_folds(splits.load_sessions(derived), k=3, seed=0)
    validated = sorted(sid for _train, validation in folds for sid in validation)
    assert validated == ["170815-133921", "170904-141730", "170913-152931"]
