"""Spec tests for `weldpilot.data.splits`, written before the module.

Every analysis gets its sessions from these functions (data rule 4), so they are where data
rule 3 is enforced: a crossing session is never a training session. Each test writes a tiny
sessions.parquet to tmp_path; none reads real data.
"""

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
