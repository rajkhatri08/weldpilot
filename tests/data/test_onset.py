"""Tests for `weldpilot.data.onset`: which sessions qualify for blind onset marking, how they
are picked, and the marks file. Tiny synthetic tables; never the real data."""

import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pytest

from weldpilot.data import onset
from weldpilot.data.index import build_index, write_index
from weldpilot.data.splits import load_sessions

MakeDataset = Callable[..., Path]

# Thirteen made-up candidates in four fault classes; s12 has two.
CANDIDATES: dict[str, frozenset[int]] = {
    **{f"s{i:02d}": frozenset({1 + i % 4}) for i in range(12)},
    "s12": frozenset({1, 3}),
}

# session_id, in_train, in_test, crosses_split, n_good_to_fault, labels
Row = tuple[str, bool, bool, bool, int, list[int]]
COLUMNS = ["session_id", "in_train", "in_test", "crosses_split", "n_good_to_fault", "labels"]
TRAIN_WITH_CHANGES: Row = ("170904-141730", True, False, False, 2, [0, 2, 4])


def _table(rows: Sequence[Row]) -> pd.DataFrame:
    """Rows as a sessions table with the columns the onset code reads."""
    return pd.DataFrame(list(rows), columns=COLUMNS)


def test_the_same_seed_picks_the_same_sessions() -> None:
    first = onset.pick_sessions(CANDIDATES, 5, seed=0)
    assert onset.pick_sessions(CANDIDATES, 5, seed=0) == first
    # Nor does the order the candidates arrive in matter.
    reordered = dict(reversed(list(CANDIDATES.items())))
    assert onset.pick_sessions(reordered, 5, seed=0) == first


def test_the_same_seed_picks_the_same_sessions_in_a_new_process() -> None:
    # Python hashes strings differently in every process unless PYTHONHASHSEED is fixed, so
    # a pick that leaned on set order would pass the test above and still change from one
    # `weldpilot onset pick` to the next.
    code = (
        "from weldpilot.data.onset import pick_sessions\n"
        f"print(pick_sessions({CANDIDATES!r}, 5, seed=0))\n"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for hash_seed in ("1", "2")
    }
    assert outputs == {f"{onset.pick_sessions(CANDIDATES, 5, seed=0)}\n"}


def test_the_seed_changes_the_pick() -> None:
    picks = {tuple(onset.pick_sessions(CANDIDATES, 5, seed=seed)) for seed in range(20)}
    assert len(picks) > 1


def test_picks_spread_across_fault_classes_and_come_back_sorted() -> None:
    # Class 1 has ten sessions, classes 2 and 3 one each. A plain random draw of three would
    # mostly be class 1; here each class gets a turn before any gets a second.
    candidates = {
        **{f"a{i}": frozenset({1}) for i in range(10)},
        "b": frozenset({2}),
        "c": frozenset({3}),
    }
    for seed in range(20):
        picked = onset.pick_sessions(candidates, 3, seed)
        assert len(picked) == 3
        assert {"b", "c"} <= set(picked)
        # Sorted by id, so the order can't tell which class a session was drawn for.
        assert picked == sorted(picked)


def test_a_small_pool_is_picked_whole_with_each_session_once() -> None:
    candidates = {
        "d": frozenset({2}),
        "a": frozenset({1, 2}),  # in two pools, picked once
        "c": frozenset({1}),
        "b": frozenset({3}),
    }
    assert onset.pick_sessions(candidates, 10, seed=0) == ["a", "b", "c", "d"]


def test_only_training_sessions_with_a_good_to_fault_change_qualify() -> None:
    sessions = _table(
        [
            TRAIN_WITH_CHANGES,
            ("170815-133921", True, False, False, 0, [3]),  # train-only, no change
            ("170904-150144", True, True, True, 1, [0, 1]),  # crossing: never trains
            ("170906-153326", False, True, False, 1, [0, 5]),  # test-only
        ]
    )
    # The fault classes are the session's classes other than good weld (0).
    assert onset.onset_candidates(sessions) == {"170904-141730": frozenset({2, 4})}


@pytest.mark.parametrize("column", ["n_good_to_fault", "labels"])
def test_a_missing_column_is_named(column: str) -> None:
    sessions = _table([TRAIN_WITH_CHANGES]).drop(columns=[column])
    with pytest.raises(ValueError, match=column):
        onset.onset_candidates(sessions)


def test_a_change_with_no_fault_class_is_an_error() -> None:
    # The index can't write this; a hand-edited table could. Kept silently, the session
    # would count as qualified but sit in no class's pool, so it could never be picked.
    sessions = _table([("170904-141730", True, False, False, 1, [0])])
    with pytest.raises(ValueError, match="170904-141730"):
        onset.onset_candidates(sessions)


def test_candidates_come_from_the_table_the_index_writes(
    make_dataset: MakeDataset, tmp_path: Path
) -> None:
    # Written by the code behind `weldpilot data index`, from tests/data/synth.py. Only
    # 170904-141730 is a training session with a good-to-fault change; the crossing
    # 170904-150144 has one too, but never trains.
    derived = tmp_path / "derived"
    write_index(build_index(make_dataset()), derived, {})
    candidates = onset.onset_candidates(load_sessions(derived))
    assert candidates == {"170904-141730": frozenset({2, 4})}


def test_a_new_marks_file_has_the_header_and_blank_cells() -> None:
    text = onset.marks_csv(["170904-141730", "170913-152931"])
    assert text == (
        "session_id,pass,onset_frame,marked_on,note\n170904-141730,,,,\n170913-152931,,,,\n"
    )
    assert onset.marked_sessions(text, "marks.csv") == ["170904-141730", "170913-152931"]


def test_marked_sessions_are_distinct_and_sorted_and_skip_blank_rows() -> None:
    text = (
        "session_id,pass,onset_frame,marked_on,note\n"
        "170913-152931,1,120,2026-11-04,\n"
        "170904-141730,1,85,2026-11-04,after a gap\n"
        "170913-152931,2,125,2026-11-11,\n"  # a second pass: still one sheet
        ",,,,\n"  # an empty row a spreadsheet app added
    )
    assert onset.marked_sessions(text, "marks.csv") == ["170904-141730", "170913-152931"]


def test_marks_without_a_session_id_column_are_an_error() -> None:
    with pytest.raises(ValueError, match="session_id"):
        onset.marked_sessions("session,pass\n170904-141730,1\n", "marks.csv")
