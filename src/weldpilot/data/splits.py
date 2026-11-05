"""Session splits: which sessions may train, and which are kept for the final test.

Every analysis gets its sessions from here (data rule 4). Crossing sessions,
with folders in both train/ and test/, never train (data rule 3). Cross-validation
folds come from here too, and never split a session.
"""

import random
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from weldpilot.data.index import SESSIONS_FILE

REQUIRED_COLUMNS = ("session_id", "in_train", "in_test", "crosses_split")


def load_sessions(derived_dir: Path) -> pd.DataFrame:
    """Read sessions.parquet from the index in `derived_dir`.

    Raises FileNotFoundError if it's missing, and ValueError naming every
    required column the table lacks.
    """
    path = derived_dir / SESSIONS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `weldpilot data index` first")
    sessions = pd.read_parquet(path)
    missing = [name for name in REQUIRED_COLUMNS if name not in sessions.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return sessions  # untouched: converting types here would hide empty flags


def _rows(sessions: pd.DataFrame) -> Iterator[tuple[Any, Any, Any, Any]]:
    """(session_id, in_train, in_test, crosses_split) for every row, in table order."""
    return zip(
        sessions["session_id"],
        sessions["in_train"],
        sessions["in_test"],
        sessions["crosses_split"],
        strict=True,
    )


def _check_sessions(sessions: pd.DataFrame) -> None:
    """Raise ValueError, naming the session, if any row can't be trusted.

    The index writes one row per session, with every flag filled in and
    crosses_split equal to (in_train and in_test). A table that breaks this was
    edited or built wrongly, so we stop instead of guessing.
    """
    seen: set[str] = set()
    for position, (session_id, in_train, in_test, crosses) in enumerate(_rows(sessions)):
        if pd.isna(session_id):
            raise ValueError(f"row {position} of the session table has no session_id")
        if session_id in seen:
            raise ValueError(f"session {session_id} appears more than once")
        seen.add(session_id)
        if pd.isna(in_train) or pd.isna(in_test) or pd.isna(crosses):
            raise ValueError(f"session {session_id} has an empty split flag")
        if not in_train and not in_test:
            raise ValueError(f"session {session_id} is in neither train/ nor test/")
        if bool(crosses) != (bool(in_train) and bool(in_test)):
            raise ValueError(
                f"session {session_id}: crosses_split is {bool(crosses)}, but in_train is "
                f"{bool(in_train)} and in_test is {bool(in_test)}"
            )


def training_sessions(sessions: pd.DataFrame) -> list[str]:
    """Sorted ids of the sessions that are in train/ and not in test/."""
    _check_sessions(sessions)
    return sorted(
        str(session_id)
        for session_id, in_train, in_test, _crosses in _rows(sessions)
        if in_train and not in_test
    )


def test_sessions(sessions: pd.DataFrame) -> list[str]:
    """Sorted ids of every session with folders in test/, crossing ones included."""
    _check_sessions(sessions)
    return sorted(
        str(session_id) for session_id, _in_train, in_test, _crosses in _rows(sessions) if in_test
    )


def _training_classes(sessions: pd.DataFrame, training: list[str]) -> dict[str, list[int]]:
    """Each training session's classes, from the labels column the index writes.

    Raises ValueError if the column is missing, or naming a training session that lists
    no class.
    """
    if "labels" not in sessions.columns:
        raise ValueError("the session table has no labels column")
    wanted = set(training)
    classes: dict[str, list[int]] = {}
    for session_id, labels in zip(sessions["session_id"], sessions["labels"], strict=True):
        if session_id in wanted:
            classes[str(session_id)] = sorted({int(c) for c in ([] if labels is None else labels)})
    for session_id in training:
        if not classes[session_id]:
            raise ValueError(f"session {session_id} has no classes in its labels column")
    return classes


def grouped_folds(
    sessions: pd.DataFrame, k: int = 5, seed: int = 0
) -> list[tuple[list[str], list[str]]]:
    """k cross-validation folds over the training sessions: (training ids, validation ids).

    Every training session is validated in exactly one fold, and a session is never split
    between training and validation. Folds are stratified by each session's rarest class:
    of its classes, the one the fewest training sessions have, ties going to the lower
    class. The strata are dealt round-robin into the folds, rarest first, with one counter,
    so the fold sizes, and each stratum's count per fold, differ by at most one. The seed
    only shuffles the sessions within each stratum.

    A class that only one training session has is never trained on in the fold that
    validates it (burn through, in this release).

    Raises ValueError if k is below 2 or above the number of training sessions, or naming
    a training session that lists no class.
    """
    training = training_sessions(sessions)  # checks every row first (data rules 3 and 4)
    if not 2 <= k <= len(training):
        raise ValueError(
            f"k is {k}, but it must be between 2 and the number of training sessions "
            f"({len(training)})"
        )
    classes = _training_classes(sessions, training)
    holders = Counter(c for session_classes in classes.values() for c in session_classes)

    def rarity(c: int) -> tuple[int, int]:
        """Fewer training sessions first, then the lower class number."""
        return holders[c], c

    strata: dict[int, list[str]] = {}
    for session_id in training:  # sorted, so every stratum starts sorted by id
        strata.setdefault(min(classes[session_id], key=rarity), []).append(session_id)

    rng = random.Random(seed)
    dealing_order: list[str] = []
    for stratum in sorted(strata, key=rarity):
        members = strata[stratum]
        rng.shuffle(members)
        dealing_order.extend(members)

    validation: list[list[str]] = [[] for _ in range(k)]
    for position, session_id in enumerate(dealing_order):
        validation[position % k].append(session_id)  # one counter across every stratum

    folds: list[tuple[list[str], list[str]]] = []
    for held_out in validation:
        excluded = set(held_out)
        folds.append(([sid for sid in training if sid not in excluded], sorted(held_out)))
    return folds
