"""Blind onset marking: which sessions Raj marks by eye, and the file his marks go in.

The labels choose and spread the sample here and go no further: nothing in this module
returns a session's classes to the person marking, and the sample comes back sorted by id.
"""

import csv
import io
import random
from collections.abc import Collection, Mapping, Sequence

import pandas as pd

from weldpilot.data import splits
from weldpilot.data.al5083 import GOOD_WELD

MARKS_FILE_NAME = "onset_marks.csv"
"""In `<data root>/annotations/`, which git keeps: it holds marks made by hand."""

MARKS_COLUMNS = ("session_id", "pass", "onset_frame", "marked_on", "note")

_CANDIDATE_COLUMNS = ("n_good_to_fault", "labels")


def onset_candidates(sessions: pd.DataFrame) -> dict[str, frozenset[int]]:
    """Training sessions with at least one good-to-fault change, each with its fault classes.

    Training sessions come from `splits.training_sessions` (data rules 3 and 4), so crossing
    and test-only sessions never qualify. A session's fault classes are its classes other
    than good weld. Raises ValueError naming any missing column, or a qualifying session
    that lists no fault class.
    """
    missing = [name for name in _CANDIDATE_COLUMNS if name not in sessions.columns]
    if missing:
        raise ValueError(f"the session table is missing columns: {', '.join(missing)}")
    training = set(splits.training_sessions(sessions))
    candidates: dict[str, frozenset[int]] = {}
    for session_id, n_changes, labels in zip(
        sessions["session_id"], sessions["n_good_to_fault"], sessions["labels"], strict=True
    ):
        if str(session_id) not in training or n_changes < 1:
            continue
        faults = frozenset(int(c) for c in labels) - {GOOD_WELD}
        if not faults:
            raise ValueError(f"session {session_id} has a good-to-fault change but no fault class")
        candidates[str(session_id)] = faults
    return candidates


def pick_sessions(candidates: Mapping[str, Collection[int]], n: int, seed: int) -> list[str]:
    """Up to `n` session ids drawn at random with `seed`, spread across fault classes.

    Each fault class has a pool of its sessions, shuffled. Classes take turns, in an order
    shuffled every round, and each turn takes the next session of that pool not picked yet,
    so every class gets a pick before any gets a second while it has sessions left. A
    session with two fault classes sits in both pools and is picked once.

    Pools are sorted before they are shuffled, so the pick depends only on the seed and the
    candidates: not on the order they arrive in, nor on Python's per-process string hashing.
    The ids come back sorted, so their order can't tell which class a session was drawn for.
    """
    rng = random.Random(seed)
    classes = sorted({fault for faults in candidates.values() for fault in faults})
    pools: dict[int, list[str]] = {}
    for fault in classes:
        pool = sorted(sid for sid, faults in candidates.items() if fault in faults)
        rng.shuffle(pool)
        pools[fault] = pool
    picked: set[str] = set()
    while len(picked) < n:
        turns = [fault for fault in classes if any(s not in picked for s in pools[fault])]
        if not turns:
            break
        rng.shuffle(turns)
        for fault in turns:
            if len(picked) == n:
                break
            # None when another class just took this pool's last session in this round.
            chosen = next((s for s in pools[fault] if s not in picked), None)
            if chosen is not None:
                picked.add(chosen)
    return sorted(picked)


def marks_csv(session_ids: Sequence[str]) -> str:
    """A new marks file: the header, then one row per session with only session_id filled."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(MARKS_COLUMNS)
    blanks = [""] * (len(MARKS_COLUMNS) - 1)
    writer.writerows([session_id, *blanks] for session_id in session_ids)
    return out.getvalue()


def marked_sessions(text: str, source: str) -> list[str]:
    """The distinct session ids in a marks file, sorted.

    A session marked in two passes appears once. Rows whose cells are all blank, such as one
    a spreadsheet app adds, are skipped; a row with other cells but no id gives "". Raises
    ValueError if the file has no session_id column.
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "session_id" not in reader.fieldnames:
        raise ValueError(f"{source} has no session_id column")
    return sorted(
        {
            row["session_id"] or ""
            for row in reader
            if any(str(cell).strip() for cell in row.values() if cell)
        }
    )
