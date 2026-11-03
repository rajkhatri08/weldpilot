"""Session splits: which sessions may train, and which are kept for the final test.

Every analysis gets its sessions from here (data rule 4). Crossing sessions,
with folders in both train/ and test/, never train (data rule 3).
"""

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
