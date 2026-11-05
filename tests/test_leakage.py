"""The leak test: fails if any session or folder of the official test set could reach training.

It reads the real local index that `weldpilot data index` writes to <data root>/derived/,
never an image. So it's marked `data`, and it skips where there is no index, as in CI.

Failure messages name only session ids and folders, never labels: some training sessions
are marked blind until onset pass 2 (data/annotations/onset_marks.csv).
"""

from pathlib import Path

import pandas as pd
import pytest

from weldpilot.config import get_settings
from weldpilot.data import splits
from weldpilot.data.index import SEGMENTS_FILE, SESSIONS_FILE

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def derived() -> Path:
    """The real index folder, or a skip when this machine has none."""
    path = get_settings().data_root / "derived"
    if not (path / SESSIONS_FILE).is_file() or not (path / SEGMENTS_FILE).is_file():
        pytest.skip(f"no index in {path}: run `weldpilot data index` (CI has no dataset)")
    return path


@pytest.fixture(scope="module")
def sessions(derived: Path) -> pd.DataFrame:
    return splits.load_sessions(derived)


@pytest.fixture(scope="module")
def segments(derived: Path) -> pd.DataFrame:
    """One row per folder. Only these columns are read, so no label leaves the disk."""
    return pd.read_parquet(derived / SEGMENTS_FILE, columns=["folder", "session_id", "split"])


def test_no_session_is_both_a_training_and_a_test_session(sessions: pd.DataFrame) -> None:
    both = sorted(set(splits.training_sessions(sessions)) & set(splits.test_sessions(sessions)))
    assert not both, f"sessions in both training and test: {both}"  # data rule 3


def test_every_session_with_a_test_folder_is_a_test_session(
    sessions: pd.DataFrame, segments: pd.DataFrame
) -> None:
    # Checks the flags in sessions.parquet against the folders themselves, so a wrong in_test
    # flag can't hide a test session among the training sessions.
    with_test_folders = set(segments.loc[segments["split"] == "test", "session_id"])
    missed = sorted(with_test_folders - set(splits.test_sessions(sessions)))
    assert not missed, f"sessions with folders in test/ that test_sessions misses: {missed}"


def test_no_folder_is_in_both_train_and_test(segments: pd.DataFrame) -> None:
    train = set(segments.loc[segments["split"] == "train", "folder"])
    test = set(segments.loc[segments["split"] == "test", "folder"])
    both = sorted(train & test)
    assert not both, f"folders in both train/ and test/: {both}"


@pytest.mark.parametrize("k", [2, 5])
def test_no_test_session_is_in_any_fold(sessions: pd.DataFrame, k: int) -> None:
    test_ids = set(splits.test_sessions(sessions))
    for fold, (train, validation) in enumerate(splits.grouped_folds(sessions, k=k, seed=0)):
        leaked = sorted(test_ids.intersection(train + validation))
        assert not leaked, f"fold {fold} of {k} holds test sessions: {leaked}"
