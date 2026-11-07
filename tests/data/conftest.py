from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from tests.data.invented import Invented, build
from tests.data.synth import DEFAULT_FOLDERS, SynthFolder, write_dataset


@pytest.fixture
def make_dataset(tmp_path: Path) -> Callable[..., Path]:
    """Build the synthetic release at tmp_path/raw/tig-al5083 and return that folder."""

    def make(folders: Sequence[SynthFolder] = DEFAULT_FOLDERS) -> Path:
        return write_dataset(tmp_path / "raw" / "tig-al5083", folders)

    return make


@pytest.fixture
def invented(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Invented:
    """The release with invented session ids, indexed, with WELDPILOT_DATA and the working
    directory in tmp_path, so no command can touch the real data or reports."""
    monkeypatch.setenv("WELDPILOT_DATA", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return build(tmp_path)
