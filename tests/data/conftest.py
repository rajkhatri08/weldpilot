from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from tests.data.synth import DEFAULT_FOLDERS, SynthFolder, write_dataset


@pytest.fixture
def make_dataset(tmp_path: Path) -> Callable[..., Path]:
    """Build the synthetic release at tmp_path/raw/tig-al5083 and return that folder."""

    def make(folders: Sequence[SynthFolder] = DEFAULT_FOLDERS) -> Path:
        return write_dataset(tmp_path / "raw" / "tig-al5083", folders)

    return make
