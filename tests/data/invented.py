"""A tiny synthetic release whose session ids are invented: year 99, round times.

The default folders in synth.py copy real sessions, some of them marked blind until onset
pass 2, so the tests added while that marking runs use these folders instead. The release
is indexed by the real `weldpilot data verify --record` and `data index` commands, so its
tables carry the provenance (`git_sha`, `dataset_fingerprint`) the commands under test read.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from tests.data.synth import SynthFolder, write_dataset
from weldpilot.cli import app
from weldpilot.data.al5083 import DATASET_DIR
from weldpilot.data.fingerprint import parse_record
from weldpilot.data.index import FRAMES_FILE
from weldpilot.data.onset import MARKS_FILE_NAME, marks_csv

SESSION_A = "990101-100000"  # training: good weld, then burn through
SESSION_B = "990101-110000"  # training: good weld, then contamination
SESSION_C = "990101-120000"  # test only

FOLDERS = (
    SynthFolder("train", f"{SESSION_A}-Al 2mm-part1", 0, (0, 1, 2, 3)),
    SynthFolder("train", f"{SESSION_A}-Al 2mm-part2", 1, (4, 5, 6)),
    SynthFolder("train", f"{SESSION_B}-Al 2mm-part1", 0, (0, 1, 2)),
    SynthFolder("train", f"{SESSION_B}-Al 2mm-part2", 2, (3, 4)),
    SynthFolder("test", f"{SESSION_C}-Al 2mm", 3, (0, 1)),
)


@dataclass(frozen=True)
class Invented:
    """The invented release, indexed."""

    root: Path  # the data root (WELDPILOT_DATA), which is also the working directory
    release: Path  # <root>/raw/tig-al5083
    derived: Path  # <root>/derived: the index tables
    fingerprint: str  # the dataset fingerprint that the index recorded


def build(root: Path) -> Invented:
    """Write the release under `root`, record its fingerprint and index it.

    Expects WELDPILOT_DATA and the working directory to be `root` already, so that the
    commands' relative defaults (configs/, reports/) land there too.
    """
    release = write_dataset(root / "raw" / DATASET_DIR, FOLDERS)
    runner = CliRunner()
    for args in (["data", "verify", "--record"], ["data", "index"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
    record = parse_record((root / "configs" / "dataset.yaml").read_text(encoding="utf-8"))
    return Invented(root, release, root / "derived", record.fingerprint)


def write_marks(invented: Invented, session_ids: Sequence[str]) -> Path:
    """A new onset_marks.csv listing `session_ids`, as `weldpilot onset pick` starts it."""
    path = invented.root / "annotations" / MARKS_FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(marks_csv(session_ids), encoding="utf-8")
    return path


def point_into_test(invented: Invented, session_id: str) -> None:
    """Give every frame of `session_id` the path of a test/ frame while its split still
    says train: the disagreement an image guard must refuse.

    The files exist, so hashing them still works; only drawing them would show a test frame
    (data rule 2). The table keeps its provenance in `attrs`.
    """
    path = invented.derived / FRAMES_FILE
    frames = pd.read_parquet(path)
    test_paths = frames.loc[frames["split"] == "test", "rel_path"].tolist()
    rows = frames.index[frames["session_id"] == session_id]
    frames.loc[rows, "rel_path"] = [test_paths[n % len(test_paths)] for n in range(len(rows))]
    frames.to_parquet(path, index=False)
