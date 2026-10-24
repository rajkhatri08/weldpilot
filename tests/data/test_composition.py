from collections.abc import Callable
from pathlib import Path

import pandas as pd

from tests.data.synth import SynthFolder
from weldpilot.data.composition import (
    FLAG,
    class_counts,
    composition_markdown,
    split_table,
)
from weldpilot.data.index import build_index, segments_dataframe, sessions_dataframe

MakeDataset = Callable[..., Path]


def _tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = build_index(root)
    return segments_dataframe(index.segments), sessions_dataframe(index.sessions)


def test_split_table_counts_folders_sessions_frames_and_footage(make_dataset: MakeDataset) -> None:
    segments, _ = _tables(make_dataset())
    rows = split_table(segments).splitlines()
    assert rows[0] == "| Split | Folders | Sessions | Frames | Footage (s) |"
    # A crossing session counts in both split rows, and once in "all".
    assert rows[2:] == [
        "| train | 7 | 4 | 20 | 0.4 |",
        "| test | 2 | 2 | 4 | 0.1 |",
        "| all | 9 | 5 | 24 | 0.4 |",
    ]


def test_a_class_needs_two_train_only_sessions(make_dataset: MakeDataset) -> None:
    root = make_dataset(
        [
            SynthFolder("train", "170904-112347-Al 2mm", 0, (0, 1)),  # class 0: train-only
            SynthFolder("train", "170904-115503-Al 2mm", 0, (0, 1)),  # class 0: train-only
            SynthFolder("train", "170905-110711-Al 2mm", 2, (0, 1)),  # class 2: train-only
            # Class 2 again, in the train/ folder of a crossing session: that doesn't count.
            SynthFolder("train", "170904-150144-Al 2mm-part1", 2, (0, 1)),
            SynthFolder("test", "170904-150144-Al 2mm-part2", 1, (2, 3)),  # class 1: crossing
        ]
    )
    segments, sessions = _tables(root)
    counts = {c.label: c for c in class_counts(segments, sessions)}
    assert (counts[0].train_only, counts[0].flagged) == (2, False)
    assert (counts[2].train_only, counts[2].crossing, counts[2].flagged) == (1, 1, True)
    assert (counts[1].train_only, counts[1].crossing, counts[1].flagged) == (0, 1, True)
    assert [c.label for c in counts.values() if c.flagged] == [1, 2, 3, 4, 5]  # 3-5 absent

    markdown = composition_markdown(segments, sessions)
    assert f"| 2 contamination | 4 | 2 | 1 | 0 | 1 | {FLAG} |" in markdown
    assert "| 0 good weld | 4 | 2 | 2 | 0 | 0 |  |" in markdown
    assert f"Flagged ({FLAG}): 1 burn through, 2 contamination, 3 lack of fusion," in markdown


def test_markdown_lists_crossing_sessions_and_states_the_assumption(
    make_dataset: MakeDataset,
) -> None:
    segments, sessions = _tables(make_dataset())
    markdown = composition_markdown(segments, sessions)
    assert "3 train-only, 1 test-only and 1 crossing" in markdown
    assert "(data rule 3): 170904-150144." in markdown
    assert "divided by 55 fps, an assumption" in markdown
    assert "git unknown" in markdown  # no provenance attrs on tables built in memory
