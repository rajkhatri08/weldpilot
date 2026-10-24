"""Composition of the release for the datasheet: markdown tables by split and by class.

Counts are exact for this release (a census, not an estimate), so they carry no interval.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from weldpilot.data.al5083 import ASSUMED_FPS, CLASS_NAMES, SPLITS
from weldpilot.data.index import session_status

MIN_TRAIN_ONLY_SESSIONS = 2
"""With fewer, no session of a class can be held out for validation while another trains.
Crossing sessions don't count: data rule 3 keeps them out of training and model selection."""

FLAG = f"fewer than {MIN_TRAIN_ONLY_SESSIONS} train-only sessions"


@dataclass(frozen=True)
class ClassCounts:
    label: int
    frames: int
    folders: int
    train_only: int  # sessions that hold the class, by the session's split status
    test_only: int
    crossing: int

    @property
    def flagged(self) -> bool:
        return self.train_only < MIN_TRAIN_ONLY_SESSIONS


def statuses(sessions: pd.DataFrame) -> dict[str, str]:
    """Session id -> "train-only", "test-only" or "crossing"."""
    ids = sessions["session_id"].tolist()
    in_train = sessions["in_train"].tolist()
    in_test = sessions["in_test"].tolist()
    return {
        sid: session_status(bool(train), bool(test))
        for sid, train, test in zip(ids, in_train, in_test, strict=True)
    }


def class_counts(segments: pd.DataFrame, sessions: pd.DataFrame) -> list[ClassCounts]:
    """Frames, folders and sessions for every class 0-5; a class that is absent counts 0."""
    status = statuses(sessions)
    counts: list[ClassCounts] = []
    for label in CLASS_NAMES:
        mine = segments[segments["label"] == label]
        per_status = Counter(status[sid] for sid in set(mine["session_id"].tolist()))
        counts.append(
            ClassCounts(
                label=label,
                frames=sum(mine["n_frames"].tolist()),
                folders=len(mine),
                train_only=per_status["train-only"],
                test_only=per_status["test-only"],
                crossing=per_status["crossing"],
            )
        )
    return counts


def markdown_table(
    header: Sequence[str], rows: Sequence[Sequence[str]], numeric: Sequence[bool]
) -> str:
    """A GitHub markdown table with numeric columns right-aligned."""
    rule = ["---:" if n else ":---" for n in numeric]
    return "\n".join("| " + " | ".join(cells) + " |" for cells in [header, rule, *rows])


def split_table(segments: pd.DataFrame) -> str:
    """Folders, sessions, frames and footage for train, test and both."""
    rows: list[list[str]] = []
    for name in (*SPLITS, "all"):
        part = segments if name == "all" else segments[segments["split"] == name]
        frames = sum(part["n_frames"].tolist())
        sessions = part["session_id"].nunique()
        footage = frames / ASSUMED_FPS
        rows.append([name, f"{len(part):,}", f"{sessions:,}", f"{frames:,}", f"{footage:,.1f}"])
    header = ["Split", "Folders", "Sessions", "Frames", "Footage (s)"]
    return markdown_table(header, rows, [False, True, True, True, True])


def class_table(counts: Sequence[ClassCounts]) -> str:
    """One row per class, flagged when it has too few train-only sessions."""
    rows = [
        [
            f"{c.label} {CLASS_NAMES[c.label]}",
            f"{c.frames:,}",
            f"{c.folders:,}",
            str(c.train_only),
            str(c.test_only),
            str(c.crossing),
            FLAG if c.flagged else "",
        ]
        for c in counts
    ]
    header = [
        "Class",
        "Frames",
        "Folders",
        "Train-only sessions",
        "Test-only sessions",
        "Crossing sessions",
        "Flag",
    ]
    return markdown_table(header, rows, [False, True, True, True, True, True, False])


def composition_markdown(segments: pd.DataFrame, sessions: pd.DataFrame) -> str:
    """Both tables with their notes, and where the numbers come from (data rule 8)."""
    status = statuses(sessions)
    n = Counter(status.values())
    crossing = sorted(sid for sid, s in status.items() if s == "crossing")
    counts = class_counts(segments, sessions)
    flagged = [f"{c.label} {CLASS_NAMES[c.label]}" for c in counts if c.flagged]
    attrs = segments.attrs
    return "\n".join(
        [
            "### By split",
            "",
            split_table(segments),
            "",
            f"Sessions: {n['train-only']} train-only, {n['test-only']} test-only and "
            f"{n['crossing']} crossing (folders in both train/ and test/), {len(status)} in "
            "all. A crossing session counts in both split rows. Footage is the frames present "
            f"divided by {ASSUMED_FPS:g} fps, an assumption (the camera rate reported for the "
            "SS304 set); missing frame numbers add no footage.",
            "",
            "Crossing sessions, never used for training, calibration or model selection "
            f"(data rule 3): {', '.join(crossing) or 'none'}.",
            "",
            "### By class",
            "",
            class_table(counts),
            "",
            "A session can hold several classes, so the session columns don't add up to "
            f"{len(status)}. Flagged ({FLAG}): {', '.join(flagged) or 'none'}. Such a class "
            "can't have one session held out for validation while another trains; crossing "
            "sessions count for neither (data rule 3).",
            "",
            "Source: `weldpilot data composition`, from the index built at git "
            f"{attrs.get('git_sha', 'unknown')} on the dataset with fingerprint "
            f"{attrs.get('dataset_fingerprint', 'unknown')} (checked against "
            "configs/dataset.yaml). Counts are exact for this release, not estimates.",
        ]
    )
