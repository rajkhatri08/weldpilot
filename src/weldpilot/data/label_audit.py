"""Label audit: the operator's labels of the training sessions, in time order.

A session's timeline is its frames from all its parts in frame-number order. Missing numbers
stay gaps and are never filled. The timeline gives the label changes, the gaps and the
operator's onset, which `weldpilot audit labels` compares with the onsets marked by eye in
onset_marks.csv. These functions are pure; the command does the reading and writing.
"""

import csv
import io
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import pandas as pd

from weldpilot.data.al5083 import ASSUMED_FPS, CLASS_NAMES, GOOD_WELD
from weldpilot.data.composition import markdown_table

LONG_GAP_FRAMES = 10
"""A gap is long when more than this many frame numbers are missing."""

FRAME_COLUMNS = ("session_id", "frame_number", "label")
"""All the timelines need from frames.parquet."""

MARK_COLUMNS = ("session_id", "pass", "onset_frame")
"""The columns of onset_marks.csv that the comparison reads (see `onset.MARKS_COLUMNS`)."""

SUMMARY_COLUMNS = (
    "session_id",
    "n_frames",
    "labels_in_time_order",
    "n_label_changes",
    "long_gaps",
)

_WHOLE_NUMBER = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class Timeline:
    """One session's frames from all its parts, in frame-number order."""

    session_id: str
    frame_numbers: tuple[int, ...]  # ascending, each once
    labels: tuple[int, ...]  # the label of each frame, in the same order

    @property
    def n_frames(self) -> int:
        return len(self.frame_numbers)

    @property
    def first_frame(self) -> int:
        return self.frame_numbers[0]

    @property
    def last_frame(self) -> int:
        return self.frame_numbers[-1]


@dataclass(frozen=True)
class Segment:
    """Frames `first` to `last` of a session: all present, all with one label."""

    label: int
    first: int
    last: int

    @property
    def n_frames(self) -> int:
        return self.last - self.first + 1


@dataclass(frozen=True)
class Gap:
    """Frame numbers `first` to `last`, missing between two present frames of a session."""

    first: int
    last: int

    @property
    def n_missing(self) -> int:
        return self.last - self.first + 1


@dataclass(frozen=True)
class Change:
    """A label change between two consecutive present frames of a session."""

    before: int  # label of the earlier frame
    after: int  # label of the later frame
    last_before: int  # frame number of the earlier frame
    first_after: int  # frame number of the later frame

    @property
    def n_missing(self) -> int:
        """Frame numbers missing between the two frames: the change happened in one of them."""
        return self.first_after - self.last_before - 1

    @property
    def good_to_fault(self) -> bool:
        return self.before == GOOD_WELD and self.after != GOOD_WELD


def timelines(frames: pd.DataFrame, session_ids: Sequence[str]) -> list[Timeline]:
    """One timeline per session in `session_ids`, sorted by id. Other sessions are ignored.

    `frames` needs the columns in FRAME_COLUMNS, as in frames.parquet. Raises ValueError
    naming any missing column, or naming the session that has no frames, a frame number
    twice, or a label that is not a class 0-5.
    """
    missing = [name for name in FRAME_COLUMNS if name not in frames.columns]
    if missing:
        raise ValueError(f"the frames table is missing columns: {', '.join(missing)}")
    wanted: dict[str, list[tuple[int, int]]] = {str(sid): [] for sid in session_ids}
    for session_id, number, label in zip(
        frames["session_id"], frames["frame_number"], frames["label"], strict=True
    ):
        rows = wanted.get(str(session_id))
        if rows is not None:
            rows.append((int(number), int(label)))
    found: list[Timeline] = []
    for session_id in sorted(wanted):
        rows = sorted(wanted[session_id])
        if not rows:
            raise ValueError(f"session {session_id} has no frames in the frames table")
        numbers = tuple(number for number, _label in rows)
        labels = tuple(label for _number, label in rows)
        repeated = sorted({a for a, b in pairwise(numbers) if a == b})
        if repeated:
            raise ValueError(f"session {session_id} has frame {repeated[0]} more than once")
        if any(label not in CLASS_NAMES for label in labels):
            raise ValueError(f"session {session_id} has a label that is not a class 0-5")
        found.append(Timeline(session_id, numbers, labels))
    return found


def segments(timeline: Timeline) -> list[Segment]:
    """The timeline cut wherever the label changes or frames are missing, in time order.

    Drawn as bars, they leave every gap blank.
    """
    numbers, labels = timeline.frame_numbers, timeline.labels
    found: list[Segment] = []
    start = 0
    for i in range(1, len(numbers) + 1):
        if i == len(numbers) or numbers[i] != numbers[i - 1] + 1 or labels[i] != labels[i - 1]:
            found.append(Segment(labels[start], numbers[start], numbers[i - 1]))
            start = i
    return found


def gaps(timeline: Timeline) -> list[Gap]:
    """Runs of frame numbers missing between the session's first and last frames.

    Numbers before the first frame or after the last one are not gaps.
    """
    return [Gap(a + 1, b - 1) for a, b in pairwise(timeline.frame_numbers) if b > a + 1]


def changes(timeline: Timeline) -> list[Change]:
    """Every label change, in time order. Frames missing between two frames make no change."""
    frames = zip(timeline.frame_numbers, timeline.labels, strict=True)
    return [
        Change(before, after, last_before, first_after)
        for (last_before, before), (first_after, after) in pairwise(frames)
        if before != after
    ]


def label_sequence(timeline: Timeline) -> list[int]:
    """The labels in time order with repeats in a row collapsed: 0 0 2 2 0 gives 0 2 0."""
    labels = timeline.labels
    return [label for i, label in enumerate(labels) if i == 0 or label != labels[i - 1]]


def operator_onset(timeline: Timeline) -> Change | None:
    """The session's first good-to-fault change, or None if it has none.

    Its `first_after` is the operator's onset: the first frame of the first fault that
    follows good weld. A fault before any good weld is not an onset.
    """
    return next((change for change in changes(timeline) if change.good_to_fault), None)


@dataclass(frozen=True)
class Mark:
    """One row of onset_marks.csv."""

    line: int  # its line in the file; the header is line 1
    session_id: str
    pass_number: int
    onset_frame: int | None  # None: not marked yet


def _pass_number(value: str | None, source: str, line: int) -> int:
    text = (value or "").strip()
    if not text:
        return 1  # a blank pass is pass 1
    if _WHOLE_NUMBER.fullmatch(text) is None or int(text) < 1:
        raise ValueError(f"{source} line {line}: pass must be blank or 1, 2, ..., not {text!r}")
    return int(text)


def _onset_frame(value: str | None, source: str, line: int, session_id: str) -> int | None:
    text = (value or "").strip()
    if not text:
        return None
    if _WHOLE_NUMBER.fullmatch(text) is None:
        # The value stays out of the message: it may be a mark made blind.
        raise ValueError(
            f"{source} line {line} (session {session_id}): onset_frame must be blank or a "
            "whole number"
        )
    return int(text)


def parse_marks(text: str, source: str) -> list[Mark]:
    """The rows of an onset marks file, in file order.

    Rows whose cells are all blank are skipped, as in `onset.marked_sessions`. A blank pass
    is pass 1, and a blank onset_frame means not marked yet. Raises ValueError naming any
    missing column; the line of a row with no session_id, a pass that isn't 1, 2, ... or an
    onset_frame that isn't a whole number; and any session with two rows for one pass.
    """
    reader = csv.DictReader(io.StringIO(text))
    fields = reader.fieldnames or []
    missing = [name for name in MARK_COLUMNS if name not in fields]
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")
    marks: list[Mark] = []
    seen: set[tuple[str, int]] = set()
    for row in reader:
        if not any(str(cell).strip() for cell in row.values() if cell):
            continue
        line = reader.line_num
        session_id = (row["session_id"] or "").strip()
        if not session_id:
            raise ValueError(f"{source} line {line} has no session_id")
        pass_number = _pass_number(row["pass"], source, line)
        if (session_id, pass_number) in seen:
            raise ValueError(f"{source}: session {session_id} has two rows for pass {pass_number}")
        seen.add((session_id, pass_number))
        onset_frame = _onset_frame(row["onset_frame"], source, line, session_id)
        marks.append(Mark(line, session_id, pass_number, onset_frame))
    return marks


@dataclass(frozen=True)
class Comparison:
    """An onset marked by eye against the operator's onset of the same session."""

    session_id: str
    pass_number: int
    mark: int  # the marked onset frame
    operator: Change | None  # the session's first good-to-fault change; None if it has none

    @property
    def delta_frames(self) -> int | None:
        """Mark minus the operator's onset: positive when the mark is later."""
        return None if self.operator is None else self.mark - self.operator.first_after

    @property
    def delta_s(self) -> float | None:
        """`delta_frames` in seconds at the assumed 55 fps."""
        delta = self.delta_frames
        return None if delta is None else delta / ASSUMED_FPS


def compare_marks(marks: Sequence[Mark], sessions: Sequence[Timeline]) -> list[Comparison]:
    """Every mark with an onset frame against its session's operator onset, by pass, then id.

    Raises ValueError naming the line and session of a mark whose session is not among
    `sessions`, or whose onset_frame lies outside the session's first to last frame, which is
    likely a typo. The value stays out of the message: it may be a mark made blind.
    """
    by_id = {timeline.session_id: timeline for timeline in sessions}
    found: list[Comparison] = []
    for mark in marks:
        timeline = by_id.get(mark.session_id)
        if timeline is None:
            raise ValueError(f"line {mark.line}: session {mark.session_id} is not audited")
        if mark.onset_frame is None:
            continue
        if not timeline.first_frame <= mark.onset_frame <= timeline.last_frame:
            raise ValueError(
                f"line {mark.line}: the onset_frame of session {mark.session_id} is outside "
                f"its frames, {timeline.first_frame} to {timeline.last_frame}"
            )
        operator = operator_onset(timeline)
        found.append(Comparison(mark.session_id, mark.pass_number, mark.onset_frame, operator))
    return sorted(found, key=lambda c: (c.pass_number, c.session_id))


@dataclass(frozen=True)
class PassSummary:
    """One pass of marking against the operator's onsets."""

    pass_number: int
    n_sessions: int  # sessions compared: marked, and with an operator onset
    n_frames: int  # frames in those sessions
    median_frames: float | None  # median of their deltas; None when none is compared
    n_unmarked: int  # rows of this pass with no onset frame yet
    n_no_onset: int  # marked sessions whose labels have no good-to-fault change

    @property
    def median_s(self) -> float | None:
        """`median_frames` in seconds at the assumed 55 fps."""
        return None if self.median_frames is None else self.median_frames / ASSUMED_FPS


def pass_summaries(
    marks: Sequence[Mark], comparisons: Sequence[Comparison], sessions: Sequence[Timeline]
) -> list[PassSummary]:
    """One summary per pass that appears in `marks`, in pass order."""
    frames_of = {timeline.session_id: timeline.n_frames for timeline in sessions}
    summaries: list[PassSummary] = []
    for number in sorted({mark.pass_number for mark in marks}):
        mine = [c for c in comparisons if c.pass_number == number]
        deltas = [delta for c in mine if (delta := c.delta_frames) is not None]
        summaries.append(
            PassSummary(
                pass_number=number,
                n_sessions=len(deltas),
                n_frames=sum(frames_of[c.session_id] for c in mine if c.delta_frames is not None),
                median_frames=float(statistics.median(deltas)) if deltas else None,
                n_unmarked=sum(
                    1 for m in marks if m.pass_number == number and m.onset_frame is None
                ),
                n_no_onset=len(mine) - len(deltas),
            )
        )
    return summaries


def summary_csv(sessions: Sequence[Timeline]) -> str:
    """labels_summary.csv: one row per session, with SUMMARY_COLUMNS.

    Labels in time order are joined by ">"; long gaps are "first..last" missing frame
    numbers joined by ";". Not "first-last": a spreadsheet would read 5-12 as a date.
    """
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(SUMMARY_COLUMNS)
    for timeline in sessions:
        long_gaps = [gap for gap in gaps(timeline) if gap.n_missing > LONG_GAP_FRAMES]
        writer.writerow(
            [
                timeline.session_id,
                timeline.n_frames,
                ">".join(str(label) for label in label_sequence(timeline)),
                len(changes(timeline)),
                ";".join(f"{gap.first}..{gap.last}" for gap in long_gaps),
            ]
        )
    return out.getvalue()


@dataclass(frozen=True)
class Provenance:
    """Where the report's numbers come from (data rule 8)."""

    code_git_sha: str
    index_git_sha: str
    dataset_fingerprint: str
    marks_sha256: str


def _signed(value: float) -> str:
    """A sign on every difference, so its direction can't be misread: +2, -1, +2.5, +0."""
    return f"{value:+g}"


def _signed_s(frames: float) -> str:
    return f"{frames / ASSUMED_FPS:+.2f}"


def _class(label: int) -> str:
    return f"{label} {CLASS_NAMES[label]}"


def _gap_section(sessions: Sequence[Timeline]) -> list[str]:
    onsets = [(t, c) for t in sessions for c in changes(t) if c.good_to_fault]
    with_onset = {t.session_id for t, _change in onsets}
    frames = sum(t.n_frames for t in sessions if t.session_id in with_onset)
    gapped = [(t, c) for t, c in onsets if c.n_missing > 0]
    n_long = sum(1 for _t, c in gapped if c.n_missing > LONG_GAP_FRAMES)
    rows = [
        [
            t.session_id,
            str(c.last_before),
            str(c.first_after),
            _class(c.after),
            str(c.n_missing),
            f"{c.n_missing / ASSUMED_FPS:.2f}",
        ]
        for t, c in gapped
    ]
    header = [
        "Session",
        "Last good frame",
        "First fault frame",
        "Fault",
        "Frames missing",
        "Missing (s)",
    ]
    table = markdown_table(header, rows, [False, True, True, False, True, True])
    return [
        "## Good-to-fault changes with frames missing just before them",
        "",
        f"- Good-to-fault changes (good weld to any fault): {len(onsets)}, in "
        f"{len(with_onset)} of the {len(sessions)} sessions ({frames:,} frames).",
        "- With frame numbers missing between the last good frame and the first fault frame: "
        f"{len(gapped)} of the {len(onsets)}. Each of these happened somewhere in its gap, so "
        "its timing is uncertain by up to that many frames.",
        f"- Of those, with a gap longer than {LONG_GAP_FRAMES} frames: {n_long}.",
        "",
        table if rows else "No good-to-fault change has frames missing just before it.",
    ]


def _marks_section(
    sessions: Sequence[Timeline], marks: Sequence[Mark], comparisons: Sequence[Comparison]
) -> list[str]:
    rows = [
        [
            str(c.pass_number),
            c.session_id,
            str(c.mark),
            "none" if c.operator is None else str(c.operator.first_after),
            "n/a" if c.delta_frames is None else _signed(c.delta_frames),
            "n/a" if c.delta_frames is None else _signed_s(c.delta_frames),
            "n/a" if c.operator is None else str(c.operator.n_missing),
        ]
        for c in comparisons
    ]
    header = [
        "Pass",
        "Session",
        "Mark (frame)",
        "Operator onset (frame)",
        "Δ (frames)",
        "Δ (s)",
        "Missing just before the operator onset (frames)",
    ]
    per_mark = markdown_table(header, rows, [True, False, True, True, True, True, True])
    per_pass = markdown_table(
        [
            "Pass",
            "Sessions",
            "Frames",
            "Median Δ (frames)",
            "Median Δ (s)",
            "Rows not marked yet",
            "Marked sessions without an operator onset",
        ],
        [
            [
                str(s.pass_number),
                str(s.n_sessions),
                f"{s.n_frames:,}",
                "n/a" if s.median_frames is None else _signed(s.median_frames),
                "n/a" if s.median_frames is None else _signed_s(s.median_frames),
                str(s.n_unmarked),
                str(s.n_no_onset),
            ]
            for s in pass_summaries(marks, comparisons, sessions)
        ],
        [True, True, True, True, True, True, True],
    )
    by_id = {t.session_id: t for t in sessions}
    marked = sorted({m.session_id for m in marks})
    several = sum(1 for sid in marked if sum(c.good_to_fault for c in changes(by_id[sid])) > 1)
    return [
        "## Onset marks against the operator's onset",
        "",
        "The operator's onset is the first frame of the first fault that follows good weld "
        "(the session's first good-to-fault change). Δ = mark - operator onset, in frames and "
        "in seconds; a positive Δ means the mark is later. Marks are the rows of "
        "onset_marks.csv with an onset frame; a blank pass counts as pass 1.",
        "",
        per_mark if rows else "No onset marks yet.",
        "",
        "Per pass, over the sessions with both a mark and an operator onset:",
        "",
        per_pass if marks else "No rows in onset_marks.csv.",
        "",
        "- Sessions in onset_marks.csv with more than one good-to-fault change: "
        f"{several} of the {len(marked)}. Δ uses the first.",
        "",
        "No 95% intervals yet: the session-level bootstrap (data rule 6) comes with the stats "
        "module in Session 5. Until then, these are point values for these sessions only.",
    ]


def audit_markdown(
    sessions: Sequence[Timeline],
    marks: Sequence[Mark],
    comparisons: Sequence[Comparison],
    provenance: Provenance,
) -> str:
    """labels.md: label changes, gaps before good-to-fault changes, marks against onsets."""
    n_frames = sum(t.n_frames for t in sessions)
    n_changing = sum(1 for t in sessions if changes(t))
    lines = [
        "# Label audit: training sessions",
        "",
        f"The operator's labels of the {len(sessions)} training sessions ({n_frames:,} "
        "frames) from `weldpilot.data.splits.training_sessions`. Crossing and test-only "
        "sessions are left out (data rules 3 and 4). A session's timeline is its frames from "
        "all its parts in frame-number order; missing frame numbers are gaps, never filled. "
        f"Seconds assume {ASSUMED_FPS:g} fps, the camera rate reported for the SS304 set: an "
        "assumption, not a measurement.",
        "",
        "## Label changes",
        "",
        f"- Sessions that change label at least once: {n_changing} of the {len(sessions)} "
        f"training sessions ({n_frames:,} frames).",
        "",
        "A label change is two consecutive present frames with different labels; frames "
        "missing between them don't make a change. labels_summary.csv has each session's "
        "labels in time order, its number of label changes and its gaps longer than "
        f"{LONG_GAP_FRAMES} frames; label_timelines.png draws the timelines.",
        "",
        *_gap_section(sessions),
        "",
        *_marks_section(sessions, marks, comparisons),
        "",
        "## Provenance",
        "",
        f"- Code: git `{provenance.code_git_sha}`.",
        f"- Index: built at git `{provenance.index_git_sha}` from the dataset with fingerprint "
        f"`{provenance.dataset_fingerprint}`.",
        f"- Marks: onset_marks.csv with SHA-256 `{provenance.marks_sha256}`.",
        f"- Parameters: a long gap has more than {LONG_GAP_FRAMES} missing frames; seconds "
        f"assume {ASSUMED_FPS:g} fps. The command has no config file, so these are all its "
        "parameters.",
        "- Command: `weldpilot audit labels --unblind`.",
    ]
    return "\n".join(lines) + "\n"
