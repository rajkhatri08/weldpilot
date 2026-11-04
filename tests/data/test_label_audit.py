"""Tests for `weldpilot.data.label_audit`: timelines, gaps, label changes, the operator's
onset and the comparison with onsets marked by eye. Tiny synthetic tables; never the real
data."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd
import pytest

from weldpilot.data import label_audit, onset
from weldpilot.data.al5083 import ASSUMED_FPS
from weldpilot.data.index import FRAMES_FILE, build_index, write_index
from weldpilot.data.splits import load_sessions, training_sessions

MakeDataset = Callable[..., Path]

SID = "170904-141730"
HEADER = "session_id,pass,onset_frame,marked_on,note\n"


def _table(rows: Sequence[tuple[str, int, int]]) -> pd.DataFrame:
    """(session_id, frame_number, label) rows as a frames table."""
    return pd.DataFrame(list(rows), columns=list(label_audit.FRAME_COLUMNS))


def _timeline(labels: Mapping[int, int], session_id: str = SID) -> label_audit.Timeline:
    """One session's timeline from {frame number: label}."""
    rows = [(session_id, number, label) for number, label in labels.items()]
    [timeline] = label_audit.timelines(_table(rows), [session_id])
    return timeline


def _run(label: int, first: int, last: int) -> dict[int, int]:
    """{frame number: label} for frames first to last."""
    return dict.fromkeys(range(first, last + 1), label)


def _mark(
    line: int, pass_number: int, onset_frame: int | None, session_id: str = SID
) -> label_audit.Mark:
    return label_audit.Mark(line, session_id, pass_number, onset_frame)


# Timelines


def test_parts_that_interleave_are_read_in_frame_number_order() -> None:
    # Like the real 170904-141730: parts 2 and 3 sit inside part1's range.
    part1 = [(SID, n, 0) for n in (0, 1, 2, 7, 8, 15)]
    part2 = [(SID, n, 2) for n in (3, 4)]
    part3 = [(SID, n, 4) for n in (10, 11)]
    [timeline] = label_audit.timelines(_table(part1 + part2 + part3), [SID])
    assert timeline.frame_numbers == (0, 1, 2, 3, 4, 7, 8, 10, 11, 15)
    assert timeline.labels == (0, 0, 0, 2, 2, 0, 0, 4, 4, 0)
    assert label_audit.label_sequence(timeline) == [0, 2, 0, 4, 0]
    assert len(label_audit.changes(timeline)) == 4


def test_only_the_sessions_asked_for_come_back_sorted_by_id() -> None:
    rows = [("170913-152931", 0, 0), ("170815-133921", 0, 3), ("170905-112213", 0, 5)]
    found = label_audit.timelines(_table(rows), ["170913-152931", "170815-133921"])
    assert [t.session_id for t in found] == ["170815-133921", "170913-152931"]


def test_no_sessions_asked_for_gives_no_timelines() -> None:
    assert label_audit.timelines(_table([(SID, 0, 0)]), []) == []


@pytest.mark.parametrize(
    ("rows", "problem"),
    [
        ([("170815-133921", 0, 3)], "has no frames"),
        ([(SID, 4, 0), (SID, 4, 2)], "has frame 4 more than once"),
        ([(SID, 0, 0), (SID, 1, 6)], "has a label that is not a class 0-5"),
    ],
    ids=["no-frames", "repeated-frame", "not-a-class"],
)
def test_a_session_that_cant_be_trusted_is_named(
    rows: list[tuple[str, int, int]], problem: str
) -> None:
    table = _table(rows)
    with pytest.raises(ValueError, match=f"session {SID} {problem}"):
        label_audit.timelines(table, [SID])


@pytest.mark.parametrize("column", label_audit.FRAME_COLUMNS)
def test_a_missing_frames_column_is_named(column: str) -> None:
    table = _table([(SID, 0, 0)]).drop(columns=[column])
    with pytest.raises(ValueError, match=column):
        label_audit.timelines(table, [SID])


# Gaps, segments and label changes


def test_more_than_10_missing_frames_is_a_long_gap() -> None:
    # Frames 1-10 are missing (10, not long), then 12-22 (11, long).
    timeline = _timeline({0: 0, 11: 0, 23: 0})
    assert label_audit.gaps(timeline) == [label_audit.Gap(1, 10), label_audit.Gap(12, 22)]
    assert [gap.n_missing for gap in label_audit.gaps(timeline)] == [10, 11]
    assert label_audit.summary_csv([timeline]).splitlines()[1] == f"{SID},3,0,0,12..22"


def test_numbers_before_the_first_frame_or_after_the_last_are_not_gaps() -> None:
    assert label_audit.gaps(_timeline(_run(0, 5, 9))) == []


def test_segments_break_where_the_label_changes_and_where_frames_are_missing() -> None:
    timeline = _timeline({**_run(0, 0, 2), **_run(2, 3, 4), **_run(2, 8, 9)})
    assert label_audit.segments(timeline) == [
        label_audit.Segment(0, 0, 2),
        label_audit.Segment(2, 3, 4),
        label_audit.Segment(2, 8, 9),  # the same label, but frames 5-7 are missing
    ]


def test_a_gap_alone_is_not_a_label_change() -> None:
    timeline = _timeline({0: 0, 1: 0, 5: 0})
    assert label_audit.changes(timeline) == []
    assert label_audit.label_sequence(timeline) == [0]


def test_a_change_across_a_gap_counts_once_and_carries_the_gap() -> None:
    [change] = label_audit.changes(_timeline({0: 0, 1: 0, 5: 2, 6: 2}))
    assert change == label_audit.Change(before=0, after=2, last_before=1, first_after=5)
    assert change.n_missing == 3  # frames 2, 3 and 4: the change happened at one of them
    assert change.good_to_fault


# The operator's onset


def test_the_operator_onset_is_the_first_fault_that_follows_good_weld() -> None:
    # The session starts with a fault. It follows no good weld, so it isn't the onset.
    timeline = _timeline({**_run(2, 0, 1), **_run(0, 2, 3), **_run(3, 4, 5)})
    found = label_audit.operator_onset(timeline)
    assert found is not None
    assert (found.first_after, found.after) == (4, 3)


def test_the_operator_onset_is_the_first_of_several() -> None:
    found = label_audit.operator_onset(_timeline({0: 0, 1: 2, 2: 0, 3: 4}))
    assert found is not None
    assert (found.first_after, found.after) == (1, 2)


@pytest.mark.parametrize(
    "labels",
    [_run(3, 0, 3), _run(0, 0, 3), {**_run(2, 0, 1), **_run(0, 2, 3)}],
    ids=["no-good-weld", "good-weld-only", "fault-then-good"],
)
def test_a_session_with_no_fault_after_good_weld_has_no_onset(labels: dict[int, int]) -> None:
    assert label_audit.operator_onset(_timeline(labels)) is None


def test_good_to_fault_changes_agree_with_the_index(
    make_dataset: MakeDataset, tmp_path: Path
) -> None:
    # The audit and `weldpilot onset pick` must mean the same by a good-to-fault change.
    # Written by the code behind `weldpilot data index`, from tests/data/synth.py.
    derived = tmp_path / "derived"
    write_index(build_index(make_dataset()), derived, {})
    sessions = load_sessions(derived)
    training = training_sessions(sessions)
    found = label_audit.timelines(pd.read_parquet(derived / FRAMES_FILE), training)
    counted = {t.session_id: sum(c.good_to_fault for c in label_audit.changes(t)) for t in found}
    indexed = dict(zip(sessions["session_id"], sessions["n_good_to_fault"], strict=True))
    assert counted == {sid: indexed[sid] for sid in training}
    assert counted[SID] == 2  # one of them after a missing frame; not a match of zeros


# labels_summary.csv


def test_the_summary_has_one_row_per_session() -> None:
    a = _timeline({**_run(0, 0, 6), **_run(3, 19, 39)}, "170907-100000")
    b = _timeline({1: 2, 2: 2, 5: 0, 6: 0}, "170913-152931")
    assert label_audit.summary_csv([a, b]) == (
        "session_id,n_frames,labels_in_time_order,n_label_changes,long_gaps\n"
        "170907-100000,28,0>3,1,7..18\n"
        "170913-152931,4,2>0,1,\n"
    )


def test_long_gaps_are_joined_by_semicolons() -> None:
    timeline = _timeline({0: 0, 20: 0, 40: 0})
    assert label_audit.summary_csv([timeline]).splitlines()[1] == f"{SID},3,0,0,1..19;21..39"


# onset_marks.csv


def test_the_columns_read_are_columns_of_the_marks_file() -> None:
    assert set(label_audit.MARK_COLUMNS) <= set(onset.MARKS_COLUMNS)


def test_marks_parse_with_a_blank_pass_as_pass_1() -> None:
    text = (
        HEADER
        + f"{SID},,85,2026-11-03,by Claude\n"
        + f"{SID},2,90,2026-11-11,\n"
        + "170913-152931,1,,,\n"  # not marked yet
        + ",,,,\n"  # an empty row a spreadsheet app added
    )
    assert label_audit.parse_marks(text, "marks.csv") == [
        label_audit.Mark(line=2, session_id=SID, pass_number=1, onset_frame=85),
        label_audit.Mark(line=3, session_id=SID, pass_number=2, onset_frame=90),
        label_audit.Mark(line=4, session_id="170913-152931", pass_number=1, onset_frame=None),
    ]


def test_a_marks_file_with_no_rows_gives_no_marks() -> None:
    assert label_audit.parse_marks(HEADER, "marks.csv") == []


@pytest.mark.parametrize("column", label_audit.MARK_COLUMNS)
def test_a_missing_marks_column_is_named(column: str) -> None:
    text = ",".join(name for name in onset.MARKS_COLUMNS if name != column) + "\n"
    with pytest.raises(ValueError, match=f"missing columns: {column}"):
        label_audit.parse_marks(text, "marks.csv")


@pytest.mark.parametrize("value", ["two", "0", "-1", "1.5"])
def test_a_pass_that_isnt_1_2_and_so_on_names_its_line(value: str) -> None:
    text = HEADER + f"{SID},1,85,,\n" + f"{SID},{value},90,,\n"
    with pytest.raises(ValueError, match=r"marks\.csv line 3: pass"):
        label_audit.parse_marks(text, "marks.csv")


@pytest.mark.parametrize("value", ["8888.5", "about 8888", "-8888"])
def test_an_onset_frame_that_isnt_a_whole_number_is_named_but_not_shown(value: str) -> None:
    text = HEADER + f"{SID},1,{value},,\n"
    with pytest.raises(ValueError, match=f"line 2 \\(session {SID}\\): onset_frame") as err:
        label_audit.parse_marks(text, "marks.csv")
    assert "8888" not in str(err.value)  # it may be a mark made blind


def test_a_row_without_a_session_id_names_its_line() -> None:
    text = HEADER + f"{SID},1,85,,\n" + ",1,90,,\n"
    with pytest.raises(ValueError, match="line 3 has no session_id"):
        label_audit.parse_marks(text, "marks.csv")


def test_two_rows_for_one_session_and_pass_are_an_error() -> None:
    # A blank pass is pass 1, so these two rows collide.
    text = HEADER + f"{SID},,85,,\n" + f"{SID},1,90,,\n"
    with pytest.raises(ValueError, match=f"session {SID} has two rows for pass 1"):
        label_audit.parse_marks(text, "marks.csv")


# Marks against the operator's onset


def test_delta_is_the_mark_minus_the_operator_onset() -> None:
    timeline = _timeline({**_run(0, 0, 2), **_run(2, 3, 5)})  # the operator's onset: 3
    later, earlier = label_audit.compare_marks([_mark(2, 1, 5), _mark(3, 2, 2)], [timeline])
    assert (later.delta_frames, earlier.delta_frames) == (2, -1)
    assert later.delta_s == pytest.approx(2 / ASSUMED_FPS)
    assert earlier.delta_s == pytest.approx(-1 / ASSUMED_FPS)


def test_comparisons_come_by_pass_then_session_without_unmarked_rows() -> None:
    a = _timeline({**_run(0, 0, 1), **_run(2, 2, 3)}, "170904-141730")
    b = _timeline({**_run(0, 0, 3), **_run(3, 4, 5)}, "170907-100000")
    marks = [
        _mark(2, 2, 2, b.session_id),
        _mark(3, 1, 3, b.session_id),
        _mark(4, 1, 2, a.session_id),
        _mark(5, 2, None, a.session_id),  # not marked yet
    ]
    found = label_audit.compare_marks(marks, [a, b])
    assert [(c.pass_number, c.session_id) for c in found] == [
        (1, a.session_id),
        (1, b.session_id),
        (2, b.session_id),
    ]


def test_a_session_without_an_operator_onset_is_compared_with_none() -> None:
    [found] = label_audit.compare_marks([_mark(2, 1, 4)], [_timeline(_run(3, 0, 5))])
    assert found.operator is None
    assert found.delta_frames is None
    assert found.delta_s is None


@pytest.mark.parametrize("frame", [5, 88])  # before and after frames 10-20
def test_a_mark_outside_its_session_names_the_line_but_not_the_value(frame: int) -> None:
    timeline = _timeline(_run(0, 10, 20))
    marks = [_mark(2, 1, frame)]
    with pytest.raises(ValueError, match=f"line 2: the onset_frame of session {SID}") as err:
        label_audit.compare_marks(marks, [timeline])
    assert str(frame) not in str(err.value)


def test_a_mark_of_a_session_not_audited_is_named() -> None:
    marks = [_mark(2, 1, 3, "170815-133921")]
    with pytest.raises(ValueError, match="line 2: session 170815-133921 is not audited"):
        label_audit.compare_marks(marks, [_timeline(_run(0, 0, 5))])


def test_each_pass_gets_its_median_and_its_n_in_sessions_and_frames() -> None:
    a = _timeline({**_run(0, 0, 1), **_run(2, 2, 3)}, "170904-141730")  # onset 2, 4 frames
    b = _timeline({**_run(0, 0, 3), **_run(3, 4, 5)}, "170907-100000")  # onset 4, 6 frames
    c = _timeline(_run(3, 0, 1), "170815-133921")  # no operator onset, 2 frames
    marks = [
        _mark(2, 1, 3, a.session_id),  # +1
        _mark(3, 1, 4, b.session_id),  # 0
        _mark(4, 1, 1, c.session_id),  # nothing to compare with
        _mark(5, 2, None, a.session_id),  # pass 2, not marked yet
    ]
    sessions = [a, b, c]
    found = label_audit.pass_summaries(marks, label_audit.compare_marks(marks, sessions), sessions)
    assert found == [
        # An even n: the median lies halfway between the middle two.
        label_audit.PassSummary(1, 2, 10, median_frames=0.5, n_unmarked=0, n_no_onset=1),
        label_audit.PassSummary(2, 0, 0, median_frames=None, n_unmarked=1, n_no_onset=0),
    ]
    assert found[0].median_s == pytest.approx(0.5 / ASSUMED_FPS)


# labels.md


def test_the_report_gives_each_number_its_n_unit_and_source() -> None:
    a = _timeline({**_run(0, 0, 6), **_run(3, 19, 39)}, "170907-100000")  # 12 missing first
    b = _timeline(_run(3, 0, 3), "170815-133921")  # never changes label
    marks = [_mark(2, 1, 22, a.session_id)]
    provenance = label_audit.Provenance("c0de", "1dx", "f1n9", "5ha")
    report = label_audit.audit_markdown(
        [a, b], marks, label_audit.compare_marks(marks, [a, b]), provenance
    )
    assert "The operator's labels of the 2 training sessions (32 frames)" in report
    assert "change label at least once: 1 of the 2 training sessions (32 frames)" in report
    assert "(good weld to any fault): 1, in 1 of the 2 sessions (28 frames)" in report
    assert "and the first fault frame: 1 of the 1." in report
    assert "with a gap longer than 10 frames: 1." in report
    assert "| 170907-100000 | 6 | 19 | 3 lack of fusion | 12 | 0.22 |" in report
    # Pass, session, mark, operator onset, delta in frames and in seconds, missing before.
    assert "| 1 | 170907-100000 | 22 | 19 | +3 | +0.05 | 12 |" in report
    assert "| 1 | 1 | 28 | +3 | +0.05 | 0 | 0 |" in report
    assert "Seconds assume 55 fps" in report
    assert "No 95% intervals yet" in report
    for value in ("c0de", "1dx", "f1n9", "5ha"):
        assert value in report
