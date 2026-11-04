"""Tests for `weldpilot.report.label_timelines`. Synthetic timelines; never the real data."""

from collections.abc import Mapping

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from weldpilot.data import label_audit
from weldpilot.report import label_timelines

SID = "170904-141730"


def _timeline(labels: Mapping[int, int], session_id: str = SID) -> label_audit.Timeline:
    """One session's timeline from {frame number: label}."""
    return label_audit.Timeline(session_id, tuple(labels), tuple(labels.values()))


def _run(label: int, first: int, last: int) -> dict[int, int]:
    return dict.fromkeys(range(first, last + 1), label)


def _rgb(colour: str) -> tuple[int, int, int]:
    r, g, b = (round(255 * c) for c in to_rgb(colour))
    return (r, g, b)


def _pixel(figure: Figure, frame: float, row: int) -> tuple[int, int, int]:
    """The colour drawn at (frame number, session row) of the figure's axes."""
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    x, y = figure.axes[0].transData.transform((frame, row))
    r, g, b = (int(v) for v in rgba[rgba.shape[0] - 1 - int(y), int(x), :3])
    return (r, g, b)


def test_missing_frames_are_left_blank_and_each_class_keeps_its_colour() -> None:
    # Good weld at 0-99 and 200-299, so frames 100-199 are missing inside one label's run;
    # frames 300-399 are missing where the label changes to class 3 (400-599). The samples
    # avoid gridlines (at round frame numbers) and the class number in the middle of a bar.
    labels = {**_run(0, 0, 99), **_run(0, 200, 299), **_run(3, 400, 599)}
    figure = label_timelines.draw_timelines([_timeline(labels)])
    good, fault = label_timelines.CLASS_COLOURS[0], label_timelines.CLASS_COLOURS[3]
    assert _pixel(figure, 37, 0) == _rgb(good)
    assert _pixel(figure, 137, 0) == _rgb(label_timelines.SURFACE)  # inside a run
    assert _pixel(figure, 237, 0) == _rgb(good)
    assert _pixel(figure, 337, 0) == _rgb(label_timelines.SURFACE)  # at a label change
    assert _pixel(figure, 437, 0) == _rgb(fault)


def test_one_strip_per_session_top_to_bottom_in_the_order_given() -> None:
    ids = ["170815-133921", SID, "170913-152931"]
    figure = label_timelines.draw_timelines([_timeline(_run(0, 0, 9), sid) for sid in ids])
    ax = figure.axes[0]
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ids
    assert ax.yaxis_inverted()  # the first session on top


def test_wide_fault_bars_carry_their_class_number_and_narrow_ones_dont() -> None:
    # Of 1,200 frames: 100 of class 2 (wide), one of class 5 (narrow), good weld unlabelled.
    labels = {**_run(0, 0, 999), **_run(2, 1000, 1099), 1100: 5, **_run(0, 1101, 1199)}
    figure = label_timelines.draw_timelines([_timeline(labels)])
    assert [text.get_text() for text in figure.axes[0].texts] == ["2"]


def test_class_numbers_take_the_ink_with_the_higher_contrast() -> None:
    assert label_timelines.label_ink("#000000") == "#ffffff"
    assert label_timelines.label_ink("#ffffff") == label_timelines.INK
    assert label_timelines.label_ink(label_timelines.CLASS_COLOURS[3]) == label_timelines.INK


def test_no_sessions_is_an_error() -> None:
    with pytest.raises(ValueError, match="no sessions"):
        label_timelines.draw_timelines([])
