"""Label timelines: one strip per session in frame-number order, coloured by the operator's
label, with missing frames left blank.

Drawn with matplotlib's `Figure` API rather than pyplot, so no GUI backend is needed: the
default on macOS is a window backend, and CI runs headless.
"""

from collections.abc import Sequence

from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from weldpilot.data.al5083 import CLASS_NAMES, GOOD_WELD
from weldpilot.data.label_audit import Timeline, segments

CLASS_COLOURS: dict[int, str] = {
    0: "#c3c2b7",  # good weld: light grey, so every fault stands out against it
    1: "#2a78d6",  # blue
    2: "#1baf7a",  # aqua
    3: "#eda100",  # yellow
    4: "#008300",  # green
    5: "#e34948",  # red
}
"""One fixed colour per class, so a class looks the same in every figure.

In a timeline any two classes can touch, so all 15 pairs were checked with the dataviz
palette validator: worst colour-blind separation dE 6.9 (target 8) and normal vision dE 15.5
(floor 15), the best of 280 sets tried. Below the target, colour must not carry the class
alone, so fault bars also carry their class number."""

SURFACE = "#fcfcfb"  # the background, which is also what a missing frame shows
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

STRIP = 0.6  # height of a strip; rows are 1 apart, so 0.4 of each row is space
LABEL_SHARE = 0.015  # a fault bar at least this share of the x range gets its class number
WIDTH_IN = 12.0
ROW_IN = 0.32  # inches per session
CHROME_IN = 1.6  # inches for the title, legend, axis and footer
DPI = 150


def _luminance(colour: str) -> float:
    """Relative luminance of a colour such as "#2a78d6", as WCAG 2 defines it."""
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def label_ink(fill: str) -> str:
    """White or near-black, whichever has the higher contrast ratio against `fill`."""
    on_white = (1.0 + 0.05) / (_luminance(fill) + 0.05)
    on_ink = (_luminance(fill) + 0.05) / (_luminance(INK) + 0.05)
    return "#ffffff" if on_white >= on_ink else INK


def draw_timelines(sessions: Sequence[Timeline], footer: str = "") -> Figure:
    """One strip per session, top to bottom in the order given, with frame number on x.

    Each run of consecutive frames with one label is a bar, so missing frames stay blank.
    Bars touch where the label changes: here a space means missing frames, so there is no
    spacer between bars. Raises ValueError if `sessions` is empty.
    """
    if not sessions:
        raise ValueError("no sessions to draw")
    height = CHROME_IN + ROW_IN * len(sessions)
    figure = Figure(figsize=(WIDTH_IN, height), dpi=DPI, facecolor=SURFACE, layout="constrained")
    ax = figure.add_subplot()
    ax.set_facecolor(SURFACE)
    right = max(t.last_frame for t in sessions) + 0.5
    ax.set_xlim(-0.5, right)
    min_labelled = LABEL_SHARE * (right + 0.5)
    present: set[int] = set()
    for row, timeline in enumerate(sessions):
        bars = segments(timeline)
        present.update(bar.label for bar in bars)
        ax.broken_barh(
            [(bar.first - 0.5, bar.n_frames) for bar in bars],
            (row - STRIP / 2, STRIP),
            facecolors=[CLASS_COLOURS[bar.label] for bar in bars],
            linewidth=0,
        )
        for bar in bars:
            if bar.label != GOOD_WELD and bar.n_frames >= min_labelled:
                ax.text(
                    (bar.first + bar.last) / 2,
                    row,
                    str(bar.label),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=label_ink(CLASS_COLOURS[bar.label]),
                )
    ax.set_yticks(range(len(sessions)), labels=[t.session_id for t in sessions])
    ax.set_ylim(len(sessions) - 0.5, -0.5)  # the first session on top
    ax.set_xlabel("Frame number (position in the session's video)", color=SECONDARY_INK)
    ax.tick_params(axis="y", length=0, labelsize=8, labelcolor=SECONDARY_INK)
    ax.tick_params(axis="x", labelsize=8, labelcolor=SECONDARY_INK, color=BASELINE)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    handles = [
        Patch(facecolor=CLASS_COLOURS[label], label=f"{label} {CLASS_NAMES[label]}")
        for label in sorted(present)
    ]
    handles.append(
        Patch(
            facecolor=SURFACE,
            edgecolor=MUTED_INK,
            linewidth=0.6,
            label="blank: missing frames",
        )
    )
    # On the axes, just above the plot: constrained layout then keeps the title clear of it.
    # A figure legend placed "outside upper left" lands under the title instead.
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        borderaxespad=0.4,
        ncols=len(handles),
        frameon=False,
        fontsize=8,
        labelcolor=SECONDARY_INK,
    )
    figure.suptitle(
        f"Operator labels of {len(sessions)} training sessions, in frame-number order",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    if footer:
        figure.supxlabel(footer, x=0.01, ha="left", fontsize=7, color=MUTED_INK)
    return figure
