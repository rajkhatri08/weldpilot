"""Contact sheets for marking fault onsets by eye, blind to the labels.

A sheet shows one session's frames in time order: every 5th frame of each unbroken run of
frame numbers, with its number under it, and a red tile wherever numbers are missing. The
code here gets frame numbers and image paths only, never labels, folders or parts, so a
sheet can't show them.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from weldpilot.data.al5083 import ASSUMED_FPS

STEP = 5
"""Show every STEP-th frame of each unbroken run of frame numbers."""

COLUMNS = 8
THUMB = 192  # px: each thumbnail fits in a THUMB x THUMB box
PAD = 6  # px around each tile
CAPTION = 24  # px under each tile for its frame number
HEADER = 64  # px at the top for two lines about the session

BACKGROUND = (255, 255, 255)
INK = (0, 0, 0)
GAP_FILL = (255, 200, 200)  # light red
GAP_INK = (170, 0, 0)

Font = ImageFont.FreeTypeFont | ImageFont.ImageFont


@dataclass(frozen=True)
class Gap:
    """Frame numbers `first` to `last`, absent from the release."""

    first: int
    last: int

    @property
    def n_missing(self) -> int:
        return self.last - self.first + 1


Tile = int | Gap
"""A frame number to show, or a run of missing frame numbers."""


def sheet_tiles(frame_numbers: Iterable[int], step: int = STEP) -> list[Tile]:
    """Every `step`-th frame of each unbroken run, in time order, with a Gap between runs.

    Sampling restarts at the first frame after each gap, so the frame right after every cut
    is shown. Numbers before the first frame or after the last one are not gaps.
    """
    tiles: list[Tile] = []
    previous: int | None = None
    run_start = 0
    for number in sorted(set(frame_numbers)):
        if previous is None or number > previous + 1:
            if previous is not None:
                tiles.append(Gap(previous + 1, number - 1))
            run_start = number
        if (number - run_start) % step == 0:
            tiles.append(number)
        previous = number
    return tiles


def _header(session_id: str, numbers: Sequence[int], tiles: Sequence[Tile]) -> str:
    gaps = [t for t in tiles if isinstance(t, Gap)]
    missing = sum(g.n_missing for g in gaps)
    return (
        f"Session {session_id}: frames {numbers[0]}-{numbers[-1]}, {len(numbers)} present, "
        f"{missing} missing in {len(gaps)} gaps (red tiles)\n"
        f"Every {STEP}th frame of each unbroken run, about {STEP / ASSUMED_FPS:.2f} s apart "
        f"at an assumed {ASSUMED_FPS:g} fps. Frame number under each thumbnail."
    )


def _paste_thumbnail(sheet: Image.Image, path: Path, x: int, y: int) -> None:
    with Image.open(path, formats=["PNG"]) as frame:
        frame.thumbnail((THUMB, THUMB))
        thumb = frame.convert("RGB")
    sheet.paste(thumb, (x + (THUMB - thumb.width) // 2, y + (THUMB - thumb.height) // 2))


def _draw_gap(draw: ImageDraw.ImageDraw, gap: Gap, x: int, y: int, font: Font) -> None:
    draw.rectangle((x, y, x + THUMB - 1, y + THUMB - 1), fill=GAP_FILL, outline=GAP_INK, width=4)
    span = str(gap.first) if gap.n_missing == 1 else f"{gap.first}-{gap.last}"
    text = f"GAP\n{gap.n_missing} missing\n{span}"
    draw.multiline_text((x + 14, y + 14), text, fill=GAP_INK, font=font, spacing=8)


def make_sheet(session_id: str, frames: Mapping[int, Path]) -> Image.Image:
    """Draw one session's contact sheet from {frame number: PNG path}.

    Raises ValueError if `frames` is empty.
    """
    if not frames:
        raise ValueError(f"session {session_id} has no frames to draw")
    numbers = sorted(frames)
    tiles = sheet_tiles(numbers)
    cell_width, cell_height = THUMB + 2 * PAD, THUMB + CAPTION + 2 * PAD
    rows = ceil(len(tiles) / COLUMNS)
    sheet = Image.new("RGB", (COLUMNS * cell_width, HEADER + rows * cell_height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    text_font = ImageFont.load_default(size=16)
    tile_font = ImageFont.load_default(size=14)
    draw.multiline_text(
        (PAD, PAD), _header(session_id, numbers, tiles), fill=INK, font=text_font, spacing=8
    )
    for position, tile in enumerate(tiles):
        x = (position % COLUMNS) * cell_width + PAD
        y = HEADER + (position // COLUMNS) * cell_height + PAD
        if isinstance(tile, Gap):
            _draw_gap(draw, tile, x, y, tile_font)
        else:
            _paste_thumbnail(sheet, frames[tile], x, y)
            draw.text((x, y + THUMB + 4), str(tile), fill=INK, font=tile_font)
    return sheet
