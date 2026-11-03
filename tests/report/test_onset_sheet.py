"""Tests for `weldpilot.report.onset_sheet`. Tiny synthetic PNGs; never the real data."""

from collections.abc import Iterable
from pathlib import Path

import pytest
from PIL import Image

from weldpilot.report import onset_sheet
from weldpilot.report.onset_sheet import Gap


def _frames(folder: Path, numbers: Iterable[int]) -> dict[int, Path]:
    """Small grayscale PNGs, one per frame number, as {frame number: path}."""
    frames: dict[int, Path] = {}
    for number in numbers:
        path = folder / f"frame_{number:05d}.png"
        Image.new("L", (32, 24), color=number * 10 % 256).save(path)
        frames[number] = path
    return frames


def _has_colour(image: Image.Image, colour: tuple[int, int, int]) -> bool:
    colours = image.getcolors(maxcolors=image.width * image.height)
    assert colours is not None
    return any(found == colour for _count, found in colours)


def test_tiles_take_every_5th_frame_in_time_order() -> None:
    assert onset_sheet.sheet_tiles([12, 3, 0, 7, 5, 1, 2, 4, 6, 8, 9, 10, 11]) == [0, 5, 10]
    assert onset_sheet.sheet_tiles([42]) == [42]


def test_every_gap_gets_one_tile_and_sampling_restarts_after_it() -> None:
    # Runs 0-6, 8-13 and 17-18: a gap of one frame (7) and one of three (14-16).
    numbers = [*range(7), *range(8, 14), 17, 18]
    assert onset_sheet.sheet_tiles(numbers) == [0, 5, Gap(7, 7), 8, 13, Gap(14, 16), 17]


def test_the_gap_colour_is_drawn_exactly_when_frames_are_missing(tmp_path: Path) -> None:
    whole = onset_sheet.make_sheet("170904-141730", _frames(tmp_path, range(12)))
    gapped = onset_sheet.make_sheet("170904-141730", _frames(tmp_path, [*range(5), 9, 10, 11]))
    # The test frames are grayscale, so only a gap tile can hold this light red.
    assert not _has_colour(whole, onset_sheet.GAP_FILL)
    assert _has_colour(gapped, onset_sheet.GAP_FILL)


def test_a_session_with_no_frames_is_an_error() -> None:
    with pytest.raises(ValueError, match="170904-141730"):
        onset_sheet.make_sheet("170904-141730", {})
