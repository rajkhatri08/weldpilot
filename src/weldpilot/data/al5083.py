"""Facts about the TIG Al5083 release (Bacioiu et al., 2019): layout, classes and naming."""

import re
from dataclasses import dataclass
from datetime import datetime

DATASET_DIR = "tig-al5083"
"""Folder under `<data root>/raw/` that holds the release."""

SPLITS = ("train", "test")
"""The official split folders. Each holds `<split>.json` with that split's labels."""

FINGERPRINT_PARTS = ("train", "test", "README")
"""What the fingerprint covers. The byte-identical `al5083/` copy in the archive is left out."""

IGNORED_NAMES = frozenset({".DS_Store"})
"""Files macOS Finder writes. Never part of the release."""

CLASS_NAMES = {
    0: "good weld",
    1: "burn through",
    2: "contamination",
    3: "lack of fusion",
    4: "misalignment",
    5: "lack of penetration",
}
GOOD_WELD = 0

ASSUMED_FPS = 55.0
"""Frames per second. An assumption, not a measurement: the camera rate reported for the
SS304 set. State it wherever frames become seconds."""

_FOLDER = re.compile(r"(?P<session>[0-9]{6}-[0-9]{6})-Al 2mm(?:-part(?P<part>[1-9][0-9]*))?")
_FRAME = re.compile(r"frame_(?P<number>[0-9]{5})\.png")


@dataclass(frozen=True)
class FolderName:
    session_id: str  # YYMMDD-HHMMSS: names the recording session
    part: int | None  # N from "-partN"; None when the folder has no suffix


def parse_folder_name(name: str) -> FolderName:
    """Split a folder name such as "170904-150144-Al 2mm-part1" into session id and part.

    Raises ValueError for anything other than `<YYMMDD-HHMMSS>-Al 2mm[-partN]` with a real
    date and time. `fullmatch` rejects trailing text, including a newline that `$` allows.
    """
    match = _FOLDER.fullmatch(name)
    if match is None:
        raise ValueError(f"not a <YYMMDD-HHMMSS>-Al 2mm[-partN] folder name: {name!r}")
    try:
        datetime.strptime(match["session"], "%y%m%d-%H%M%S")
    except ValueError:
        raise ValueError(f"session id is not a real date and time: {name!r}") from None
    part = match["part"]
    return FolderName(match["session"], None if part is None else int(part))


def parse_frame_name(name: str) -> int:
    """Return the frame number in "frame_#####.png". Raises ValueError for any other name."""
    match = _FRAME.fullmatch(name)
    if match is None:
        raise ValueError(f"not a frame_#####.png file name: {name!r}")
    return int(match["number"])
