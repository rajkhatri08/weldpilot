"""Session-level index of the TIG Al5083 release: tables of frames, segments and sessions.

Built from folder listings, the two label files and the PNG headers of training frames.
Test images are never opened (data rule 2). The index records facts only; which sessions
train a model is decided in `weldpilot.data.splits`.
"""

import bisect
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from itertools import pairwise
from operator import attrgetter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from weldpilot.data.al5083 import (
    ASSUMED_FPS,
    CLASS_NAMES,
    GOOD_WELD,
    IGNORED_NAMES,
    SPLITS,
    parse_folder_name,
    parse_frame_name,
)

FRAMES_FILE = "frames.parquet"
SEGMENTS_FILE = "segments.parquet"
SESSIONS_FILE = "sessions.parquet"

_SHOWN = 5  # offenders listed per failed check


class DatasetError(ValueError):
    """The raw dataset breaks a layout or labelling rule. The message lists the offenders."""

    def __init__(self, root: Path, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        details = "\n".join(f"- {p}" for p in self.problems)
        super().__init__(f"{root} failed {len(self.problems)} check(s):\n{details}")


def _problem(title: str, offenders: Sequence[str]) -> str:
    """One failed check: its title, how many offenders, and the first few of them."""
    shown = ", ".join(offenders[:_SHOWN])
    more = f", and {len(offenders) - _SHOWN} more" if len(offenders) > _SHOWN else ""
    return f"{title} ({len(offenders)}): {shown}{more}"


def _parses(parse: Callable[[str], object], name: str) -> bool:
    try:
        parse(name)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class SplitListing:
    """Names found in one split folder. Entries are listed, never opened."""

    split: str
    files: tuple[str, ...]  # files directly in the split folder
    folders: Mapping[str, tuple[str, ...]]  # folder name -> names of the files in it
    nested: tuple[str, ...]  # folders inside folders, as "<folder>/<name>"


def list_split(root: Path, split: str) -> SplitListing:
    """List the split folder `root/split` by name, skipping IGNORED_NAMES."""
    files: list[str] = []
    folders: dict[str, tuple[str, ...]] = {}
    nested: list[str] = []
    for entry in sorted((root / split).iterdir()):
        if entry.name in IGNORED_NAMES:
            continue
        if not entry.is_dir():
            files.append(entry.name)
            continue
        inside = sorted(c for c in entry.iterdir() if c.name not in IGNORED_NAMES)
        folders[entry.name] = tuple(c.name for c in inside if not c.is_dir())
        nested.extend(f"{entry.name}/{c.name}" for c in inside if c.is_dir())
    return SplitListing(split, tuple(files), folders, tuple(nested))


def check_layout(listing: SplitListing) -> list[str]:
    """Problems with names: a missing label file, bad folder names, unexpected files or
    folders, and folders with no frames."""
    split = listing.split
    label_file = f"{split}.json"
    problems = [] if label_file in listing.files else [f"missing label file: {split}/{label_file}"]
    bad_names = [f"{split}/{n}" for n in listing.folders if not _parses(parse_folder_name, n)]
    unexpected = [f"{split}/{n}" for n in listing.files if n != label_file]
    unexpected += [f"{split}/{n}/" for n in listing.nested]
    unexpected += [
        f"{split}/{folder}/{n}"
        for folder, names in listing.folders.items()
        for n in names
        if not _parses(parse_frame_name, n)
    ]
    no_frames = [
        f"{split}/{folder}"
        for folder, names in listing.folders.items()
        if not any(_parses(parse_frame_name, n) for n in names)
    ]
    checks = [
        ("folder names that are not <YYMMDD-HHMMSS>-Al 2mm[-partN]", bad_names),
        ("unexpected files or folders", unexpected),
        ("folders with no frame_#####.png", no_frames),
    ]
    return problems + [_problem(title, offenders) for title, offenders in checks if offenders]


def parse_labels(text: str, source: str) -> tuple[dict[str, int], list[str]]:
    """Read a `<split>.json` label file strictly: a JSON object that maps
    "<folder>/frame_#####.png" to a class 0-5.

    Returns the labels and the problems found (empty when the file is clean). A key that
    appears twice is a problem, although plain `json.loads` would silently keep the last.
    """
    duplicates: list[str] = []

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                duplicates.append(key)
            else:
                obj[key] = value
        return obj

    try:
        data = json.loads(text, object_pairs_hook=no_duplicates)
    except json.JSONDecodeError as err:
        return {}, [f"{source} is not valid JSON: {err}"]
    if not isinstance(data, dict):
        return {}, [f"{source} must hold a JSON object, not {type(data).__name__}"]
    labels: dict[str, int] = {}
    not_classes: list[str] = []
    for key, value in data.items():
        # bool is a subclass of int, so `true` would otherwise pass as class 1.
        if isinstance(value, int) and not isinstance(value, bool) and value in CLASS_NAMES:
            labels[key] = value
        else:
            not_classes.append(f"{key}: {value!r}")
    problems = []
    if duplicates:
        problems.append(_problem(f"{source}: keys that appear more than once", duplicates))
    if not_classes:
        problems.append(_problem(f"{source}: labels that are not a class 0-5", not_classes))
    return labels, problems


@dataclass(frozen=True)
class FrameFile:
    """A frame_#####.png found on disk, known by its path only."""

    split: str
    folder: str
    file_name: str
    session_id: str
    part: int | None
    frame_number: int

    @property
    def label_key(self) -> str:
        """Its key in `<split>.json`."""
        return f"{self.folder}/{self.file_name}"

    @property
    def rel_path(self) -> str:
        """Its path relative to the dataset folder, "/"-separated."""
        return f"{self.split}/{self.folder}/{self.file_name}"


def frame_files(listing: SplitListing) -> list[FrameFile]:
    """The frame files of a listing that passed `check_layout`."""
    found: list[FrameFile] = []
    for folder, names in listing.folders.items():
        name = parse_folder_name(folder)
        found.extend(
            FrameFile(listing.split, folder, n, name.session_id, name.part, parse_frame_name(n))
            for n in names
        )
    return found


def check_consistency(
    files: Sequence[FrameFile], labels: Mapping[str, Mapping[str, int]]
) -> list[str]:
    """Problems between files and labels in both splits: a PNG with no label, a label with
    no PNG, a folder holding more than one class, a frame number twice in one session."""
    unlabelled: list[str] = []
    orphans: list[str] = []
    for split in SPLITS:
        on_disk = {f.label_key for f in files if f.split == split}
        labelled = set(labels[split])
        unlabelled += [f"{split}/{key}" for key in sorted(on_disk - labelled)]
        orphans += [f"{split}/{key}" for key in sorted(labelled - on_disk)]

    classes: dict[str, set[int]] = defaultdict(set)
    for f in files:
        if f.label_key in labels[f.split]:
            classes[f"{f.split}/{f.folder}"].add(labels[f.split][f.label_key])
    mixed = [
        f"{folder} (classes {', '.join(str(c) for c in sorted(found))})"
        for folder, found in sorted(classes.items())
        if len(found) > 1
    ]

    places: dict[tuple[str, int], list[str]] = defaultdict(list)
    for f in files:
        places[(f.session_id, f.frame_number)].append(f"{f.split}/{f.folder}")
    repeated = [
        f"session {session_id} frame {number} in {' and '.join(sorted(where))}"
        for (session_id, number), where in sorted(places.items())
        if len(where) > 1
    ]

    checks = [
        ("PNGs with no label", unlabelled),
        ("labels with no PNG", orphans),
        ("folders holding more than one class", mixed),
        ("frame numbers that appear twice in one session", repeated),
    ]
    return [_problem(title, offenders) for title, offenders in checks if offenders]


@dataclass(frozen=True)
class PngHeader:
    width: int
    height: int
    mode: str  # Pillow's name for the pixel format, such as "L" for 8-bit grayscale


def read_png_header(path: Path) -> PngHeader:
    """Width, height and mode from a PNG's header. `Image.open` is lazy: it parses the
    header and leaves the pixel data unread until `.load()`, which is never called here."""
    with Image.open(path, formats=["PNG"]) as image:
        width, height = image.size
        return PngHeader(width, height, image.mode)


@dataclass(frozen=True)
class Frame:
    """One row of frames.parquet."""

    session_id: str
    part: int | None
    folder: str
    split: str  # the official folder, "train" or "test"
    frame_number: int  # position in the session's video
    rel_path: str
    label: int
    t_s: float  # frame_number / ASSUMED_FPS; 55 fps is an assumption
    gap_before: int | None  # numbers missing since the session's previous frame; None first
    width: int | None  # from the PNG header; None for test frames, which are never opened
    height: int | None
    mode: str | None


@dataclass(frozen=True)
class Segment:
    """One row of segments.parquet: one folder."""

    folder: str
    session_id: str
    part: int | None
    split: str
    label: int
    n_frames: int
    first_frame: int
    last_frame: int
    n_missing: int  # numbers in [first_frame, last_frame] that no folder of the session holds


@dataclass(frozen=True)
class Session:
    """One row of sessions.parquet."""

    session_id: str
    in_train: bool
    in_test: bool
    crosses_split: bool
    n_folders: int
    n_frames: int
    labels: tuple[int, ...]  # distinct classes, ascending
    has_good: bool
    n_good_to_fault: int  # consecutive frames that go from good weld to a fault
    first_frame: int
    last_frame: int
    max_gap: int | None  # largest gap_before; None for a one-frame session


@dataclass(frozen=True)
class DatasetIndex:
    frames: tuple[Frame, ...]  # in session time order: session, then frame number
    segments: tuple[Segment, ...]
    sessions: tuple[Session, ...]


_TIME_ORDER = attrgetter("session_id", "frame_number")


def build_frames(
    files: Sequence[FrameFile],
    labels: Mapping[str, Mapping[str, int]],
    headers: Mapping[str, PngHeader],
) -> list[Frame]:
    """One row per frame, in session time order. `headers` maps `rel_path` to a header and
    holds training frames only."""
    frames: list[Frame] = []
    previous: dict[str, int] = {}
    for f in sorted(files, key=_TIME_ORDER):
        header = headers.get(f.rel_path)
        before = previous.get(f.session_id)
        frames.append(
            Frame(
                session_id=f.session_id,
                part=f.part,
                folder=f.folder,
                split=f.split,
                frame_number=f.frame_number,
                rel_path=f.rel_path,
                label=labels[f.split][f.label_key],
                t_s=f.frame_number / ASSUMED_FPS,
                gap_before=None if before is None else f.frame_number - before - 1,
                width=None if header is None else header.width,
                height=None if header is None else header.height,
                mode=None if header is None else header.mode,
            )
        )
        previous[f.session_id] = f.frame_number
    return frames


def build_segments(frames: Sequence[Frame]) -> list[Segment]:
    """One row per folder. `n_missing` counts the numbers in the folder's range that no
    folder of the session holds: frames of an interleaved part are present, not missing."""
    held: dict[str, list[int]] = defaultdict(list)  # session -> its frame numbers, ascending
    by_folder: dict[tuple[str, str], list[Frame]] = defaultdict(list)
    for fr in sorted(frames, key=_TIME_ORDER):
        held[fr.session_id].append(fr.frame_number)
        by_folder[(fr.split, fr.folder)].append(fr)
    segments: list[Segment] = []
    for rows in by_folder.values():
        first, last = rows[0].frame_number, rows[-1].frame_number
        numbers = held[rows[0].session_id]
        present = bisect.bisect_right(numbers, last) - bisect.bisect_left(numbers, first)
        segments.append(
            Segment(
                folder=rows[0].folder,
                session_id=rows[0].session_id,
                part=rows[0].part,
                split=rows[0].split,
                label=rows[0].label,  # one class per folder: check_consistency made sure
                n_frames=len(rows),
                first_frame=first,
                last_frame=last,
                n_missing=last - first + 1 - present,
            )
        )
    return sorted(segments, key=lambda s: (s.session_id, s.part or 0, s.split, s.folder))


def build_sessions(frames: Sequence[Frame]) -> list[Session]:
    """One row per session, from its frames in time order."""
    timelines: dict[str, list[Frame]] = defaultdict(list)
    for fr in sorted(frames, key=_TIME_ORDER):
        timelines[fr.session_id].append(fr)
    sessions: list[Session] = []
    for session_id, timeline in timelines.items():
        splits = {fr.split for fr in timeline}
        labels = [fr.label for fr in timeline]
        gaps = [fr.gap_before for fr in timeline if fr.gap_before is not None]
        onsets = sum(
            1 for now, after in pairwise(labels) if now == GOOD_WELD and after != GOOD_WELD
        )
        sessions.append(
            Session(
                session_id=session_id,
                in_train="train" in splits,
                in_test="test" in splits,
                crosses_split=splits == {"train", "test"},
                n_folders=len({(fr.split, fr.folder) for fr in timeline}),
                n_frames=len(timeline),
                labels=tuple(sorted(set(labels))),
                has_good=GOOD_WELD in labels,
                n_good_to_fault=onsets,
                first_frame=timeline[0].frame_number,
                last_frame=timeline[-1].frame_number,
                max_gap=max(gaps) if gaps else None,
            )
        )
    return sessions


def build_index(root: Path) -> DatasetIndex:
    """Index the release in `root` (`<data root>/raw/tig-al5083`).

    Reads folder listings, both label files and the PNG headers of training frames only.
    Raises DatasetError listing every broken rule; all name and label checks run before
    any header is read.
    """
    missing = [f"{split}/" for split in SPLITS if not (root / split).is_dir()]
    if missing:
        raise DatasetError(root, [_problem("missing split folders", missing)])
    listings = [list_split(root, split) for split in SPLITS]
    problems = [p for listing in listings for p in check_layout(listing)]
    labels: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        path = root / split / f"{split}.json"
        if path.is_file():
            parsed, found = parse_labels(path.read_text(encoding="utf-8"), f"{split}/{path.name}")
            labels[split] = parsed
            problems += found
    if problems:
        raise DatasetError(root, problems)

    files = [f for listing in listings for f in frame_files(listing)]
    problems = check_consistency(files, labels)
    if problems:
        raise DatasetError(root, problems)

    headers: dict[str, PngHeader] = {}
    unreadable: list[str] = []
    for f in files:
        if f.split != "train":
            continue  # data rule 2: test images are never opened
        try:
            headers[f.rel_path] = read_png_header(root / f.rel_path)
        except OSError as err:
            unreadable.append(f"{f.rel_path} ({err})")
    if unreadable:
        raise DatasetError(root, [_problem("training PNGs whose header can't be read", unreadable)])

    frames = build_frames(files, labels, headers)
    return DatasetIndex(tuple(frames), tuple(build_segments(frames)), tuple(build_sessions(frames)))


def session_status(in_train: bool, in_test: bool) -> str:
    """Return "train-only", "test-only" or "crossing" (folders in both train/ and test/)."""
    if in_train and in_test:
        return "crossing"
    return "train-only" if in_train else "test-only"


def summarize(index: DatasetIndex) -> str:
    """Counts per split, session statuses and the training PNG header formats."""
    lines = [f"{'split':<8}{'folders':>9}{'sessions':>10}{'frames':>9}"]
    for split in SPLITS:
        segments = [s for s in index.segments if s.split == split]
        sessions = {s.session_id for s in segments}
        frames = sum(s.n_frames for s in segments)
        lines.append(f"{split:<8}{len(segments):>9,}{len(sessions):>10,}{frames:>9,}")
    totals = (len(index.segments), len(index.sessions), len(index.frames))
    lines.append(f"{'all':<8}{totals[0]:>9,}{totals[1]:>10,}{totals[2]:>9,}")
    status = Counter(session_status(s.in_train, s.in_test) for s in index.sessions)
    lines.append(
        "sessions: "
        + ", ".join(f"{status[k]} {k}" for k in ("train-only", "test-only", "crossing"))
    )
    formats = Counter(f"{f.width}x{f.height} {f.mode}" for f in index.frames if f.split == "train")
    lines.append(
        "train PNG headers: " + ", ".join(f"{k} ({n:,} frames)" for k, n in formats.most_common())
    )
    return "\n".join(lines)


_NULLABLE_INTS = frozenset({"part", "gap_before", "width", "height", "max_gap"})


def _dataframe(rows: Sequence[Any], names: Sequence[str]) -> pd.DataFrame:
    """Columns in field order. Integer columns that can be null use pandas' nullable Int64,
    so a missing part is <NA> instead of turning the whole column into floats."""
    columns: dict[str, Any] = {}
    for name in names:
        values = [getattr(row, name) for row in rows]
        columns[name] = pd.array(values, dtype="Int64") if name in _NULLABLE_INTS else values
    return pd.DataFrame(columns, columns=list(names))


def frames_dataframe(frames: Sequence[Frame]) -> pd.DataFrame:
    return _dataframe(frames, [f.name for f in fields(Frame)])


def segments_dataframe(segments: Sequence[Segment]) -> pd.DataFrame:
    return _dataframe(segments, [f.name for f in fields(Segment)])


def sessions_dataframe(sessions: Sequence[Session]) -> pd.DataFrame:
    table = _dataframe(sessions, [f.name for f in fields(Session)])
    table["labels"] = [list(s.labels) for s in sessions]  # a parquet list<int64> column
    return table


def write_index(
    index: DatasetIndex, out_dir: Path, provenance: Mapping[str, str | float | None]
) -> list[Path]:
    """Write the three tables to `out_dir`. `provenance` goes into each file's parquet
    metadata (pandas stores `DataFrame.attrs` there) and comes back on read."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        FRAMES_FILE: frames_dataframe(index.frames),
        SEGMENTS_FILE: segments_dataframe(index.segments),
        SESSIONS_FILE: sessions_dataframe(index.sessions),
    }
    written: list[Path] = []
    for name, table in tables.items():
        for key, value in provenance.items():
            table.attrs[key] = value
        table.to_parquet(out_dir / name, index=False)
        written.append(out_dir / name)
    return written


def read_table(out_dir: Path, name: str) -> pd.DataFrame:
    """Read one index table, with its provenance in `.attrs`."""
    return pd.read_parquet(out_dir / name)
