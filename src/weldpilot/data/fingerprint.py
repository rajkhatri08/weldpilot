"""Content fingerprint of a dataset folder: one SHA-256 over every file's path, size and hash.

The fingerprint is the SHA-256 of a manifest with one line per file, sorted by path:

    <path relative to the dataset folder, "/"-separated>\\t<size in bytes>\\t<SHA-256>\\n

Sorting makes it independent of directory listing order, which differs between file
systems (APFS on a Mac, ext4 on CI). Files are hashed as bytes and never decoded, which
data rule 2 allows for test files: SHA-256 checksums for the dataset fingerprint.
"""

import hashlib
from collections.abc import Collection, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath

import yaml
from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class FileDigest:
    rel_path: str  # relative to the dataset folder, "/"-separated
    size: int  # bytes
    sha256: str  # hex digest of the file's bytes


class PartCounts(BaseModel):
    """Files under one part of the dataset: PNGs, other files, and their total size."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    png: int
    other: int
    bytes: int


class FingerprintRecord(BaseModel):
    """What `weldpilot data verify --record` saves to configs/dataset.yaml."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    parts: dict[str, PartCounts]


def manifest(files: Iterable[FileDigest]) -> str:
    """The text the fingerprint hashes: one line per file, sorted by path."""
    ordered = sorted(files, key=lambda f: f.rel_path)
    return "".join(f"{f.rel_path}\t{f.size}\t{f.sha256}\n" for f in ordered)


def combine(files: Iterable[FileDigest]) -> str:
    """SHA-256 of the manifest: one hex string for the whole dataset."""
    return hashlib.sha256(manifest(files).encode("utf-8")).hexdigest()


def count_parts(files: Iterable[FileDigest], parts: Sequence[str]) -> dict[str, PartCounts]:
    """PNGs, other files and bytes under each part (the first component of each path)."""
    png = dict.fromkeys(parts, 0)
    other = dict.fromkeys(parts, 0)
    size = dict.fromkeys(parts, 0)
    for f in files:
        path = PurePosixPath(f.rel_path)
        part = path.parts[0]
        if path.suffix == ".png":
            png[part] += 1
        else:
            other[part] += 1
        size[part] += f.size
    return {p: PartCounts(png=png[p], other=other[p], bytes=size[p]) for p in parts}


def list_files(root: Path, parts: Sequence[str], ignored: Collection[str]) -> list[Path]:
    """Every file in each part of `root` (a folder, walked, or a single file), except names
    in `ignored`. Raises FileNotFoundError if a part is missing."""
    found: list[Path] = []
    for part in parts:
        path = root / part
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            for folder, _subfolders, names in path.walk():
                found.extend(folder / name for name in names if name not in ignored)
        else:
            raise FileNotFoundError(f"{path} is missing")
    return found


def digest_file(path: Path, root: Path) -> FileDigest:
    """Hash one file's bytes."""
    with path.open("rb") as f:
        sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    return FileDigest(path.relative_to(root).as_posix(), path.stat().st_size, sha256)


def fingerprint_dataset(
    root: Path, parts: Sequence[str], ignored: Collection[str], workers: int = 4
) -> FingerprintRecord:
    """Fingerprint every file in `parts` of `root`.

    Files are hashed on `workers` threads. File reads and OpenSSL's SHA-256 release the GIL,
    so threads overlap disk waits with hashing: 6.0 GB took 15.1 s on 1 thread and 4.3 s
    on 4 (M1, 2026-10-24). The manifest is sorted, so thread timing can't change the result.
    """
    paths = list_files(root, parts, ignored)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(partial(digest_file, root=root), paths))
    return FingerprintRecord(
        dataset=root.name, fingerprint=combine(files), parts=count_parts(files, parts)
    )


def compare(recorded: FingerprintRecord, current: FingerprintRecord) -> list[str]:
    """Differences between a recorded and a current fingerprint, counts first. Empty if equal."""
    problems: list[str] = []
    if current.dataset != recorded.dataset:
        problems.append(f"dataset: recorded {recorded.dataset!r}, now {current.dataset!r}")
    names = [*recorded.parts, *(p for p in current.parts if p not in recorded.parts)]
    for name in names:
        was, now = recorded.parts.get(name), current.parts.get(name)
        if was != now:
            problems.append(f"{name}: recorded {_describe(was)}, now {_describe(now)}")
    if current.fingerprint != recorded.fingerprint:
        problems.append(f"fingerprint: recorded {recorded.fingerprint}, now {current.fingerprint}")
    return problems


def _describe(counts: PartCounts | None) -> str:
    if counts is None:
        return "nothing"
    return f"{counts.png:,} PNG + {counts.other:,} other files, {counts.bytes:,} bytes"


def dump_record(record: FingerprintRecord, ignored: Collection[str]) -> str:
    """YAML for configs/dataset.yaml, with a header that documents the manifest format."""
    header = (
        "# Written by `weldpilot data verify --record`. Re-record rather than edit by hand.\n"
        "# fingerprint = SHA-256 of a manifest with one line per file in the parts below,\n"
        '# sorted by path: "<path>\\t<size in bytes>\\t<SHA-256 of the file>\\n". Paths are\n'
        '# relative to the dataset folder and "/"-separated. Files named '
        f"{', '.join(sorted(ignored))} are skipped.\n"
    )
    return header + yaml.safe_dump(record.model_dump(), sort_keys=False)


def parse_record(text: str) -> FingerprintRecord:
    """Read configs/dataset.yaml. Raises ValueError if it isn't a valid record."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise ValueError(f"not valid YAML: {err}") from err
    return FingerprintRecord.model_validate(data)  # ValidationError is a ValueError


def format_record(record: FingerprintRecord) -> str:
    """The summary that `weldpilot data verify` prints: counts per part, then the fingerprint."""
    lines = [f"dataset      {record.dataset}", f"{'part':<10}{'png':>10}{'other':>8}{'bytes':>17}"]
    for name, c in record.parts.items():
        lines.append(f"{name:<10}{c.png:>10,}{c.other:>8,}{c.bytes:>17,}")
    png = sum(c.png for c in record.parts.values())
    other = sum(c.other for c in record.parts.values())
    size = sum(c.bytes for c in record.parts.values())
    lines.append(f"{'total':<10}{png:>10,}{other:>8,}{size:>17,}")
    lines.append(f"fingerprint  {record.fingerprint}")
    return "\n".join(lines)
