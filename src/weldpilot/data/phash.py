"""Perceptual hashes of frames, and an exact search for pairs of near-identical frames.

A perceptual hash (pHash) sums up what an image looks like in 64 bits: frames that look
alike get hashes a few bits apart, whereas their SHA-256s share nothing. This is the DCT
pHash as `imagehash.phash` defines it: grayscale, resized to 32x32 with Lanczos, a 2-D
DCT-II, the top-left 8x8 block of low frequencies (DC included), and one bit per
coefficient, set when it is above the block's median. Bits run row by row, the first one
most significant. Because the median splits the 64 coefficients in half, a hash has 32 bits
set unless coefficients tie, and the distance between two such hashes is even.

Hashes of test frames exist for the leakage audit only, which data rule 2 allows. Never use
them, or phash.parquet, as features for anything else.
"""

import multiprocessing
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image
from scipy.fft import dctn  # type: ignore[import-untyped]

PHASH_FILE = "phash.parquet"
"""The hash cache, in `<data root>/derived/`."""

HASH_BITS = 64
IMAGE_SIZE = 32  # px: images are resized to IMAGE_SIZE x IMAGE_SIZE before the DCT
LOW_FREQUENCIES = 8  # the top-left 8x8 block of the DCT gives the 64 bits

PHASH_ALGORITHM = (
    "dct-phash-v1: grayscale, 32x32 Lanczos, 2-D DCT-II (scipy, unnormalized), top-left 8x8 "
    "with DC, bit = coefficient > median, row-major, first bit most significant"
)
"""Names the hash exactly; the cache is valid only for this value. Change it whenever the
hash changes, so that old caches are recomputed instead of silently mixed with new hashes."""

CACHE_COLUMNS = ("rel_path", "size", "mtime_ns", "phash")

CHUNKSIZE = 64
"""Paths sent to a worker process at a time."""

BATCH_PAIRS = 1 << 22
"""About the most candidate pairs held in memory at once by `near_pairs` (4.2 million)."""


def _dct2(pixels: NDArray[np.float64]) -> NDArray[np.float64]:
    """2-D DCT-II, unnormalized like scipy.fftpack's, which imagehash uses. scipy has no
    type hints, so its result is made a typed array here."""
    return np.asarray(dctn(pixels, type=2), dtype=np.float64)


def phash_image(image: Image.Image) -> int:
    """The 64-bit DCT pHash of an image, as an int from 0 to 2**64 - 1."""
    small = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    low = _dct2(np.asarray(small, dtype=np.float64))[:LOW_FREQUENCIES, :LOW_FREQUENCIES]
    bits = low > np.median(low)
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def phash_file(path: Path) -> int:
    """The pHash of one PNG. At module level so that worker processes can import it."""
    try:
        with Image.open(path, formats=["PNG"]) as image:
            return phash_image(image)
    except OSError as err:
        raise OSError(f"{path}: {err}") from err


def hamming(a: int, b: int) -> int:
    """The number of bits that differ between two hashes."""
    return (a ^ b).bit_count()


def hash_files(paths: Sequence[Path], workers: int) -> list[int]:
    """The pHash of each file, in the order given.

    With more than one worker, files are hashed in that many processes, started with
    `spawn`: the macOS default, made explicit so that Linux CI starts them the same way.
    Raises ValueError if `workers` is below 1.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, not {workers}")
    if workers == 1 or len(paths) <= 1:
        return [phash_file(path) for path in paths]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(phash_file, paths, chunksize=CHUNKSIZE))


@dataclass(frozen=True)
class HashTable:
    """Hashes of frames, as cached in phash.parquet."""

    table: pd.DataFrame  # CACHE_COLUMNS, one row per frame, in the order asked for
    n_hashed: int  # hashed in this call
    n_reused: int  # taken from the cache


def _cached_rows(cached: pd.DataFrame | None) -> Mapping[str, tuple[int, int, int]]:
    """rel_path -> (size, mtime_ns, phash) from a cache made with this algorithm."""
    if cached is None or cached.attrs.get("phash_algorithm") != PHASH_ALGORITHM:
        return {}
    rows = zip(cached["rel_path"], cached["size"], cached["mtime_ns"], cached["phash"], strict=True)
    return {str(rel): (int(size), int(mtime), int(value)) for rel, size, mtime, value in rows}


def refresh_hashes(
    rel_paths: Sequence[str], root: Path, cached: pd.DataFrame | None, workers: int
) -> HashTable:
    """Hash every file in `rel_paths` (relative to `root`), reusing the cache where valid.

    A cached row is reused when the cache was made with PHASH_ALGORITHM (in its `attrs`) and
    the file's size and modification time are unchanged; every other file is hashed. Checking
    the cache only stats files, so a run with a full cache opens no image. Raises
    FileNotFoundError naming a missing file, and OSError naming a PNG that can't be read.
    """
    stats = [(root / rel).stat() for rel in rel_paths]
    known = _cached_rows(cached)
    hashes: list[int] = []
    to_hash: list[int] = []  # positions in rel_paths
    for position, (rel, stat) in enumerate(zip(rel_paths, stats, strict=True)):
        row = known.get(rel)
        if row is not None and row[:2] == (stat.st_size, stat.st_mtime_ns):
            hashes.append(row[2])
        else:
            hashes.append(0)  # replaced below
            to_hash.append(position)
    for position, value in zip(
        to_hash, hash_files([root / rel_paths[p] for p in to_hash], workers), strict=True
    ):
        hashes[position] = value
    table = pd.DataFrame(
        {
            "rel_path": list(rel_paths),
            "size": np.array([s.st_size for s in stats], dtype=np.int64),
            "mtime_ns": np.array([s.st_mtime_ns for s in stats], dtype=np.int64),
            "phash": np.array(hashes, dtype=np.uint64),
        },
        columns=list(CACHE_COLUMNS),
    )
    table.attrs["phash_algorithm"] = PHASH_ALGORITHM
    return HashTable(table, n_hashed=len(to_hash), n_reused=len(rel_paths) - len(to_hash))


@dataclass(frozen=True)
class NearPairs:
    """Pairs of positions in a hash array whose hashes differ in at most a few bits."""

    first: NDArray[np.int64]  # first < second in every pair; sorted by first, then second
    second: NDArray[np.int64]
    distance: NDArray[np.int64]  # bits that differ
    n_candidates: int  # pairs whose full distance was checked
    chunk_widths: tuple[int, ...]  # bits in each chunk, most significant chunk first


def chunk_widths(max_distance: int) -> list[int]:
    """Widths of the `max_distance + 1` chunks the 64 bits are cut into, as equal as possible.

    For 4: [13, 13, 13, 13, 12]. Raises ValueError unless 0 <= max_distance < 64.
    """
    if not 0 <= max_distance < HASH_BITS:
        raise ValueError(f"max_distance must be from 0 to {HASH_BITS - 1}, not {max_distance}")
    n_chunks = max_distance + 1
    base, extra = divmod(HASH_BITS, n_chunks)
    return [base + 1] * extra + [base] * (n_chunks - extra)


def _batches(partners: NDArray[np.int64], batch_pairs: int) -> Iterator[tuple[int, int]]:
    """Consecutive slices of positions that each make at most about `batch_pairs` pairs.

    A single position can exceed it on its own; it then forms a slice by itself.
    """
    made = np.cumsum(partners)  # pairs made by positions 0..p
    before = made - partners  # pairs made before position p
    start, n = 0, len(partners)
    while start < n:
        limit = int(before[start]) + batch_pairs
        stop = max(int(np.searchsorted(made, limit, side="right")), start + 1)
        yield start, stop
        start = stop


def near_pairs(
    hashes: NDArray[np.uint64], max_distance: int, batch_pairs: int = BATCH_PAIRS
) -> NearPairs:
    """Every pair of positions whose hashes differ in at most `max_distance` bits. Exact.

    Comparing all n * (n - 1) / 2 pairs grows with n squared. Instead the hash is cut into
    `max_distance + 1` chunks. Two hashes that differ in at most `max_distance` bits can
    differ in at most that many chunks, so at least one chunk is identical (the pigeonhole
    principle). For each chunk, the hashes are sorted by that chunk's value and only those
    with equal values are compared, on all 64 bits. A pair is kept only from the first chunk
    on which its two hashes agree, so each pair comes out exactly once. No pair within
    `max_distance` is missed, unlike approximate methods such as locality-sensitive hashing.

    The cost is the number of candidate pairs, `n_candidates`. It is small when hashes are
    spread out, and grows when many hashes share chunk values. Raises ValueError unless
    0 <= max_distance < 64.
    """
    widths = chunk_widths(max_distance)
    h = np.asarray(hashes, dtype=np.uint64)
    n = len(h)
    shifts: list[np.uint64] = []
    masks: list[np.uint64] = []
    low_bit = HASH_BITS
    for width in widths:
        low_bit -= width
        shifts.append(np.uint64(low_bit))
        masks.append(np.uint64((1 << width) - 1))

    firsts: list[NDArray[np.int64]] = []
    seconds: list[NDArray[np.int64]] = []
    distances: list[NDArray[np.int64]] = []
    n_candidates = 0
    positions = np.arange(n, dtype=np.int64)
    for chunk in range(len(widths) if n > 1 else 0):
        keys = (h >> shifts[chunk]) & masks[chunk]
        order = np.argsort(keys, kind="stable").astype(np.int64)
        ordered = keys[order]
        starts = np.flatnonzero(np.concatenate(([True], ordered[1:] != ordered[:-1])))
        ends = np.append(starts[1:], n)
        # Each sorted position pairs with the later positions of its run of equal keys.
        partners = np.repeat(ends, ends - starts) - positions - 1
        n_candidates += int(partners.sum())
        for start, stop in _batches(partners, batch_pairs):
            counts = partners[start:stop]
            total = int(counts.sum())
            if total == 0:
                continue
            left = np.repeat(positions[start:stop], counts)
            # 1, 2, ..., counts[p] for each p, in one array: a "ragged arange".
            step = np.arange(1, total + 1, dtype=np.int64) - np.repeat(
                np.cumsum(counts) - counts, counts
            )
            i, j = order[left], order[left + step]
            differ = h[i] ^ h[j]
            distance = np.bitwise_count(differ).astype(np.int64)
            keep = distance <= max_distance
            for earlier in range(chunk):  # found already if an earlier chunk agrees
                keep &= ((differ >> shifts[earlier]) & masks[earlier]) != 0
            firsts.append(np.minimum(i, j)[keep])
            seconds.append(np.maximum(i, j)[keep])
            distances.append(distance[keep])

    if not firsts:
        empty = np.zeros(0, dtype=np.int64)
        return NearPairs(empty, empty.copy(), empty.copy(), n_candidates, tuple(widths))
    first, second = np.concatenate(firsts), np.concatenate(seconds)
    order = np.lexsort((second, first))
    return NearPairs(
        first[order], second[order], np.concatenate(distances)[order], n_candidates, tuple(widths)
    )
