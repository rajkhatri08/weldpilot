"""Tests for the perceptual hash, its cache and the near-pair search. Synthetic images only."""

import hashlib
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from weldpilot.data import phash


def scene(seed: int, size: tuple[int, int] = (160, 128)) -> Image.Image:
    """A grayscale image a bit like a weld pool: a bright blob on a smooth random background,
    with noise. Different seeds give unrelated images."""
    rng = np.random.default_rng(seed)
    width, height = size
    coarse = Image.fromarray((rng.random((6, 8)) * 255).astype(np.uint8))
    background = np.asarray(coarse.resize(size, Image.Resampling.BICUBIC), dtype=np.float64)
    y, x = np.mgrid[0:height, 0:width]
    cx, cy = rng.uniform(0.3, 0.7) * width, rng.uniform(0.3, 0.7) * height
    blob = 255 * np.exp(-(((x - cx) / (0.12 * width)) ** 2 + ((y - cy) / (0.12 * height)) ** 2))
    noise = rng.normal(0, 8, (height, width))
    pixels = np.clip(0.5 * background + 0.6 * blob + noise, 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))


def analytic(size: tuple[int, int] = (160, 128)) -> Image.Image:
    """A fixed image from a formula, with no random numbers: numpy doesn't promise the same
    random stream across versions, so a seeded image could change under a golden value."""
    width, height = size
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    spot = np.exp(-(((x - 0.6 * width) / 25.0) ** 2 + ((y - 0.4 * height) / 20.0) ** 2))
    pixels = 120 + 50 * np.sin(x / 9.0) * np.cos(y / 13.0) + 80 * spot
    return Image.fromarray(np.clip(np.round(pixels), 0, 255).astype(np.uint8))


def save(image: Image.Image, path: Path) -> Path:
    image.save(path)
    return path


# --- The hash ---------------------------------------------------------------------------


def test_identical_pixels_give_distance_0_whatever_the_png_bytes(tmp_path: Path) -> None:
    image = scene(0)
    fast, small = tmp_path / "fast.png", tmp_path / "small.png"
    image.save(fast, compress_level=0)
    image.save(small, compress_level=9)
    # Different bytes, so the dataset fingerprint (SHA-256) tells the files apart...
    assert hashlib.sha256(fast.read_bytes()).digest() != hashlib.sha256(small.read_bytes()).digest()
    # ...but the pixels are the same, so the perceptual hash can't.
    assert phash.hamming(phash.phash_file(fast), phash.phash_file(small)) == 0
    assert phash.phash_file(fast) == phash.phash_image(image)


@pytest.mark.parametrize("seed", range(5))
def test_a_lightly_blurred_copy_stays_within_4(seed: int) -> None:
    image = scene(seed)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=1))
    assert phash.hamming(phash.phash_image(image), phash.phash_image(blurred)) <= 4


@pytest.mark.parametrize("seed", range(5))
def test_an_unrelated_image_is_far_away(seed: int) -> None:
    # A quarter of the 64 bits; unrelated scenes differ in about half of them.
    assert (
        phash.hamming(phash.phash_image(scene(seed)), phash.phash_image(scene(seed + 1000))) >= 16
    )


@pytest.mark.parametrize("seed", range(5))
def test_a_textured_image_has_32_bits_set(seed: int) -> None:
    # The median splits the 64 coefficients in half, so two such hashes are an even
    # distance apart: "at most 4" means 0, 2 or 4.
    assert phash.phash_image(scene(seed)).bit_count() == 32


def test_a_fixed_image_keeps_its_recorded_hash() -> None:
    # A regression guard, not a spec: if this fails, the hash changed. Change
    # PHASH_ALGORITHM too, so that caches made with the old hash are recomputed.
    assert f"{phash.phash_image(analytic()):016x}" == "9b9c637c9c939c22"


def test_a_png_that_cant_be_read_is_named(tmp_path: Path) -> None:
    broken = tmp_path / "frame_00001.png"
    broken.write_bytes(b"not a png")
    with pytest.raises(OSError, match=r"frame_00001\.png"):
        phash.phash_file(broken)


# --- Hashing many files, and the cache --------------------------------------------------


def test_processes_give_the_same_hashes_in_the_same_order(tmp_path: Path) -> None:
    paths = [save(scene(seed), tmp_path / f"{seed}.png") for seed in range(5)]
    expected = [phash.phash_image(scene(seed)) for seed in range(5)]
    assert phash.hash_files(paths, workers=1) == expected
    assert phash.hash_files(paths, workers=2) == expected


def test_fewer_than_1_worker_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers must be at least 1, not 0"):
        phash.hash_files([save(scene(0), tmp_path / "a.png")], workers=0)


@pytest.fixture
def release(tmp_path: Path) -> tuple[Path, list[str]]:
    """Three frames under tmp_path/root, and their paths relative to it."""
    root = tmp_path / "root"
    (root / "train" / "a").mkdir(parents=True)
    rel_paths = [f"train/a/frame_{n:05d}.png" for n in range(3)]
    for seed, rel in enumerate(rel_paths):
        save(scene(seed), root / rel)
    return root, rel_paths


def test_a_second_refresh_takes_every_hash_from_the_cache(
    release: tuple[Path, list[str]], tmp_path: Path
) -> None:
    root, rel_paths = release
    first = phash.refresh_hashes(rel_paths, root, None, workers=1)
    assert (first.n_hashed, first.n_reused) == (3, 0)
    assert list(first.table.columns) == list(phash.CACHE_COLUMNS)
    assert first.table["phash"].tolist() == [phash.phash_image(scene(s)) for s in range(3)]
    # Through parquet, as the command stores it: the algorithm in attrs must survive.
    first.table.to_parquet(tmp_path / phash.PHASH_FILE, index=False)
    cached = pd.read_parquet(tmp_path / phash.PHASH_FILE)
    second = phash.refresh_hashes(rel_paths, root, cached, workers=1)
    assert (second.n_hashed, second.n_reused) == (0, 3)
    pd.testing.assert_frame_equal(second.table, first.table)


def test_a_changed_file_is_hashed_again(release: tuple[Path, list[str]]) -> None:
    root, rel_paths = release
    first = phash.refresh_hashes(rel_paths, root, None, workers=1)
    changed = root / rel_paths[1]
    save(scene(99), changed)
    stat = changed.stat()
    os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    second = phash.refresh_hashes(rel_paths, root, first.table, workers=1)
    assert (second.n_hashed, second.n_reused) == (1, 2)
    assert second.table["phash"].tolist()[1] == phash.phash_image(scene(99))


def test_a_cache_from_another_algorithm_is_ignored(release: tuple[Path, list[str]]) -> None:
    root, rel_paths = release
    cached = phash.refresh_hashes(rel_paths, root, None, workers=1).table
    cached.attrs["phash_algorithm"] = "dct-phash-v0"
    again = phash.refresh_hashes(rel_paths, root, cached, workers=1)
    assert (again.n_hashed, again.n_reused) == (3, 0)


def test_a_missing_file_is_named(release: tuple[Path, list[str]]) -> None:
    root, rel_paths = release
    with pytest.raises(FileNotFoundError, match=r"frame_00007\.png"):
        phash.refresh_hashes([*rel_paths, "train/a/frame_00007.png"], root, None, workers=1)


# --- The near-pair search ---------------------------------------------------------------


def flip(value: int, bits: Iterable[int]) -> int:
    for bit in bits:
        value ^= 1 << int(bit)
    return value


def brute_force(hashes: NDArray[np.uint64], max_distance: int) -> set[tuple[int, int, int]]:
    """Every pair compared on all 64 bits: the slow reference."""
    i, j = np.triu_indices(len(hashes), 1)
    d = np.bitwise_count(hashes[i] ^ hashes[j]).astype(np.int64)
    close = d <= max_distance
    return set(zip(i[close].tolist(), j[close].tolist(), d[close].tolist(), strict=True))


def as_set(found: phash.NearPairs) -> set[tuple[int, int, int]]:
    columns = (found.first.tolist(), found.second.tolist(), found.distance.tolist())
    return set(zip(*columns, strict=True))


BASE = 0x9B9C637C9C939C22
# With chunks of 13, 13, 13, 13 and 12 bits, chunk 0 holds bits 63-51, chunk 1 bits 50-38,
# chunk 2 bits 37-25, chunk 3 bits 24-12 and chunk 4 bits 11-0.
WORST_CASES = [
    flip(BASE, [63, 50, 37, 24]),  # 4 bits in chunks 0-3: only chunk 4 agrees
    flip(BASE, [50, 37, 24, 11]),  # 4 bits in chunks 1-4: only chunk 0 agrees
]


@pytest.fixture
def hashes() -> NDArray[np.uint64]:
    """Spread-out random hashes, pairs planted at distances 0 to 6, the pigeonhole worst
    cases, and a cluster that shares chunks 0-3, so its pairs agree on several chunks."""
    rng = np.random.default_rng(1)
    spread = rng.integers(0, 2**64, size=2000, dtype=np.uint64).tolist()
    planted = [flip(spread[k], rng.choice(64, size=k % 7, replace=False)) for k in range(1, 50)]
    cluster = [(BASE >> 12 << 12) | int(low) for low in rng.integers(0, 1 << 12, size=60)]
    return np.array([*spread, *planted, BASE, *WORST_CASES, *cluster], dtype=np.uint64)


def test_it_finds_exactly_the_pairs_that_brute_force_finds(hashes: NDArray[np.uint64]) -> None:
    expected = brute_force(hashes, 4)
    base = 2049  # position of BASE in the fixture
    assert {(base, base + 1, 4), (base, base + 2, 4), (base + 1, base + 2, 2)} <= expected
    assert as_set(phash.near_pairs(hashes, 4)) == expected


def test_small_batches_give_the_same_pairs(hashes: NDArray[np.uint64]) -> None:
    assert as_set(phash.near_pairs(hashes, 4, batch_pairs=50)) == brute_force(hashes, 4)


def test_each_pair_comes_out_once_in_order(hashes: NDArray[np.uint64]) -> None:
    found = phash.near_pairs(hashes, 4)
    pairs = list(zip(found.first.tolist(), found.second.tolist(), strict=True))
    assert all(a < b for a, b in pairs)
    assert pairs == sorted(set(pairs))  # sorted, and no pair twice


def test_it_checks_far_fewer_pairs_than_all_of_them(hashes: NDArray[np.uint64]) -> None:
    n = len(hashes)
    found = phash.near_pairs(hashes, 4)
    assert found.chunk_widths == (13, 13, 13, 13, 12)
    assert 0 < found.n_candidates < n * (n - 1) // 2 // 100


def test_distance_4_is_in_and_5_is_out() -> None:
    hashes = np.array([BASE, flip(BASE, range(4)), flip(BASE, range(10, 15))], dtype=np.uint64)
    assert as_set(phash.near_pairs(hashes, 4)) == {(0, 1, 4)}


def test_max_distance_0_finds_identical_hashes_only() -> None:
    hashes = np.array([BASE, flip(BASE, [0]), BASE], dtype=np.uint64)
    assert as_set(phash.near_pairs(hashes, 0)) == {(0, 2, 0)}


@pytest.mark.parametrize("n", [0, 1])
def test_fewer_than_2_hashes_give_no_pairs(n: int) -> None:
    found = phash.near_pairs(np.full(n, BASE, dtype=np.uint64), 4)
    assert (len(found.first), len(found.second), len(found.distance)) == (0, 0, 0)
    assert found.n_candidates == 0


@pytest.mark.parametrize(
    ("max_distance", "widths"), [(0, [64]), (4, [13, 13, 13, 13, 12]), (63, [1] * 64)]
)
def test_chunk_widths_add_up_to_64(max_distance: int, widths: list[int]) -> None:
    assert phash.chunk_widths(max_distance) == widths


@pytest.mark.parametrize("max_distance", [-1, 64])
def test_a_max_distance_out_of_range_is_an_error(max_distance: int) -> None:
    with pytest.raises(ValueError, match=f"not {max_distance}"):
        phash.chunk_widths(max_distance)
