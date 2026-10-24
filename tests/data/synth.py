"""A tiny synthetic copy of the TIG Al5083 layout for tests. Never the real data."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class SynthFolder:
    split: str
    name: str
    label: int
    frames: tuple[int, ...]


DEFAULT_FOLDERS = (
    # Interleaved, like the real 170904-141730: parts 2 and 3 sit inside part1's range.
    SynthFolder("train", "170904-141730-Al 2mm-part1", 0, (0, 1, 2, 7, 8, 15)),
    SynthFolder("train", "170904-141730-Al 2mm-part2", 2, (3, 4)),
    SynthFolder("train", "170904-141730-Al 2mm-part3", 4, (10, 11)),
    # Crossing: part1 in train/ and part2 in test/, so the onset crosses the split boundary.
    SynthFolder("train", "170904-150144-Al 2mm-part1", 0, (5, 6, 7)),
    SynthFolder("test", "170904-150144-Al 2mm-part2", 1, (8, 9)),
    # Part numbers out of time order, like the real 170913-152931: the fault comes first.
    SynthFolder("train", "170913-152931-Al 2mm-part1", 0, (5, 6)),
    SynthFolder("train", "170913-152931-Al 2mm-part2", 2, (1, 2)),
    # No part suffix and no good weld.
    SynthFolder("train", "170815-133921-Al 2mm", 3, (20, 21, 23)),
    # Test-only.
    SynthFolder("test", "170905-112213-Al 2mm", 5, (0, 1)),
)

README_TEXT = "LABELS:\n\ngood weld           0\nburn through        1\n"


def write_png(path: Path) -> None:
    """An 8x8 grayscale PNG."""
    Image.new("L", (8, 8)).save(path)


def write_dataset(root: Path, folders: Sequence[SynthFolder] = DEFAULT_FOLDERS) -> Path:
    """Write the folders of frame_#####.png, train.json, test.json and README under `root`."""
    labels: dict[str, dict[str, int]] = {"train": {}, "test": {}}
    for split in labels:
        (root / split).mkdir(parents=True)
    for folder in folders:
        path = root / folder.split / folder.name
        path.mkdir()
        for number in folder.frames:
            name = f"frame_{number:05d}.png"
            write_png(path / name)
            labels[folder.split][f"{folder.name}/{name}"] = folder.label
    for split, mapping in labels.items():
        (root / split / f"{split}.json").write_text(json.dumps(mapping, indent=2))
    (root / "README").write_text(README_TEXT)
    return root
