import pytest

from weldpilot.data.al5083 import (
    FolderName,
    in_train_folder,
    parse_folder_name,
    parse_frame_name,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("170904-150144-Al 2mm-part1", FolderName("170904-150144", 1)),
        ("170906-113317-Al 2mm-part3", FolderName("170906-113317", 3)),
        ("170815-133921-Al 2mm", FolderName("170815-133921", None)),
    ],
)
def test_folder_name_gives_session_and_part(name: str, expected: FolderName) -> None:
    assert parse_folder_name(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "170904-150144-Al 2mm-part",  # part without a number
        "170904-150144-Al 2mm-part0",  # parts start at 1
        "170904-150144-Al2mm",  # no space
        "170904-1501-Al 2mm",  # short time
        "171304-150144-Al 2mm",  # month 13
        "170904-150144-Al 2mm copy",  # a Finder duplicate
        "170904-150144-Al 2mm\n",  # `$` would accept a trailing newline; fullmatch doesn't
        "",
    ],
)
def test_any_other_folder_name_is_an_error(name: str) -> None:
    with pytest.raises(ValueError, match=r"folder name|date and time"):
        parse_folder_name(name)


def test_frame_name_gives_frame_number() -> None:
    assert parse_frame_name("frame_00647.png") == 647


@pytest.mark.parametrize(
    "name", ["frame_647.png", "frame_00647.PNG", "frame_00647.png.bak", "._frame_00647.png"]
)
def test_any_other_frame_name_is_an_error(name: str) -> None:
    with pytest.raises(ValueError, match=r"frame_#####\.png"):
        parse_frame_name(name)


@pytest.mark.parametrize(
    ("rel_path", "expected"),
    [
        ("train/990101-100000-Al 2mm-part1/frame_00001.png", True),
        ("test/990101-120000-Al 2mm/frame_00001.png", False),
        ("train/../test/990101-120000-Al 2mm/frame_00001.png", False),  # starts with train/
        ("/train/990101-100000-Al 2mm/frame_00001.png", False),  # absolute
        ("training/990101-100000-Al 2mm/frame_00001.png", False),
        ("train", False),  # the folder itself, not a frame in it
        ("", False),
    ],
)
def test_only_paths_inside_train_are_in_the_train_folder(rel_path: str, expected: bool) -> None:
    assert in_train_folder(rel_path) is expected
