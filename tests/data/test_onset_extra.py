"""More tests of the marks file reader in weldpilot.data.onset, with invented session ids.

tests/data/test_onset.py stays closed until onset pass 2, because its fixtures copy real
sessions that are marked blind. Tests added meanwhile live here; merge them into it after
pass 2.
"""

import pytest

from tests.data.invented import SESSION_A, SESSION_B
from weldpilot.data.label_audit import parse_marks
from weldpilot.data.onset import MARKS_COLUMNS, marked_sessions, read_marks

HEADER = ",".join(MARKS_COLUMNS)


def test_read_marks_strips_cells_drops_a_byte_order_mark_and_skips_blank_rows() -> None:
    text = f"﻿{HEADER}\n , , , , \n {SESSION_A} , 2 ,17 , , a note \n"
    columns, rows = read_marks(text)
    assert columns == list(MARKS_COLUMNS)
    assert [row.line for row in rows] == [3]  # the header is line 1, the blank row line 2
    assert rows[0].cells == {
        "session_id": SESSION_A,
        "pass": "2",
        "onset_frame": "17",
        "marked_on": "",
        "note": "a note",
    }


def test_a_short_row_gives_blank_cells() -> None:
    _, rows = read_marks(f"{HEADER}\n{SESSION_A}\n")
    assert rows[0].cells["note"] == ""


def test_an_empty_file_has_no_columns_and_no_rows() -> None:
    assert read_marks("") == ([], [])


def test_marked_sessions_drop_spaces_around_ids_as_parse_marks_does() -> None:
    text = f"{HEADER}\n {SESSION_B} ,1,,,\n{SESSION_A},1,,,\n{SESSION_A},2,,,\n"
    assert marked_sessions(text, "marks.csv") == [SESSION_A, SESSION_B]
    assert sorted({mark.session_id for mark in parse_marks(text, "marks.csv")}) == [
        SESSION_A,
        SESSION_B,
    ]


def test_marked_sessions_read_a_file_that_keeps_its_byte_order_mark() -> None:
    assert marked_sessions(f"﻿{HEADER}\n{SESSION_A},,,,\n", "marks.csv") == [SESSION_A]


def test_parse_marks_names_the_line_of_a_row_without_an_id_after_a_blank_row() -> None:
    with pytest.raises(ValueError, match=r"marks\.csv line 3 has no session_id"):
        parse_marks(f"{HEADER}\n,,,,\n,1,,,\n", "marks.csv")
