"""`weldpilot onset ...`: pick sessions to mark fault onsets by eye, and draw their sheets.

Blind on purpose: the labels choose and spread the sample, but nothing here prints a
session's labels or fault classes, and the sheet code never receives them.
"""

from pathlib import Path
from typing import Annotated, NoReturn

import pandas as pd
import typer

from weldpilot.config import get_settings
from weldpilot.data import splits
from weldpilot.data.al5083 import DATASET_DIR
from weldpilot.data.index import FRAMES_FILE
from weldpilot.data.onset import (
    MARKS_FILE_NAME,
    marked_sessions,
    marks_csv,
    onset_candidates,
    pick_sessions,
)
from weldpilot.provenance import git_sha
from weldpilot.report.onset_sheet import make_sheet

app = typer.Typer(no_args_is_help=True, help="Mark fault onsets by eye, blind to the labels.")

SHEETS_DIR = Path("reports") / "onset_sheets"
"""Relative to the working directory, like the configs/dataset.yaml default."""

SHEET_COLUMNS = ["session_id", "split", "frame_number", "rel_path"]
"""All that `sheets` reads from frames.parquet: the label column never leaves the disk."""


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _derived_dir() -> Path:
    return get_settings().data_root / "derived"


def _marks_path() -> Path:
    return get_settings().data_root / "annotations" / MARKS_FILE_NAME


@app.command()
def pick(
    n: Annotated[int, typer.Option("--n", min=1, help="The most sessions to pick.")] = 10,
    seed: Annotated[int, typer.Option(help="The same seed picks the same sessions.")] = 0,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite onset_marks.csv, hand marks and all.")
    ] = False,
) -> None:
    """Pick up to --n training sessions with a good-to-fault change; start onset_marks.csv."""
    marks = _marks_path()
    if marks.exists() and not force:
        _fail(f"{marks} already exists and may hold your marks; pass --force to overwrite it")
    try:
        candidates = onset_candidates(splits.load_sessions(_derived_dir()))
    except (FileNotFoundError, ValueError) as err:
        _fail(str(err))
    picked = pick_sessions(candidates, n, seed)
    marks.parent.mkdir(parents=True, exist_ok=True)
    marks.write_text(marks_csv(picked), encoding="utf-8", newline="")
    typer.echo(f"qualified  {len(candidates)} training sessions with a good-to-fault change")
    typer.echo(f"picked     {len(picked)} (seed {seed}, git {git_sha()})")
    typer.echo(f"wrote      {marks}")


@app.command()
def sheets() -> None:
    """Draw reports/onset_sheets/<session_id>.png for each session in onset_marks.csv."""
    marks = _marks_path()
    if not marks.is_file():
        _fail(f"no {marks}; run `weldpilot onset pick` first")
    derived = _derived_dir()
    try:
        # utf-8-sig drops the byte-order mark that Excel and Numbers can add.
        session_ids = marked_sessions(marks.read_text(encoding="utf-8-sig"), str(marks))
        training = set(splits.training_sessions(splits.load_sessions(derived)))
    except (FileNotFoundError, ValueError) as err:
        _fail(str(err))
    # Every id is checked before anything is drawn, so a refusal leaves no sheets behind.
    refused = [repr(sid) for sid in session_ids if sid not in training]
    if refused:
        _fail(f"not training sessions, so no sheets were drawn: {', '.join(refused)}")
    if not session_ids:
        typer.echo(f"no sessions listed in {marks}")
        return
    try:
        frames = pd.read_parquet(derived / FRAMES_FILE, columns=SHEET_COLUMNS)
    except FileNotFoundError:
        _fail(f"no {FRAMES_FILE} in {derived}; run `weldpilot data index` first")
    root = get_settings().data_root / "raw" / DATASET_DIR
    to_draw: dict[str, dict[int, Path]] = {}
    for session_id in session_ids:
        mine = frames[frames["session_id"] == session_id]
        # A training session has frames in train/ only. If not, the two tables disagree, and
        # no image of the session is opened (data rule 2).
        if mine.empty or set(mine["split"]) != {"train"}:
            _fail(
                f"session {session_id}: frames.parquet doesn't match sessions.parquet; "
                "rebuild the index with `weldpilot data index`"
            )
        to_draw[session_id] = {
            int(number): root / rel_path
            for number, rel_path in zip(mine["frame_number"], mine["rel_path"], strict=True)
        }
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    for session_id, paths in to_draw.items():
        out = SHEETS_DIR / f"{session_id}.png"
        make_sheet(session_id, paths).save(out)
        typer.echo(f"wrote {out}")
