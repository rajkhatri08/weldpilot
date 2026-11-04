"""`weldpilot audit labels`: audit the operator's labels of the training sessions.

Its outputs show the labels and onsets of the sessions in onset_marks.csv, which are marked
blind, so it refuses to run without --unblind. Only after onset pass 2 is done should the
labels of those sessions be seen.
"""

import hashlib
from pathlib import Path
from typing import Annotated, NoReturn

import pandas as pd
import typer

from weldpilot.config import get_settings
from weldpilot.data import splits
from weldpilot.data.index import FRAMES_FILE
from weldpilot.data.label_audit import (
    Provenance,
    audit_markdown,
    compare_marks,
    parse_marks,
    summary_csv,
    timelines,
)
from weldpilot.data.onset import MARKS_FILE_NAME
from weldpilot.provenance import git_sha
from weldpilot.report.label_timelines import draw_timelines

app = typer.Typer(no_args_is_help=True, help="Audit the dataset, by session.")

AUDIT_DIR = Path("reports") / "audit"
"""Relative to the working directory, like reports/onset_sheets."""

TIMELINES_FILE = "label_timelines.png"
SUMMARY_FILE = "labels_summary.csv"
REPORT_FILE = "labels.md"

AUDIT_COLUMNS = ["session_id", "split", "frame_number", "label"]
"""All that `labels` reads from frames.parquet."""

BLIND_WARNING = (
    "this writes the operator's labels and onsets of the sessions in onset_marks.csv, which "
    "are marked blind. Run it after onset pass 2 is done, with --unblind."
)


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


@app.command()
def labels(
    unblind: Annotated[
        bool,
        typer.Option(
            "--unblind", help="Onset pass 2 is done: write the labels of the marked sessions."
        ),
    ] = False,
) -> None:
    """Write label timelines, a per-session summary and the onset comparison to reports/audit/."""
    if not unblind:
        _fail(BLIND_WARNING)  # before anything is read
    data_root = get_settings().data_root
    derived = data_root / "derived"
    marks_path = data_root / "annotations" / MARKS_FILE_NAME
    try:
        training = splits.training_sessions(splits.load_sessions(derived))
    except (FileNotFoundError, ValueError) as err:
        _fail(str(err))
    if not training:
        _fail(f"the index in {derived} has no training sessions to audit")
    if not marks_path.is_file():
        _fail(f"no {marks_path}; run `weldpilot onset pick` first")
    marks_bytes = marks_path.read_bytes()
    try:
        # utf-8-sig drops the byte-order mark that Excel and Numbers can add.
        marks = parse_marks(marks_bytes.decode("utf-8-sig"), str(marks_path))
    except ValueError as err:  # UnicodeDecodeError is a ValueError too
        _fail(str(err))
    allowed = set(training)
    refused = sorted({repr(m.session_id) for m in marks if m.session_id not in allowed})
    if refused:
        _fail(f"not training sessions, so nothing was written: {', '.join(refused)}")
    try:
        # The filter keeps every other session's rows, test sessions included, on disk.
        frames = pd.read_parquet(
            derived / FRAMES_FILE,
            columns=AUDIT_COLUMNS,
            filters=[("session_id", "in", training)],
        )
    except FileNotFoundError:
        _fail(f"no {FRAMES_FILE} in {derived}; run `weldpilot data index` first")
    disagree = sorted(set(frames.loc[frames["split"] != "train", "session_id"]))
    if disagree:
        _fail(
            f"frames.parquet doesn't match sessions.parquet for {', '.join(disagree)}; "
            "rebuild the index with `weldpilot data index`"
        )
    try:
        sessions = timelines(frames, training)
    except ValueError as err:
        _fail(f"{derived / FRAMES_FILE}: {err}")
    try:
        comparisons = compare_marks(marks, sessions)
    except ValueError as err:
        _fail(f"{marks_path} {err}")

    # Every check has passed: write all three files.
    code_sha = git_sha()
    provenance = Provenance(
        code_git_sha=code_sha,
        index_git_sha=str(frames.attrs.get("git_sha", "unknown")),
        dataset_fingerprint=str(frames.attrs.get("dataset_fingerprint", "unknown")),
        marks_sha256=hashlib.sha256(marks_bytes).hexdigest(),
    )
    source = f"weldpilot audit labels, git {code_sha}"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    figure = draw_timelines(sessions, footer=source)
    figure.savefig(AUDIT_DIR / TIMELINES_FILE, metadata={"Source": source})
    (AUDIT_DIR / SUMMARY_FILE).write_text(summary_csv(sessions), encoding="utf-8", newline="")
    report = audit_markdown(sessions, marks, comparisons, provenance)
    (AUDIT_DIR / REPORT_FILE).write_text(report, encoding="utf-8", newline="")

    n_frames = sum(t.n_frames for t in sessions)
    passes = ", ".join(str(p) for p in sorted({m.pass_number for m in marks})) or "none"
    typer.echo(f"audited  {len(sessions)} training sessions ({n_frames:,} frames)")
    typer.echo(f"marks    {len(comparisons)} onset marks (passes: {passes})")
    for name in (TIMELINES_FILE, SUMMARY_FILE, REPORT_FILE):
        typer.echo(f"wrote    {AUDIT_DIR / name}")
