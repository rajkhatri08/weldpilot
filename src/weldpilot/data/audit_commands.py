"""`weldpilot audit ...`: audit the dataset, by session.

`labels` audits the operator's labels of the training sessions. Its outputs show the labels
and onsets of the sessions in onset_marks.csv, which are marked blind, so it refuses to run
without --unblind. Only after onset pass 2 is done should the labels of those sessions be
seen.

`duplicates` finds near-identical frames with perceptual hashes. Its outputs never show the
labels of the marked sessions, so it runs as it is.
"""

import hashlib
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, NoReturn

import numpy as np
import pandas as pd
import typer
from PIL import Image

from weldpilot.config import get_settings
from weldpilot.data import duplicate_audit, phash, splits
from weldpilot.data.al5083 import DATASET_DIR, in_train_folder
from weldpilot.data.fingerprint import digest_file
from weldpilot.data.index import FRAMES_FILE
from weldpilot.data.label_audit import (
    Provenance,
    audit_markdown,
    compare_marks,
    parse_marks,
    summary_csv,
    timelines,
)
from weldpilot.data.onset import MARKS_FILE_NAME, marked_sessions
from weldpilot.provenance import git_sha, provenance_json, write_provenance
from weldpilot.report.label_timelines import draw_timelines

app = typer.Typer(no_args_is_help=True, help="Audit the dataset, by session.")

AUDIT_FOLDER = "audit"
"""In the reports folder (WELDPILOT_REPORTS, default reports/), like onset_sheets."""

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
    """Write label timelines, a per-session summary and the onset comparison to audit/ in
    the reports folder."""
    if not unblind:
        _fail(BLIND_WARNING)  # before anything is read
    code_sha = git_sha()  # before the long steps (data rule 8)
    settings = get_settings()
    data_root = settings.data_root
    audit_dir = settings.reports_root / AUDIT_FOLDER
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

    # Every check has passed: write all three files, then their provenance.
    provenance = Provenance(
        code_git_sha=code_sha,
        index_git_sha=str(frames.attrs.get("git_sha", "unknown")),
        dataset_fingerprint=str(frames.attrs.get("dataset_fingerprint", "unknown")),
        marks_sha256=hashlib.sha256(marks_bytes).hexdigest(),
    )
    source = f"weldpilot audit labels, git {code_sha}"
    audit_dir.mkdir(parents=True, exist_ok=True)
    figure = draw_timelines(sessions, footer=source)
    figure.savefig(audit_dir / TIMELINES_FILE, metadata={"Source": source})
    (audit_dir / SUMMARY_FILE).write_text(summary_csv(sessions), encoding="utf-8", newline="")
    report = audit_markdown(sessions, marks, comparisons, provenance)
    (audit_dir / REPORT_FILE).write_text(report, encoding="utf-8", newline="")
    outputs = (TIMELINES_FILE, SUMMARY_FILE, REPORT_FILE)
    record = provenance_json(
        "weldpilot audit labels",
        [f"{AUDIT_FOLDER}/{name}" for name in outputs],
        code_sha,
        provenance.dataset_fingerprint,
    )
    written = write_provenance(settings.reports_root, "audit_labels", record)

    n_frames = sum(t.n_frames for t in sessions)
    passes = ", ".join(str(p) for p in sorted({m.pass_number for m in marks})) or "none"
    typer.echo(f"audited  {len(sessions)} training sessions ({n_frames:,} frames)")
    typer.echo(f"marks    {len(comparisons)} onset marks (passes: {passes})")
    for name in outputs:
        typer.echo(f"wrote    {audit_dir / name}")
    typer.echo(f"wrote    {written}")


DUPLICATES_CSV = "duplicates.csv"
DUPLICATES_REPORT = "duplicates.md"

MAX_DISTANCE = 4
"""Frames whose hashes differ in at most this many of the 64 bits are flagged."""

THUMB = 192  # px: each frame of a thumbnail fits in a THUMB x THUMB box
THUMB_GAP = 8  # px between the two frames of a training_vs_training thumbnail

HASH_COLUMNS = [*duplicate_audit.FRAME_COLUMNS, "rel_path"]
"""All that `duplicates` reads from frames.parquet to hash and pair frames: no labels."""


def _marked(data_root: Path, training: Sequence[str]) -> list[str]:
    """Ids of the sessions marked blind. Without the marks file they are unknown: stop."""
    path = data_root / "annotations" / MARKS_FILE_NAME
    if not path.is_file():
        _fail(f"no {path}, so the sessions marked blind are unknown; run `weldpilot onset pick`")
    try:
        # utf-8-sig drops the byte-order mark that Excel and Numbers can add.
        marked = marked_sessions(path.read_text(encoding="utf-8-sig"), str(path))
    except ValueError as err:
        _fail(str(err))
    allowed = set(training)
    refused = sorted(repr(sid) for sid in marked if sid not in allowed)
    if refused:
        _fail(f"not training sessions, so nothing was written: {', '.join(refused)}")
    return marked


def _draw_thumbnail(paths: Sequence[Path]) -> Image.Image:
    """Frames side by side on white, each fitted into THUMB x THUMB."""
    tiles: list[Image.Image] = []
    for path in paths:
        with Image.open(path, formats=["PNG"]) as frame:
            frame.thumbnail((THUMB, THUMB))
            tiles.append(frame.convert("L"))
    width = sum(tile.width for tile in tiles) + THUMB_GAP * (len(tiles) - 1)
    sheet = Image.new("L", (width, max(tile.height for tile in tiles)), 255)
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width + THUMB_GAP
    return sheet


@app.command()
def duplicates(
    workers: Annotated[
        int, typer.Option("--workers", min=1, help="Processes that hash frames not in the cache.")
    ] = os.cpu_count() or 1,
) -> None:
    """Hash every frame; report near-identical pairs across the split and between training
    sessions in audit/ in the reports folder."""
    started_run = time.perf_counter()
    code_sha = git_sha()  # before the hashing, the long step (data rule 8)
    settings = get_settings()
    data_root = settings.data_root
    audit_dir = settings.reports_root / AUDIT_FOLDER
    derived = data_root / "derived"
    release = data_root / "raw" / DATASET_DIR
    try:
        sessions = splits.load_sessions(derived)
        training = splits.training_sessions(sessions)
        test = splits.test_sessions(sessions)
    except (FileNotFoundError, ValueError) as err:
        _fail(str(err))
    marked = _marked(data_root, training)
    try:
        frames = pd.read_parquet(derived / FRAMES_FILE, columns=HASH_COLUMNS)
    except FileNotFoundError:
        _fail(f"no {FRAMES_FILE} in {derived}; run `weldpilot data index` first")
    try:
        duplicate_audit.check_frames(frames, training, test)
    except ValueError as err:
        _fail(
            f"frames.parquet doesn't match sessions.parquet: {err}; rebuild the index with "
            "`weldpilot data index`"
        )

    # Hash every frame, reusing the cache. The only place test images are opened (rule 2).
    cache_path = derived / phash.PHASH_FILE
    cached = pd.read_parquet(cache_path) if cache_path.is_file() else None
    started = time.perf_counter()
    try:
        hashed = phash.refresh_hashes(frames["rel_path"].tolist(), release, cached, workers)
    except OSError as err:
        _fail(f"can't hash every frame: {err}")
    hash_seconds = time.perf_counter() - started
    index_sha = str(frames.attrs.get("git_sha", "unknown"))
    fingerprint = str(frames.attrs.get("dataset_fingerprint", "unknown"))
    table = hashed.table
    table.attrs.update(
        {
            "git_sha": code_sha,
            "index_git_sha": index_sha,
            "dataset_fingerprint": fingerprint,
            "use": "the leakage audit only (data rule 2)",
        }
    )
    table.to_parquet(cache_path, index=False)  # saved now, so the hashing is never lost

    started = time.perf_counter()
    hashes = table["phash"].to_numpy(dtype=np.uint64)
    found = phash.near_pairs(hashes, MAX_DISTANCE)
    search_seconds = time.perf_counter() - started
    pairs = duplicate_audit.with_thumbnails(duplicate_audit.classify_pairs(frames, found, training))

    # Labels for the class table only. The filter keeps the marked sessions' rows on disk.
    filters = [("session_id", "not in", marked)] if marked else None
    labels = pd.read_parquet(
        derived / FRAMES_FILE, columns=list(duplicate_audit.LABEL_COLUMNS), filters=filters
    )
    try:
        classes = duplicate_audit.class_table(pairs, labels, training, marked)
    except ValueError as err:
        _fail(f"{derived / FRAMES_FILE}: {err}")

    # Thumbnails show frames of training sessions only: never a test frame (data rule 2).
    # Every frame is checked before any is drawn.
    where = {
        (str(sid), int(number)): (str(split), str(rel))
        for sid, split, number, rel in zip(
            frames["session_id"],
            frames["split"],
            frames["frame_number"],
            frames["rel_path"],
            strict=True,
        )
    }
    allowed = set(training)
    to_draw: list[tuple[Path, list[Path]]] = []
    chosen = pairs[pairs["thumbnail"] != ""]
    for kind, a_sid, a_number, b_sid, b_number, name in zip(
        chosen["kind"],
        chosen["a_session"],
        chosen["a_frame"],
        chosen["b_session"],
        chosen["b_frame"],
        chosen["thumbnail"],
        strict=True,
    ):
        paths: list[Path] = []
        shown = duplicate_audit.thumbnail_frames(kind, a_sid, int(a_number), b_sid, int(b_number))
        for sid, number in shown:
            split, rel = where[(sid, number)]
            # The split column and the path must both say train/ (data rule 2).
            if split != "train" or sid not in allowed or not in_train_folder(rel):
                _fail(f"session {sid} frame {number} is not a training frame, so it isn't drawn")
            paths.append(release / rel)
        to_draw.append((audit_dir / name, paths))

    # Exact copies: SHA-256 of the files of training_vs_test pairs 0 bits apart, and of no
    # other file. Data rule 2 allows checksums for the leakage audit; bytes are read, never
    # decoded, so no test frame is shown.
    same = duplicate_audit.same_hash_pairs(pairs)
    checked = [
        (where[(str(a_sid), int(a_number))][1], where[(str(b_sid), int(b_number))][1])
        for a_sid, a_number, b_sid, b_number in zip(
            same["a_session"], same["a_frame"], same["b_session"], same["b_frame"], strict=True
        )
    ]
    try:
        sha256 = {
            rel: digest_file(release / rel, release).sha256
            for rel in sorted({rel for pair in checked for rel in pair})
        }
    except OSError as err:
        _fail(f"can't checksum a file of a pair with identical hashes: {err}")
    copies = duplicate_audit.exact_copies(checked, sha256)

    thumb_dir = audit_dir / duplicate_audit.THUMBNAIL_DIR
    thumb_dir.mkdir(parents=True, exist_ok=True)
    for old in thumb_dir.glob("*.png"):  # thumbnails of an earlier run
        old.unlink()
    for out, paths in to_draw:
        _draw_thumbnail(paths).save(out)

    bits = np.bitwise_count(hashes)
    train_ids = set(frames.loc[frames["split"] == "train", "session_id"].astype(str))
    scope = duplicate_audit.Scope(
        n_frames=len(frames),
        n_sessions=frames["session_id"].nunique(),
        n_train_frames=int((frames["split"] == "train").sum()),
        n_test_frames=int((frames["split"] == "test").sum()),
        n_training_sessions=len(training),
        n_test_sessions=len(test),
        n_crossing_sessions=len(train_ids & set(test)),
        n_marked=len(marked),
        fewest_bits=int(bits.min()) if len(bits) else 0,
        most_bits=int(bits.max()) if len(bits) else 0,
        max_distance=MAX_DISTANCE,
        n_candidates=found.n_candidates,
        chunk_widths=found.chunk_widths,
        n_thumbnails=len(to_draw),
    )
    provenance = duplicate_audit.Provenance(
        code_git_sha=code_sha,
        index_git_sha=index_sha,
        dataset_fingerprint=fingerprint,
        phash_algorithm=phash.PHASH_ALGORITHM,
    )
    csv_text = duplicate_audit.duplicates_csv(pairs)
    (audit_dir / DUPLICATES_CSV).write_text(csv_text, encoding="utf-8", newline="")
    report = duplicate_audit.duplicates_markdown(pairs, classes, copies, scope, provenance)
    (audit_dir / DUPLICATES_REPORT).write_text(report, encoding="utf-8", newline="")
    # Thumbnail file names are session ids, so the record names their folder.
    record = provenance_json(
        "weldpilot audit duplicates",
        [
            f"{AUDIT_FOLDER}/{DUPLICATES_CSV}",
            f"{AUDIT_FOLDER}/{DUPLICATES_REPORT}",
            f"{AUDIT_FOLDER}/{duplicate_audit.THUMBNAIL_DIR}/",
        ],
        code_sha,
        fingerprint,
    )
    written = write_provenance(settings.reports_root, "audit_duplicates", record)

    n = len(frames)
    all_pairs = n * (n - 1) // 2
    share = 100 * found.n_candidates / all_pairs if all_pairs else 0.0
    processes = f" on {workers} processes" if hashed.n_hashed and workers > 1 else ""
    counts = pairs["kind"].value_counts()
    typer.echo(
        f"hashed    {n:,} frames: {hashed.n_hashed:,} new, {hashed.n_reused:,} from the cache, "
        f"in {hash_seconds:.1f} s{processes}"
    )
    typer.echo(
        f"searched  {found.n_candidates:,} candidate pairs of {all_pairs:,} ({share:.2f}%) "
        f"in {search_seconds:.1f} s"
    )
    for kind in duplicate_audit.KINDS:
        typer.echo(f"pairs     {kind:<24}{int(counts.get(kind, 0)):>10,}")
    typer.echo(
        f"copies    {len(copies.identical):,} byte-identical of the {copies.n_checked:,} "
        f"{duplicate_audit.TRAINING_VS_TEST} pairs 0 bits apart"
    )
    for path in (cache_path, audit_dir / DUPLICATES_CSV, audit_dir / DUPLICATES_REPORT):
        typer.echo(f"wrote     {path}")
    typer.echo(f"wrote     {thumb_dir}/ ({len(to_draw)} thumbnails)")
    typer.echo(f"wrote     {written}")
    typer.echo(f"total     {time.perf_counter() - started_run:.1f} s")
