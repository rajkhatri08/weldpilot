"""`weldpilot data ...`: thin commands over the dataset functions."""

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from weldpilot.config import get_settings
from weldpilot.data.al5083 import ASSUMED_FPS, DATASET_DIR, FINGERPRINT_PARTS, IGNORED_NAMES
from weldpilot.data.composition import composition_markdown
from weldpilot.data.fingerprint import (
    FingerprintRecord,
    compare,
    dump_record,
    fingerprint_dataset,
    format_record,
    parse_record,
)
from weldpilot.data.index import (
    SEGMENTS_FILE,
    SESSIONS_FILE,
    DatasetError,
    build_index,
    read_table,
    summarize,
    write_index,
)
from weldpilot.provenance import git_sha

app = typer.Typer(no_args_is_help=True, help="Dataset commands for TIG Al5083, by session.")

DATASET_CONFIG = Path("configs") / "dataset.yaml"

ConfigOption = Annotated[
    Path, typer.Option("--config", help="The recorded dataset fingerprint (YAML).")
]


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _dataset_root() -> Path:
    return get_settings().data_root / "raw" / DATASET_DIR


def _derived_dir() -> Path:
    return get_settings().data_root / "derived"


def _fingerprint(root: Path) -> FingerprintRecord:
    try:
        record = fingerprint_dataset(root, FINGERPRINT_PARTS, IGNORED_NAMES)
    except FileNotFoundError as err:
        _fail(str(err))
    typer.echo(format_record(record))
    return record


def _recorded(config: Path) -> FingerprintRecord:
    if not config.is_file():
        _fail(f"no recorded fingerprint at {config}; run `weldpilot data verify --record` first")
    try:
        return parse_record(config.read_text(encoding="utf-8"))
    except ValueError as err:
        _fail(f"{config} is not a valid fingerprint record: {err}")


def _check_fingerprint(root: Path, config: Path) -> FingerprintRecord:
    """Fingerprint the dataset and stop unless it matches the record in `config`. The record
    is read first, so a missing one fails before 6 GB are hashed."""
    recorded = _recorded(config)
    current = _fingerprint(root)
    problems = compare(recorded, current)
    if problems:
        _fail(f"the dataset does not match {config}:\n" + "\n".join(f"- {p}" for p in problems))
    return current


@app.command()
def verify(
    record: Annotated[
        bool, typer.Option("--record", help="Save the fingerprint to --config instead.")
    ] = False,
    config: ConfigOption = DATASET_CONFIG,
) -> None:
    """Fingerprint train/, test/ and README, and check it against --config."""
    root = _dataset_root()
    if not record:
        _check_fingerprint(root, config)
        typer.echo(f"OK: the dataset matches {config}")
        return
    current = _fingerprint(root)
    try:
        previous: str | None = parse_record(config.read_text(encoding="utf-8")).fingerprint
    except (FileNotFoundError, ValueError):
        previous = None
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(dump_record(current, IGNORED_NAMES), encoding="utf-8")
    if previous is None:
        typer.echo(f"Recorded the fingerprint in {config}")
    elif previous == current.fingerprint:
        typer.echo(f"Recorded the fingerprint in {config} (unchanged)")
    else:
        typer.echo(f"Recorded the fingerprint in {config} (it was {previous})")


@app.command()
def index(config: ConfigOption = DATASET_CONFIG) -> None:
    """Check the fingerprint, then write frame, segment and session tables to derived/."""
    root = _dataset_root()
    checked = _check_fingerprint(root, config)
    typer.echo(f"Fingerprint matches {config}. Indexing {root}")
    try:
        built = build_index(root)
    except DatasetError as err:
        _fail(str(err))
    provenance: dict[str, str | float | None] = {
        "git_sha": git_sha(),
        "dataset_fingerprint": checked.fingerprint,  # checked above, not copied
        "assumed_fps": ASSUMED_FPS,
    }
    written = write_index(built, _derived_dir(), provenance)
    typer.echo(summarize(built))
    for path in written:
        typer.echo(f"wrote {path}")


@app.command()
def composition() -> None:
    """Print markdown tables of the release by split and by class, from the index."""
    derived = _derived_dir()
    try:
        segments = read_table(derived, SEGMENTS_FILE)
        sessions = read_table(derived, SESSIONS_FILE)
    except FileNotFoundError:
        _fail(f"no index in {derived}; run `weldpilot data index` first")
    typer.echo(composition_markdown(segments, sessions))
