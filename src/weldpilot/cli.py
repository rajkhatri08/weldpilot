"""Command-line interface: `weldpilot <command>`."""

import platform

import typer

from weldpilot import __version__
from weldpilot.config import get_settings
from weldpilot.data.audit_commands import app as audit_app
from weldpilot.data.commands import app as data_app
from weldpilot.data.onset_commands import app as onset_app
from weldpilot.provenance import git_sha

# A crash must never print local variables: they can hold labels that Raj marks blind, or
# test-set details (data rule 2). This is Typer's default too; here it is pinned.
app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
app.add_typer(data_app, name="data")
app.add_typer(onset_app, name="onset")
app.add_typer(audit_app, name="audit")


@app.callback()
def main() -> None:
    """WeldPilot: advisory real-time TIG weld monitoring."""
    # An explicit callback keeps Typer in subcommand mode, so `weldpilot info` works
    # even while `info` is the only command.


@app.command()
def info() -> None:
    """Print the package version, Python version, git commit and data root."""
    settings = get_settings()
    typer.echo(f"weldpilot  {__version__}")
    typer.echo(f"python     {platform.python_version()}")
    typer.echo(f"git sha    {git_sha()}")
    typer.echo(f"data root  {settings.data_root}")
