"""Command-line interface: `weldpilot <command>`."""

import platform

import typer

from weldpilot import __version__
from weldpilot.config import get_settings
from weldpilot.data.commands import app as data_app
from weldpilot.provenance import git_sha

app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")


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
