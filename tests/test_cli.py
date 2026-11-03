import platform
from pathlib import Path

import pytest
from typer.testing import CliRunner

from weldpilot import __version__
from weldpilot.cli import app

runner = CliRunner()


def test_info_prints_versions_and_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WELDPILOT_DATA", str(tmp_path))
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output
    assert platform.python_version() in result.output
    assert str(tmp_path) in result.output


def test_tracebacks_never_print_local_variables() -> None:
    # Locals can hold labels that Raj marks blind, or test-set details (data rule 2).
    assert app.pretty_exceptions_show_locals is False
