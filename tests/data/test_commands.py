from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from weldpilot.cli import app
from weldpilot.data.fingerprint import parse_record
from weldpilot.data.index import FRAMES_FILE, SEGMENTS_FILE, SESSIONS_FILE

runner = CliRunner()


@pytest.fixture
def root(
    make_dataset: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """The synthetic release, with WELDPILOT_DATA and the working directory in tmp_path, so
    the relative default configs/dataset.yaml can never touch the real one."""
    monkeypatch.setenv("WELDPILOT_DATA", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return make_dataset()


def _flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01
    path.write_bytes(bytes(data))


def test_verify_records_then_checks_the_fingerprint(root: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["data", "verify", "--record"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "configs" / "dataset.yaml").is_file()

    result = runner.invoke(app, ["data", "verify"])
    assert result.exit_code == 0, result.output
    assert "OK: the dataset matches configs/dataset.yaml" in result.output

    _flip_last_byte(root / "README")
    result = runner.invoke(app, ["data", "verify"])
    assert result.exit_code == 1
    assert "does not match" in result.output


def test_verify_without_a_record_fails_with_a_hint(root: Path) -> None:
    result = runner.invoke(app, ["data", "verify"])
    assert result.exit_code == 1
    assert "run `weldpilot data verify --record` first" in result.output


def test_index_refuses_without_a_record(root: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["data", "index"])
    assert result.exit_code == 1
    assert "run `weldpilot data verify --record` first" in result.output
    assert not (tmp_path / "derived").exists()


def test_index_refuses_when_the_data_changed(root: Path, tmp_path: Path) -> None:
    assert runner.invoke(app, ["data", "verify", "--record"]).exit_code == 0
    _flip_last_byte(root / "train" / "train.json")
    result = runner.invoke(app, ["data", "index"])
    assert result.exit_code == 1
    assert "- fingerprint: recorded" in result.output
    assert not (tmp_path / "derived").exists()


def test_index_stores_the_checked_fingerprint_and_composition_reports_it(
    root: Path, tmp_path: Path
) -> None:
    assert runner.invoke(app, ["data", "verify", "--record"]).exit_code == 0
    recorded = parse_record((tmp_path / "configs" / "dataset.yaml").read_text())

    result = runner.invoke(app, ["data", "index"])
    assert result.exit_code == 0, result.output
    for name in (FRAMES_FILE, SEGMENTS_FILE, SESSIONS_FILE):
        table = pd.read_parquet(tmp_path / "derived" / name)
        assert table.attrs["dataset_fingerprint"] == recorded.fingerprint

    result = runner.invoke(app, ["data", "composition"])
    assert result.exit_code == 0, result.output
    assert "| train | 7 | 4 | 20 | 0.4 |" in result.output
    assert "fewer than 2 train-only sessions" in result.output
    assert recorded.fingerprint in result.output


def test_composition_without_an_index_fails_with_a_hint(root: Path) -> None:
    result = runner.invoke(app, ["data", "composition"])
    assert result.exit_code == 1
    assert "run `weldpilot data index` first" in result.output
