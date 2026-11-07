from pathlib import Path

import pytest

from weldpilot.config import get_settings


def test_data_root_defaults_to_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WELDPILOT_DATA", raising=False)
    assert get_settings().data_root == Path("data")


def test_data_root_follows_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WELDPILOT_DATA", str(tmp_path))
    assert get_settings().data_root == tmp_path


def test_reports_root_defaults_to_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WELDPILOT_REPORTS", raising=False)
    assert get_settings().reports_root == Path("reports")


def test_reports_root_follows_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WELDPILOT_REPORTS", str(tmp_path))
    assert get_settings().reports_root == tmp_path
