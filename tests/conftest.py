"""Fixtures for every test."""

import pytest


@pytest.fixture(autouse=True)
def _no_reports_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without WELDPILOT_REPORTS, so a value set in your shell can't send a
    command's outputs into your real reports folder. A test that needs it sets it itself."""
    monkeypatch.delenv("WELDPILOT_REPORTS", raising=False)
