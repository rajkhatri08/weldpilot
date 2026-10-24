import importlib

import pytest

MODULES = [
    "weldpilot",
    "weldpilot.data",
    "weldpilot.detect",
    "weldpilot.cause",
    "weldpilot.act",
    "weldpilot.edge",
    "weldpilot.eval",
    "weldpilot.report",
    "weldpilot.cli",
    "weldpilot.config",
    "weldpilot.provenance",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
