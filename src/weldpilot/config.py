"""Runtime settings read from the environment."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """WeldPilot settings. Each field can be overridden by an environment variable."""

    data_root: Path = Field(default=Path("data"), validation_alias="WELDPILOT_DATA")


def get_settings() -> Settings:
    """Read settings from the environment now.

    Not cached on purpose: a cache would freeze the first values it saw, so tests that
    change WELDPILOT_DATA would depend on the order they run in.
    """
    return Settings()
