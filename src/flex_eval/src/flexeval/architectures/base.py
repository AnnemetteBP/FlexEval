from __future__ import annotations

from typing import Protocol

from flexeval.schemas.config import RunConfig


class ArchitectureAdapter(Protocol):
    """Architecture-family interface for capability and convention checks."""

    name: str

    def validate_run(self, config: RunConfig) -> None:
        ...
