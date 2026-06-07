from __future__ import annotations

from dataclasses import dataclass

from flexeval.schemas.config import RunConfig


@dataclass(slots=True)
class GenericArchitecture:
    """Fallback adapter for architectures without Flex-family conventions."""

    name: str = "generic"

    def validate_run(self, config: RunConfig) -> None:
        if not config.model:
            raise ValueError("Generic runs require a model path or model identifier.")
