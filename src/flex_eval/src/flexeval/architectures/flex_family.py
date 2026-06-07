from __future__ import annotations

from dataclasses import dataclass

from flexeval.schemas.config import RunConfig


@dataclass(slots=True)
class FlexFamilyArchitecture:
    """Architecture adapter for Flex-family routed models."""

    name: str = "flex-family"
    public_base_expert_id: int = 0

    def validate_run(self, config: RunConfig) -> None:
        if not config.model:
            raise ValueError("Flex-family runs require a model path or model identifier.")
