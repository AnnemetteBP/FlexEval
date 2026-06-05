from __future__ import annotations

from typing import Protocol

from flexeval.schemas.config import EngineConfig, RunConfig
from flexeval.schemas.example import NormalizedExample
from flexeval.schemas.results import PredictionRecord


class InferenceEngine(Protocol):
    """Engine interface for executing model inference."""

    name: str

    def generate(
        self,
        config: RunConfig,
        engine_config: EngineConfig,
        examples: list[NormalizedExample],
    ) -> list[PredictionRecord]:
        ...

