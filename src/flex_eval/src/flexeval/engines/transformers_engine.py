from __future__ import annotations

from dataclasses import dataclass

from flexeval.engines.base import InferenceEngine
from flexeval.schemas.config import EngineConfig, RunConfig
from flexeval.schemas.example import NormalizedExample
from flexeval.schemas.results import PredictionRecord


@dataclass(slots=True)
class TransformersEngine(InferenceEngine):
    """Project-owned transformers engine adapter."""

    name: str = "transformers"

    def generate(
        self,
        config: RunConfig,
        engine_config: EngineConfig,
        examples: list[NormalizedExample],
    ) -> list[PredictionRecord]:
        raise NotImplementedError(
            "Shared transformers inference has not been wired into FlexEval yet. "
            "Implement model loading and generation here so all backends reuse the same engine path."
        )
