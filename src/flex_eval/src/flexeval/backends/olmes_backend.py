from __future__ import annotations

from dataclasses import dataclass

from flexeval.backends.base import BackendAdapter
from flexeval.schemas.config import RunConfig
from flexeval.schemas.example import NormalizedExample
from flexeval.schemas.results import PredictionRecord, ScoreRecord


@dataclass(slots=True)
class OlmesBackend(BackendAdapter):
    """Project-owned adapter for olmes/FlexOlmo datasets and scoring."""

    name: str = "olmes"

    def load_examples(self, config: RunConfig) -> list[NormalizedExample]:
        raise NotImplementedError(
            "olmes/FlexOlmo example loading has not been wired into the shared FlexEval backend yet. "
            "This adapter is the project-owned integration point that should replace separate eval scripts."
        )

    def score_predictions(
        self,
        config: RunConfig,
        examples: list[NormalizedExample],
        predictions: list[PredictionRecord],
    ) -> list[ScoreRecord]:
        raise NotImplementedError(
            "olmes/FlexOlmo scoring has not been wired into the shared FlexEval backend yet. "
            "Implement scoring through this adapter instead of separate script entrypoints."
        )
