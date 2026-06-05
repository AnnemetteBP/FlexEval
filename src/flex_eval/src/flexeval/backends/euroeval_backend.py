from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flexeval.backends.base import BackendAdapter
from flexeval.schemas.config import RunConfig
from flexeval.schemas.example import NormalizedExample
from flexeval.schemas.results import PredictionRecord, ScoreRecord


@dataclass(slots=True)
class EuroEvalBackend(BackendAdapter):
    """Project-owned adapter for EuroEval datasets and scoring."""

    name: str = "euroeval"

    def load_examples(self, config: RunConfig) -> list[NormalizedExample]:
        raise NotImplementedError(
            "EuroEval example loading has not been wired into the shared FlexEval backend yet. "
            "This adapter exists so EuroEval integration lives inside the FlexEval package instead "
            "of separate scripts."
        )

    def score_predictions(
        self,
        config: RunConfig,
        examples: list[NormalizedExample],
        predictions: list[PredictionRecord],
    ) -> list[ScoreRecord]:
        raise NotImplementedError(
            "EuroEval scoring has not been wired into the shared FlexEval backend yet. "
            "Implement scoring through this adapter instead of calling EuroEval directly from shell scripts."
        )
