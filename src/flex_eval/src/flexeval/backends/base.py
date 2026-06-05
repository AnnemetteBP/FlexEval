from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from flexeval.schemas.config import RunConfig
from flexeval.schemas.example import NormalizedExample
from flexeval.schemas.results import PredictionRecord, ScoreRecord


@dataclass(slots=True)
class BackendDataset:
    name: str
    task: str
    split: str | None = None


class BackendAdapter(Protocol):
    """Backend interface for dataset loading and scoring."""

    name: str

    def load_examples(self, config: RunConfig) -> list[NormalizedExample]:
        ...

    def score_predictions(
        self,
        config: RunConfig,
        examples: list[NormalizedExample],
        predictions: list[PredictionRecord],
    ) -> list[ScoreRecord]:
        ...

