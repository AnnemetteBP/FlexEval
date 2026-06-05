from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PredictionRecord:
    example_id: str
    dataset_name: str
    model_name: str
    output_text: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ScoreRecord:
    example_id: str
    dataset_name: str
    metric_name: str
    score: float
    metadata: dict[str, object] = field(default_factory=dict)

