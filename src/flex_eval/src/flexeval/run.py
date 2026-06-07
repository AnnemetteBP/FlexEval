from __future__ import annotations

from dataclasses import dataclass

from flexeval.architectures.registry import create_architecture
from flexeval.backends.registry import create_backend
from flexeval.engines.registry import create_engine
from flexeval.schemas.config import RunConfig
from flexeval.schemas.results import PredictionRecord, ScoreRecord


@dataclass(slots=True)
class RunOutputs:
    predictions: list[PredictionRecord]
    scores: list[ScoreRecord]


def run_evaluation(config: RunConfig) -> RunOutputs:
    architecture = create_architecture(config.architecture)
    architecture.validate_run(config)
    backend = create_backend(config.backend)
    engine = create_engine(config.engine.engine)

    examples = backend.load_examples(config)
    predictions = engine.generate(config, config.engine, examples)
    scores = backend.score_predictions(config, examples, predictions)
    return RunOutputs(predictions=predictions, scores=scores)
