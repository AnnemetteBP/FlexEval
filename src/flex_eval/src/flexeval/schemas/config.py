from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EngineConfig:
    engine: str = "transformers"
    device: str = "auto"
    num_gpus: int = 1
    tensor_parallel_size: int = 1
    dtype: str = "auto"


@dataclass(slots=True)
class SamplingConfig:
    num_samples: int | None = None
    seed: int = 13
    split: str | None = None
    subset: str | None = None


@dataclass(slots=True)
class RunConfig:
    backend: str
    dataset: str
    model: str
    task: str | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    capture_targets: tuple[str, ...] = ()
    analysis_targets: tuple[str, ...] = ()

