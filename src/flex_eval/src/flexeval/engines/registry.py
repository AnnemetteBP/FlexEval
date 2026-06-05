from __future__ import annotations

from flexeval.engines.base import InferenceEngine
from flexeval.engines.transformers_engine import TransformersEngine
from flexeval.engines.vllm_engine import VllmEngine

ENGINE_REGISTRY: dict[str, type[InferenceEngine]] = {
    "transformers": TransformersEngine,
    "hf": TransformersEngine,
    "vllm": VllmEngine,
}


def create_engine(name: str) -> InferenceEngine:
    key = name.strip().lower()
    try:
        engine_cls = ENGINE_REGISTRY[key]
    except KeyError as exc:
        supported = ", ".join(sorted(ENGINE_REGISTRY))
        raise ValueError(f"Unsupported engine '{name}'. Supported engines: {supported}.") from exc
    return engine_cls()
