from __future__ import annotations

from flexeval.backends.base import BackendAdapter
from flexeval.backends.euroeval_backend import EuroEvalBackend
from flexeval.backends.olmes_backend import OlmesBackend

BACKEND_REGISTRY: dict[str, type[BackendAdapter]] = {
    "euroeval": EuroEvalBackend,
    "olmes": OlmesBackend,
}


def create_backend(name: str) -> BackendAdapter:
    key = name.strip().lower()
    try:
        backend_cls = BACKEND_REGISTRY[key]
    except KeyError as exc:
        supported = ", ".join(sorted(BACKEND_REGISTRY))
        raise ValueError(f"Unsupported backend '{name}'. Supported backends: {supported}.") from exc
    return backend_cls()
