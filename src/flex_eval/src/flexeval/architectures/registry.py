from __future__ import annotations

from flexeval.architectures.base import ArchitectureAdapter
from flexeval.architectures.flex_family import FlexFamilyArchitecture
from flexeval.architectures.generic import GenericArchitecture

ARCHITECTURE_REGISTRY: dict[str, type[ArchitectureAdapter]] = {
    "generic": GenericArchitecture,
    "flex-family": FlexFamilyArchitecture,
    "flexolmo": FlexFamilyArchitecture,
    "flexmore": FlexFamilyArchitecture,
}


def create_architecture(name: str) -> ArchitectureAdapter:
    key = name.strip().lower()
    try:
        architecture_cls = ARCHITECTURE_REGISTRY[key]
    except KeyError as exc:
        supported = ", ".join(sorted(ARCHITECTURE_REGISTRY))
        raise ValueError(
            f"Unsupported architecture '{name}'. Supported architectures: {supported}."
        ) from exc
    return architecture_cls()
