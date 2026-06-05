from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ArtifactBundle:
    root: Path
    predictions_path: Path | None = None
    scores_path: Path | None = None
    routing_path: Path | None = None
    latents_path: Path | None = None
    weights_path: Path | None = None
    manifests: dict[str, Path] = field(default_factory=dict)

