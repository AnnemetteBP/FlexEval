from __future__ import annotations

from typing import Protocol

from flexeval.schemas.artifacts import ArtifactBundle
from flexeval.schemas.config import RunConfig
from flexeval.schemas.example import NormalizedExample


class CaptureModule(Protocol):
    """Capture module for internal model signals."""

    name: str

    def capture(
        self,
        config: RunConfig,
        examples: list[NormalizedExample],
        artifact_bundle: ArtifactBundle,
    ) -> ArtifactBundle:
        ...

