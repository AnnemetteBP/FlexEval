from __future__ import annotations

from typing import Protocol

from flexeval.schemas.artifacts import ArtifactBundle
from flexeval.schemas.config import RunConfig


class AnalysisModule(Protocol):
    """Analysis module that consumes shared artifacts only."""

    name: str

    def run(self, config: RunConfig, artifact_bundle: ArtifactBundle) -> None:
        ...

