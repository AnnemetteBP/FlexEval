from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class NormalizedExample:
    example_id: str
    dataset_name: str
    prompt: str
    reference_answer: str | None = None
    question: str | None = None
    language: str | None = None
    domain: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

