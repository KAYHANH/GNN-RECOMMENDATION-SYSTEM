from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RecommendationCandidate:
    article_id: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)

