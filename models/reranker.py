from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pickle

import numpy as np

from models.common import RecommendationCandidate

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional at inspection time
    lgb = None


DEFAULT_FEATURES = [
    "gnn_similarity_score",
    "semantic_similarity_score",
    "item_global_popularity",
    "item_recency_score",
    "user_avg_price_affinity",
    "colour_preference_match",
    "category_preference_match",
]


class LightGBMReranker:
    def __init__(self, feature_names: list[str] | None = None, model: object | None = None) -> None:
        self.feature_names = feature_names or DEFAULT_FEATURES.copy()
        self.model = model

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, group: Iterable[int]) -> None:
        if lgb is None:
            raise ImportError("lightgbm is required to train the reranker.")
        self.model = lgb.LGBMRanker(objective="lambdarank", n_estimators=500)
        self.model.fit(X_train, y_train, group=list(group))

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as handle:
            pickle.dump({"feature_names": self.feature_names, "model": self.model}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMReranker":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        return cls(feature_names=payload["feature_names"], model=payload["model"])

    def _feature_row(self, candidate: RecommendationCandidate) -> list[float]:
        return [float(candidate.features.get(name, 0.0)) for name in self.feature_names]

    def rerank(self, candidates: list[RecommendationCandidate], *, k: int = 12) -> list[RecommendationCandidate]:
        if not candidates:
            return []
        if self.model is None:
            return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:k]

        matrix = np.asarray([self._feature_row(candidate) for candidate in candidates], dtype="float32")
        scores = self.model.predict(matrix)

        reranked: list[RecommendationCandidate] = []
        for candidate, rerank_score in zip(candidates, scores):
            candidate.features["reranker_score"] = float(rerank_score)
            candidate.score = float(rerank_score)
            reranked.append(candidate)
        return sorted(reranked, key=lambda candidate: candidate.score, reverse=True)[:k]

