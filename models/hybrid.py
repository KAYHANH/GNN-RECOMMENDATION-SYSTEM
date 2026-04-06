from __future__ import annotations

from collections import defaultdict

from models.common import RecommendationCandidate


class HybridRecommender:
    def __init__(self, *, gnn_engine: object | None, semantic_engine: object | None, reranker: object | None = None) -> None:
        self.gnn_engine = gnn_engine
        self.semantic_engine = semantic_engine
        self.reranker = reranker

    @staticmethod
    def _merge_candidates(*candidate_groups: list[RecommendationCandidate]) -> list[RecommendationCandidate]:
        merged: dict[str, RecommendationCandidate] = {}
        sources = defaultdict(set)

        for group in candidate_groups:
            for candidate in group:
                existing = merged.get(candidate.article_id)
                if existing is None:
                    merged[candidate.article_id] = candidate
                else:
                    existing.score += candidate.score
                    existing.features.update(candidate.features)
                    existing.metadata = existing.metadata or candidate.metadata
                sources[candidate.article_id].add(candidate.source)

        for article_id, candidate in merged.items():
            candidate.source = "+".join(sorted(sources[article_id]))
        return list(merged.values())

    def recommend(
        self,
        *,
        customer_id: str,
        k: int = 12,
        mode: str = "hybrid",
        profile_text: str | None = None,
    ) -> list[RecommendationCandidate]:
        gnn_candidates = self.gnn_engine.get_candidates(customer_id, k=max(k * 5, 50)) if self.gnn_engine else []

        if mode == "gnn" or not self.semantic_engine:
            merged = gnn_candidates
        else:
            semantic_candidates = []
            if profile_text:
                semantic_candidates = self.semantic_engine.score_candidates(profile_text, gnn_candidates, top_k=max(k * 5, 50))
            merged = self._merge_candidates(gnn_candidates, semantic_candidates or [])

        merged = sorted(merged, key=lambda candidate: candidate.score, reverse=True)
        if self.reranker:
            return self.reranker.rerank(merged, k=k)
        return merged[:k]

