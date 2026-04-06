from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.common import RecommendationCandidate

try:
    import faiss
except ImportError:  # pragma: no cover - optional at inspection time
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional at inspection time
    SentenceTransformer = None


class SemanticEngine:
    def __init__(
        self,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        model: Any | None = None,
        index: Any | None = None,
        article_ids: list[str] | None = None,
        articles_df: pd.DataFrame | None = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.model = model
        self.index = index
        self.article_ids = article_ids or []
        self.articles_df = articles_df.copy() if articles_df is not None else None
        self.normalize = normalize

        if self.articles_df is not None and "article_id" in self.articles_df.columns:
            self.articles_df["article_id"] = self.articles_df["article_id"].astype(str).str.zfill(10)

    @classmethod
    def from_artifacts(
        cls,
        *,
        index_path: str | Path,
        article_ids_path: str | Path,
        articles_path: str | Path | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> "SemanticEngine":
        if faiss is None:
            raise ImportError("faiss-cpu is required to load semantic search artifacts.")

        index = faiss.read_index(str(index_path))

        if str(article_ids_path).lower().endswith(".csv"):
            article_ids = pd.read_csv(article_ids_path, dtype={"article_id": str})["article_id"].astype(str).tolist()
        else:
            article_ids = np.load(article_ids_path, allow_pickle=True).astype(str).tolist()

        articles_df = pd.read_csv(articles_path, dtype={"article_id": str}) if articles_path else None
        return cls(
            model_name=model_name,
            index=index,
            article_ids=article_ids,
            articles_df=articles_df,
        )

    def _get_model(self) -> Any:
        if self.model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required to encode text queries.")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms
        return embeddings

    def build_index(
        self,
        articles_df: pd.DataFrame,
        *,
        text_column: str = "text_for_embedding",
        article_id_column: str = "article_id",
    ) -> None:
        if faiss is None:
            raise ImportError("faiss-cpu is required to build a semantic search index.")

        frame = articles_df.copy()
        frame[article_id_column] = frame[article_id_column].astype(str).str.zfill(10)
        embeddings = self.encode(frame[text_column].fillna("").astype(str).tolist())
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.index = index
        self.article_ids = frame[article_id_column].tolist()
        self.articles_df = frame

    def _metadata_for_article(self, article_id: str) -> dict[str, Any]:
        if self.articles_df is None:
            return {}
        rows = self.articles_df[self.articles_df["article_id"] == str(article_id)]
        if rows.empty:
            return {}
        return {key: value for key, value in rows.iloc[0].to_dict().items() if pd.notna(value)}

    def search(self, query: str, *, k: int = 12) -> list[RecommendationCandidate]:
        if self.index is None:
            return []

        query_embedding = self.encode([query])
        scores, indices = self.index.search(query_embedding, k)

        results: list[RecommendationCandidate] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.article_ids):
                continue
            article_id = self.article_ids[int(idx)]
            results.append(
                RecommendationCandidate(
                    article_id=article_id,
                    score=float(score),
                    source="semantic",
                    metadata=self._metadata_for_article(article_id),
                    features={"semantic_similarity_score": float(score)},
                )
            )
        return results

    def score_candidates(
        self,
        profile_text: str,
        candidates: list[RecommendationCandidate],
        *,
        top_k: int | None = None,
    ) -> list[RecommendationCandidate]:
        if not candidates:
            return []
        if self.articles_df is None:
            return candidates[:top_k] if top_k else candidates

        ranked = {candidate.article_id: candidate for candidate in candidates}
        semantic_hits = self.search(profile_text, k=max(len(candidates), top_k or len(candidates)))

        for hit in semantic_hits:
            if hit.article_id in ranked:
                ranked[hit.article_id].features["semantic_similarity_score"] = hit.score
                ranked[hit.article_id].score += hit.score

        ordered = sorted(ranked.values(), key=lambda candidate: candidate.score, reverse=True)
        return ordered[:top_k] if top_k else ordered
