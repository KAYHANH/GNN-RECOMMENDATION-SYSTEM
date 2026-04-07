from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import pickle
import re

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

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover - optional at inspection time
    TruncatedSVD = None
    TfidfVectorizer = None


SEMANTIC_BACKEND_METADATA = "semantic_backend.json"
SEMANTIC_VECTORIZER_ARTIFACT = "semantic_vectorizer.pkl"
SEMANTIC_PROJECTOR_ARTIFACT = "semantic_projector.pkl"


class SemanticEngine:
    def __init__(
        self,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        backend: str = "sentence-transformer",
        model: Any | None = None,
        index: Any | None = None,
        article_ids: list[str] | None = None,
        articles_df: pd.DataFrame | None = None,
        normalize: bool = True,
        vectorizer: Any | None = None,
        projector: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.model = model
        self.index = index
        self.article_ids = article_ids or []
        self.articles_df = articles_df.copy() if articles_df is not None else None
        self.normalize = normalize
        self.vectorizer = vectorizer
        self.projector = projector

        if self.articles_df is not None and "article_id" in self.articles_df.columns:
            self.articles_df["article_id"] = self.articles_df["article_id"].astype(str).str.zfill(10)
        self.catalog_index = self._build_catalog_index(self.articles_df)

    @classmethod
    def from_artifacts(
        cls,
        *,
        index_path: str | Path,
        article_ids_path: str | Path,
        articles_path: str | Path | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> "SemanticEngine":
        index_path = Path(index_path)
        article_ids_path = Path(article_ids_path)
        artifact_dir = index_path.parent

        if str(article_ids_path).lower().endswith(".csv"):
            article_ids = pd.read_csv(article_ids_path, dtype={"article_id": str})["article_id"].astype(str).tolist()
        else:
            article_ids = np.load(article_ids_path, allow_pickle=True).astype(str).tolist()

        articles_df = pd.read_csv(articles_path, dtype={"article_id": str}) if articles_path else None

        backend = "sentence-transformer"
        resolved_model_name = model_name
        metadata_path = artifact_dir / SEMANTIC_BACKEND_METADATA
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            backend = str(metadata.get("backend", backend))
            resolved_model_name = str(metadata.get("model_name", resolved_model_name))
        elif (artifact_dir / SEMANTIC_VECTORIZER_ARTIFACT).exists():
            backend = "tfidf-svd"

        if faiss is None:
            return cls(
                model_name=resolved_model_name,
                backend="catalog-lexical",
                article_ids=article_ids,
                articles_df=articles_df,
            )

        index = faiss.read_index(str(index_path))

        vectorizer = None
        projector = None
        if backend == "tfidf-svd":
            if TfidfVectorizer is None:
                return cls(
                    model_name=resolved_model_name,
                    backend="catalog-lexical",
                    article_ids=article_ids,
                    articles_df=articles_df,
                )
            with open(artifact_dir / SEMANTIC_VECTORIZER_ARTIFACT, "rb") as handle:
                vectorizer = pickle.load(handle)

            projector_path = artifact_dir / SEMANTIC_PROJECTOR_ARTIFACT
            if projector_path.exists():
                with open(projector_path, "rb") as handle:
                    projector = pickle.load(handle)

        return cls(
            model_name=resolved_model_name,
            backend=backend,
            index=index,
            article_ids=article_ids,
            articles_df=articles_df,
            vectorizer=vectorizer,
            projector=projector,
        )

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    @classmethod
    def _article_text(cls, row: dict[str, Any]) -> str:
        fields = [
            row.get("prod_name", ""),
            row.get("product_type_name", ""),
            row.get("product_group_name", ""),
            row.get("department_name", ""),
            row.get("section_name", ""),
            row.get("colour_group_name", ""),
            row.get("detail_desc", ""),
        ]
        return ". ".join(str(field).strip() for field in fields if str(field).strip())

    @classmethod
    def _build_catalog_index(cls, articles_df: pd.DataFrame | None) -> list[dict[str, Any]]:
        if articles_df is None or articles_df.empty:
            return []

        records: list[dict[str, Any]] = []
        for row in articles_df.to_dict(orient="records"):
            article_id = str(row.get("article_id", "")).zfill(10)
            text = cls._article_text(row).lower()
            records.append(
                {
                    "article_id": article_id,
                    "text": text,
                    "tokens": cls._tokenize(text),
                }
            )
        return records

    def _get_model(self) -> Any:
        if self.backend != "sentence-transformer":
            return None

        if self.model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required to encode text queries.")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def _encode_sentence_transformer(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        if model is None:
            raise ValueError("SentenceTransformer backend is not available.")
        embeddings = model.encode(texts, convert_to_numpy=True).astype("float32")
        return self._normalize_embeddings(embeddings)

    def _encode_tfidf(self, texts: list[str]) -> np.ndarray:
        if self.vectorizer is None:
            raise ValueError("TF-IDF vectorizer is not loaded.")

        matrix = self.vectorizer.transform(texts)
        if self.projector is not None:
            embeddings = self.projector.transform(matrix).astype("float32")
        else:
            embeddings = matrix.astype("float32").toarray()
        return self._normalize_embeddings(embeddings)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "catalog-lexical":
            raise ValueError("Catalog lexical backend does not expose dense embeddings.")
        if self.backend == "tfidf-svd":
            return self._encode_tfidf(texts)
        return self._encode_sentence_transformer(texts)

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = embeddings.astype("float32")
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms
        return embeddings

    def fit_tfidf_encoder(
        self,
        texts: list[str],
        *,
        max_features: int = 4096,
        n_components: int = 256,
    ) -> np.ndarray:
        if TfidfVectorizer is None:
            raise ImportError("scikit-learn is required to build the TF-IDF semantic backend.")

        self.backend = "tfidf-svd"
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
        )
        matrix = self.vectorizer.fit_transform(texts)

        if TruncatedSVD is None:
            raise ImportError("scikit-learn is required to build the TF-IDF semantic backend.")

        max_components = min(n_components, max(1, matrix.shape[1] - 1))
        if max_components >= 2:
            self.projector = TruncatedSVD(n_components=max_components, random_state=42)
            embeddings = self.projector.fit_transform(matrix).astype("float32")
        else:
            self.projector = None
            embeddings = matrix.astype("float32").toarray()

        return self._normalize_embeddings(embeddings)

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
        texts = frame[text_column].fillna("").astype(str).tolist()

        if self.backend == "tfidf-svd":
            embeddings = self.fit_tfidf_encoder(texts)
        else:
            embeddings = self.encode(texts)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        self.index = index
        self.article_ids = frame[article_id_column].tolist()
        self.articles_df = frame
        self.catalog_index = self._build_catalog_index(frame)

    def _metadata_for_article(self, article_id: str) -> dict[str, Any]:
        if self.articles_df is None:
            return {}
        rows = self.articles_df[self.articles_df["article_id"] == str(article_id)]
        if rows.empty:
            return {}
        return {key: value for key, value in rows.iloc[0].to_dict().items() if pd.notna(value)}

    def _catalog_search(self, query: str, *, k: int = 12) -> list[RecommendationCandidate]:
        if not self.catalog_index:
            return []

        normalized_query = query.strip().lower()
        query_tokens = self._tokenize(normalized_query)
        if not normalized_query or not query_tokens:
            return []

        query_token_count = max(len(query_tokens), 1)
        results: list[RecommendationCandidate] = []

        for record in self.catalog_index:
            overlap = len(query_tokens & record["tokens"])
            contains_phrase = normalized_query in record["text"]
            if overlap == 0 and not contains_phrase:
                continue

            metadata = self._metadata_for_article(record["article_id"])
            prod_name = str(metadata.get("prod_name", "")).strip().lower()

            score = overlap / query_token_count
            if contains_phrase:
                score += 0.75
            if prod_name and normalized_query == prod_name:
                score += 1.25
            elif prod_name and normalized_query in prod_name:
                score += 0.5

            results.append(
                RecommendationCandidate(
                    article_id=record["article_id"],
                    score=float(score),
                    source="semantic",
                    metadata=metadata,
                    features={"semantic_similarity_score": float(score)},
                )
            )

        results.sort(key=lambda candidate: candidate.score, reverse=True)
        return results[:k]

    def search(self, query: str, *, k: int = 12) -> list[RecommendationCandidate]:
        if self.backend == "catalog-lexical":
            return self._catalog_search(query, k=k)
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
