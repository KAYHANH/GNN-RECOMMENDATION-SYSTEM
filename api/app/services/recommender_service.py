from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any
import json

import pandas as pd

from api.app.config import Settings
from models.common import RecommendationCandidate
from models.hybrid import HybridRecommender
from models.lightgcn import LightGCNRecommender
from models.reranker import LightGBMReranker
from models.semantic_engine import SemanticEngine


SAMPLE_CATALOG = [
    {
        "article_id": "0926246001",
        "prod_name": "Summer nights dress",
        "colour_group_name": "Black",
        "product_group_name": "Garment Full body",
        "detail_desc": "Short lace dress with a concealed zip and flared skirt.",
    },
    {
        "article_id": "0496762004",
        "prod_name": "Summer strap dress",
        "colour_group_name": "Black",
        "product_group_name": "Garment Full body",
        "detail_desc": "Short dress in soft jersey with a V-neck and flared skirt.",
    },
    {
        "article_id": "0926502001",
        "prod_name": "Hudson Wide Leg Denim",
        "colour_group_name": "Blue",
        "product_group_name": "Garment Lower body",
        "detail_desc": "5-pocket jeans with a high waist and wide legs.",
    },
    {
        "article_id": "0833622001",
        "prod_name": "WIDELEG LIGHT WEIGHT",
        "colour_group_name": "Light Blue",
        "product_group_name": "Garment Lower body",
        "detail_desc": "Cropped lightweight jeans with wide straight-cut legs.",
    },
    {
        "article_id": "0110065001",
        "prod_name": "OP T-shirt (Idro)",
        "colour_group_name": "Black",
        "product_group_name": "Underwear",
        "detail_desc": "Microfibre T-shirt bra with lightly padded cups.",
    },
    {
        "article_id": "0781135002",
        "prod_name": "Wide H.W Ankle",
        "colour_group_name": "Blue",
        "product_group_name": "Garment Lower body",
        "detail_desc": "Ankle-length jeans in washed stretch denim with wide legs.",
    },
]


class RecommenderService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.articles_df = self._load_articles()
        self.article_records = self.articles_df.to_dict(orient="records")
        self.catalog_by_id = {
            str(row["article_id"]).zfill(10): {key: value for key, value in row.items() if pd.notna(value)}
            for row in self.article_records
        }
        self.runtime_stats = self._load_runtime_stats()
        self._transactions_df: pd.DataFrame | None = None
        self._transactions_loaded = False
        self._gnn_engine: LightGCNRecommender | None = None
        self._gnn_engine_loaded = False
        self._semantic_engine: SemanticEngine | None = None
        self._semantic_engine_loaded = False
        self._reranker: LightGBMReranker | None = None
        self._reranker_loaded = False

    @property
    def transactions_df(self) -> pd.DataFrame:
        if not self._transactions_loaded:
            self._transactions_df = self._load_transactions()
            self._transactions_loaded = True
        return self._transactions_df if self._transactions_df is not None else pd.DataFrame()

    @property
    def gnn_engine(self) -> LightGCNRecommender | None:
        if not self._gnn_engine_loaded:
            self._gnn_engine = self._load_gnn_engine()
            self._gnn_engine_loaded = True
        return self._gnn_engine

    @property
    def semantic_engine(self) -> SemanticEngine | None:
        if not self._semantic_engine_loaded:
            self._semantic_engine = self._load_semantic_engine()
            self._semantic_engine_loaded = True
        return self._semantic_engine

    @property
    def reranker(self) -> LightGBMReranker | None:
        if not self._reranker_loaded:
            self._reranker = self._load_reranker()
            self._reranker_loaded = True
        return self._reranker

    @property
    def hybrid(self) -> HybridRecommender:
        return HybridRecommender(
            gnn_engine=self.gnn_engine,
            semantic_engine=self.semantic_engine,
            reranker=self.reranker,
        )

    def artifact_status(self) -> dict[str, bool]:
        return {
            "articles_ready": self.settings.articles_path.exists(),
            "transactions_ready": self.settings.transactions_path.exists(),
            "images_ready": self.settings.raw_images_dir.exists(),
            "semantic_index_ready": self.settings.semantic_index_path.exists(),
            "semantic_ids_ready": self.settings.semantic_ids_path.exists(),
            "user_embeddings_ready": self.settings.user_embeddings_path.exists(),
            "item_embeddings_ready": self.settings.item_embeddings_path.exists(),
            "user_mapping_ready": self.settings.user_mapping_path.exists(),
            "item_mapping_ready": self.settings.item_mapping_path.exists(),
            "reranker_ready": self.settings.reranker_path.exists(),
        }

    def service_snapshot(self) -> dict[str, Any]:
        artifact_status = self.artifact_status()
        graph_ready = (
            artifact_status["user_embeddings_ready"]
            and artifact_status["item_embeddings_ready"]
            and artifact_status["user_mapping_ready"]
            and artifact_status["item_mapping_ready"]
        )
        semantic_ready = artifact_status["semantic_index_ready"] and artifact_status["semantic_ids_ready"]
        article_count = int(len(self.articles_df))
        interaction_count = int(self.runtime_stats.get("interaction_count", 0))
        customer_count = int(self.runtime_stats.get("customer_count", 0))
        image_count = (
            int(self.articles_df["image_available"].fillna(False).astype(bool).sum())
            if "image_available" in self.articles_df.columns
            else 0
        )
        sample_data_active = not self.settings.articles_path.exists() or not self.settings.transactions_path.exists()

        return {
            "artifacts": artifact_status,
            "engines": {
                "graph_ready": graph_ready,
                "semantic_ready": semantic_ready,
                "reranker_ready": artifact_status["reranker_ready"],
                "fallback_active": sample_data_active or not graph_ready or not semantic_ready,
            },
            "catalog": {
                "article_count": article_count,
                "interaction_count": interaction_count,
                "customer_count": customer_count,
                "image_count": image_count,
                "sample_data_active": sample_data_active,
            },
        }

    def readiness_status(self) -> str:
        snapshot = self.service_snapshot()
        if snapshot["engines"]["fallback_active"]:
            return "degraded"
        return "ready"

    def _load_runtime_stats(self) -> dict[str, int]:
        if self.settings.runtime_stats_path.exists():
            try:
                with open(self.settings.runtime_stats_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                return {
                    "interaction_count": int(payload.get("interaction_count", 0)),
                    "customer_count": int(payload.get("customer_count", 0)),
                }
            except Exception:
                return {}
        return {}

    def _load_articles(self) -> pd.DataFrame:
        if self.settings.articles_path.exists():
            frame = pd.read_csv(self.settings.articles_path, dtype={"article_id": str})
        else:
            frame = pd.DataFrame(SAMPLE_CATALOG)

        frame["article_id"] = frame["article_id"].astype(str).str.zfill(10)
        return self._with_image_columns(frame)

    def _with_image_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()

        if "image_relative_path" not in enriched.columns:
            enriched["image_relative_path"] = enriched["article_id"].astype(str).map(self._relative_image_path)

        if "image_local_path" not in enriched.columns:
            if self.settings.raw_images_dir.exists():
                enriched["image_local_path"] = enriched["image_relative_path"].map(
                    lambda relative_path: str((self.settings.raw_images_dir / relative_path).resolve())
                    if (self.settings.raw_images_dir / relative_path).exists()
                    else ""
                )
            else:
                enriched["image_local_path"] = ""

        if "image_external_url" not in enriched.columns:
            if self.settings.external_image_base_url:
                enriched["image_external_url"] = enriched["image_relative_path"].map(
                    lambda relative_path: f"{self.settings.external_image_base_url}/{relative_path}"
                )
            else:
                enriched["image_external_url"] = ""

        enriched["image_available"] = (
            enriched["image_local_path"].astype(str).str.len() > 0
        ) | (
            enriched["image_external_url"].astype(str).str.len() > 0
        )

        return enriched

    @staticmethod
    def _normalize_article_id(article_id: str | int) -> str:
        return str(article_id).strip().zfill(10)

    @classmethod
    def _relative_image_path(cls, article_id: str | int) -> str:
        normalized_article_id = cls._normalize_article_id(article_id)
        return f"{normalized_article_id[:3]}/{normalized_article_id}.jpg"

    def _load_transactions(self) -> pd.DataFrame:
        if not self.settings.load_transactions_at_runtime:
            return pd.DataFrame(columns=["customer_id", "article_id", "t_dat"])

        if self.settings.transactions_path.exists():
            frame = pd.read_csv(
                self.settings.transactions_path,
                dtype={"customer_id": str, "article_id": str},
            )
            if "t_dat" in frame.columns:
                frame["t_dat"] = pd.to_datetime(frame["t_dat"])
            return frame

        return pd.DataFrame(
            [
                {"customer_id": "demo-customer", "article_id": "0496762004", "t_dat": "2025-08-27"},
                {"customer_id": "demo-customer", "article_id": "0926502001", "t_dat": "2025-08-29"},
                {"customer_id": "demo-customer", "article_id": "0110065001", "t_dat": "2025-08-30"},
            ]
        )

    def _load_gnn_engine(self) -> LightGCNRecommender | None:
        required_paths = [
            self.settings.user_embeddings_path,
            self.settings.item_embeddings_path,
            self.settings.user_mapping_path,
            self.settings.item_mapping_path,
        ]
        if not all(path.exists() for path in required_paths):
            return None

        try:
            return LightGCNRecommender.from_artifacts(
                user_embeddings_path=self.settings.user_embeddings_path,
                item_embeddings_path=self.settings.item_embeddings_path,
                user_mapping_path=self.settings.user_mapping_path,
                item_mapping_path=self.settings.item_mapping_path,
                articles_path=self.settings.articles_path if self.settings.articles_path.exists() else None,
                interactions_path=(
                    self.settings.transactions_path
                    if self.settings.load_transactions_at_runtime and self.settings.transactions_path.exists()
                    else None
                ),
            )
        except Exception:
            return None

    def _load_semantic_engine(self) -> SemanticEngine | None:
        required_paths = [self.settings.semantic_index_path, self.settings.semantic_ids_path]
        if not all(path.exists() for path in required_paths):
            return None

        try:
            return SemanticEngine.from_artifacts(
                index_path=self.settings.semantic_index_path,
                article_ids_path=self.settings.semantic_ids_path,
                articles_path=self.settings.articles_path if self.settings.articles_path.exists() else None,
                model_name=self.settings.semantic_model_name,
            )
        except Exception:
            return None

    def _load_reranker(self) -> LightGBMReranker | None:
        if not self.settings.reranker_path.exists():
            return None
        try:
            return LightGBMReranker.load(self.settings.reranker_path)
        except Exception:
            return None

    def _catalog_lookup(self, article_id: str) -> dict[str, Any]:
        normalized_article_id = self._normalize_article_id(article_id)
        metadata = self.catalog_by_id.get(normalized_article_id)
        if metadata is None:
            return {
                "article_id": normalized_article_id,
                "image_relative_path": self._relative_image_path(normalized_article_id),
                "image_available": False,
            }
        return dict(metadata)

    def article_image_path(self, article_id: str) -> Path | None:
        article = self._catalog_lookup(article_id)
        local_path = article.get("image_local_path")

        if isinstance(local_path, str) and local_path:
            path = Path(local_path)
            if path.exists():
                return path

        relative_path = article.get("image_relative_path")
        if isinstance(relative_path, str) and relative_path:
            candidate = self.settings.raw_images_dir / relative_path
            if candidate.exists():
                return candidate

        fallback = self.settings.raw_images_dir / self._relative_image_path(article_id)
        if fallback.exists():
            return fallback

        return None

    def article_image_url(self, article_id: str) -> str | None:
        article = self._catalog_lookup(article_id)
        external_url = article.get("image_external_url")
        if isinstance(external_url, str) and external_url:
            return external_url
        return None

    def _candidate_to_dict(self, candidate: RecommendationCandidate) -> dict[str, Any]:
        payload = asdict(candidate)
        catalog_metadata = self._catalog_lookup(candidate.article_id)
        merged_metadata = {
            **catalog_metadata,
            **payload.get("metadata", {}),
        }
        payload["metadata"] = merged_metadata
        return payload

    def get_article(self, article_id: str) -> dict[str, Any] | None:
        metadata = self._catalog_lookup(article_id)
        if not metadata:
            return None

        return {
            "article_id": self._normalize_article_id(article_id),
            "score": 1.0,
            "source": "anchor",
            "metadata": metadata,
        }

    @staticmethod
    def _merge_candidate_groups(*candidate_groups: list[RecommendationCandidate]) -> list[RecommendationCandidate]:
        merged: dict[str, RecommendationCandidate] = {}
        sources: dict[str, set[str]] = defaultdict(set)

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

    def _anchor_candidate(self, article_id: str) -> RecommendationCandidate | None:
        metadata = self._catalog_lookup(article_id)
        normalized_article_id = self._normalize_article_id(article_id)
        if not metadata:
            return None

        return RecommendationCandidate(
            article_id=normalized_article_id,
            score=1.0,
            source="anchor",
            metadata=metadata,
            features={},
        )

    def _target_profile_text(self, article_id: str) -> str:
        metadata = self._catalog_lookup(article_id)
        if not metadata:
            return ""

        fields = [
            metadata.get("prod_name", ""),
            metadata.get("product_type_name", ""),
            metadata.get("product_group_name", ""),
            metadata.get("department_name", ""),
            metadata.get("section_name", ""),
            metadata.get("colour_group_name", ""),
            metadata.get("detail_desc", ""),
        ]
        return ". ".join(str(field).strip() for field in fields if str(field).strip())

    def _semantic_related(self, article_id: str, k: int) -> list[RecommendationCandidate]:
        if self.semantic_engine is None:
            return []

        target_text = self._target_profile_text(article_id)
        if not target_text:
            return []

        try:
            results = self.semantic_engine.search(target_text, k=max(k * 4, 24))
        except Exception:
            return []

        normalized_article_id = self._normalize_article_id(article_id)
        filtered = [candidate for candidate in results if candidate.article_id != normalized_article_id]
        return filtered[:k]

    def _graph_related(self, article_id: str, k: int) -> list[RecommendationCandidate]:
        if self.gnn_engine is None:
            return []

        try:
            return self.gnn_engine.similar_items(article_id, k=max(k * 4, 24))[:k]
        except Exception:
            return []

    def _metadata_similarity_value(self, anchor_article_id: str, candidate_article_id: str) -> float:
        anchor = self._catalog_lookup(anchor_article_id)
        candidate = self._catalog_lookup(candidate_article_id)
        if not anchor or not candidate:
            return 0.0

        score = 0.0
        if str(anchor.get("product_group_name", "")).strip().lower() == str(candidate.get("product_group_name", "")).strip().lower():
            score += 1.0
        if str(anchor.get("product_type_name", "")).strip().lower() == str(candidate.get("product_type_name", "")).strip().lower():
            score += 0.8
        if str(anchor.get("department_name", "")).strip().lower() == str(candidate.get("department_name", "")).strip().lower():
            score += 0.5
        if str(anchor.get("colour_group_name", "")).strip().lower() == str(candidate.get("colour_group_name", "")).strip().lower():
            score += 0.3
        if str(anchor.get("section_name", "")).strip().lower() == str(candidate.get("section_name", "")).strip().lower():
            score += 0.2
        return score

    @staticmethod
    def _normalized_score_map(candidates: list[RecommendationCandidate]) -> dict[str, float]:
        if not candidates:
            return {}

        values = [float(candidate.score) for candidate in candidates]
        min_value = min(values)
        max_value = max(values)

        if abs(max_value - min_value) < 1e-12:
            return {candidate.article_id: 1.0 for candidate in candidates}

        return {
            candidate.article_id: (float(candidate.score) - min_value) / (max_value - min_value)
            for candidate in candidates
        }

    def _filter_aligned_candidates(
        self,
        anchor_article_id: str,
        candidates: list[RecommendationCandidate],
        *,
        minimum_similarity: float = 1.0,
        min_keep: int = 6,
    ) -> list[RecommendationCandidate]:
        aligned = [
            candidate
            for candidate in candidates
            if self._metadata_similarity_value(anchor_article_id, candidate.article_id) >= minimum_similarity
        ]
        return aligned if len(aligned) >= min_keep else candidates

    def _rank_weighted_candidates(
        self,
        anchor_article_id: str,
        *,
        weighted_groups: list[tuple[list[RecommendationCandidate], float]],
        metadata_weight: float = 0.0,
        backfill_groups: list[list[RecommendationCandidate]] | None = None,
    ) -> list[RecommendationCandidate]:
        merged: dict[str, RecommendationCandidate] = {}
        sources: dict[str, set[str]] = defaultdict(set)

        for candidates, weight in weighted_groups:
            score_map = self._normalized_score_map(candidates)
            for candidate in candidates:
                existing = merged.get(candidate.article_id)
                if existing is None:
                    existing = RecommendationCandidate(
                        article_id=candidate.article_id,
                        score=0.0,
                        source=candidate.source,
                        metadata=candidate.metadata,
                        features=dict(candidate.features),
                    )
                    merged[candidate.article_id] = existing
                else:
                    existing.features.update(candidate.features)
                    existing.metadata = existing.metadata or candidate.metadata

                existing.score += weight * score_map.get(candidate.article_id, 0.0)
                sources[candidate.article_id].add(candidate.source)

        if metadata_weight > 0 and merged:
            metadata_scores = {
                article_id: self._metadata_similarity_value(anchor_article_id, article_id)
                for article_id in merged
            }
            max_metadata_score = max(metadata_scores.values(), default=0.0)
            if max_metadata_score > 0:
                for article_id, bonus in metadata_scores.items():
                    merged[article_id].score += metadata_weight * (bonus / max_metadata_score)

        ordered = sorted(merged.values(), key=lambda candidate: candidate.score, reverse=True)

        if backfill_groups:
            seen = {candidate.article_id for candidate in ordered}
            backfill_score = (ordered[-1].score if ordered else 0.0) - 1.0
            for group in backfill_groups:
                for candidate in group:
                    if candidate.article_id in seen:
                        continue
                    seen.add(candidate.article_id)
                    ordered.append(
                        RecommendationCandidate(
                            article_id=candidate.article_id,
                            score=backfill_score,
                            source=candidate.source,
                            metadata=candidate.metadata,
                            features=dict(candidate.features),
                        )
                    )
                    backfill_score -= 1e-6

        for candidate in ordered:
            candidate.source = "+".join(sorted(sources.get(candidate.article_id, {candidate.source})))

        return ordered

    def _co_purchase_related(self, article_id: str, k: int) -> list[RecommendationCandidate]:
        if self.transactions_df.empty:
            return []

        normalized_article_id = self._normalize_article_id(article_id)
        buyers = self.transactions_df[self.transactions_df["article_id"].astype(str) == normalized_article_id]["customer_id"]
        if buyers.empty:
            return []

        co_purchases = self.transactions_df[
            self.transactions_df["customer_id"].isin(buyers.astype(str))
            & (self.transactions_df["article_id"].astype(str) != normalized_article_id)
        ]
        if co_purchases.empty:
            return []

        counts = co_purchases["article_id"].astype(str).value_counts().head(k * 4)
        max_count = float(counts.iloc[0]) if not counts.empty else 1.0

        candidates: list[RecommendationCandidate] = []
        for candidate_article_id, count in counts.items():
            score = float(count) / max_count
            candidates.append(
                RecommendationCandidate(
                    article_id=str(candidate_article_id).zfill(10),
                    score=score,
                    source="co-purchase",
                    metadata=self._catalog_lookup(str(candidate_article_id)),
                    features={"co_purchase_score": score},
                )
            )

        return candidates[:k]

    def _metadata_related(self, article_id: str, k: int) -> list[RecommendationCandidate]:
        target = self._catalog_lookup(article_id)
        if not target:
            return []

        normalized_article_id = self._normalize_article_id(article_id)
        category = str(target.get("product_group_name", "")).strip().lower()
        color = str(target.get("colour_group_name", "")).strip().lower()
        department = str(target.get("department_name", "")).strip().lower()

        candidates: list[RecommendationCandidate] = []
        for row in self.article_records:
            candidate_article_id = str(row.get("article_id", "")).zfill(10)
            if candidate_article_id == normalized_article_id:
                continue

            score = 0.0
            if category and str(row.get("product_group_name", "")).strip().lower() == category:
                score += 1.0
            if color and str(row.get("colour_group_name", "")).strip().lower() == color:
                score += 0.7
            if department and str(row.get("department_name", "")).strip().lower() == department:
                score += 0.4

            if score <= 0:
                continue

            candidates.append(
                RecommendationCandidate(
                    article_id=candidate_article_id,
                    score=score,
                    source="metadata",
                    metadata=row,
                    features={"metadata_similarity_score": score},
                )
            )

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates[:k]

    def _fallback_search(self, query: str, k: int) -> list[RecommendationCandidate]:
        query_tokens = set(query.lower().split())
        candidates: list[RecommendationCandidate] = []

        for row in self.article_records:
            haystack = " ".join(str(value).lower() for value in row.values())
            overlap = len(query_tokens & set(haystack.split()))
            if overlap == 0:
                continue
            candidates.append(
                RecommendationCandidate(
                    article_id=str(row["article_id"]),
                    score=float(overlap),
                    source="lexical",
                    metadata=row,
                    features={"semantic_similarity_score": float(overlap)},
                )
            )

        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:k]

    def _fallback_recommend(self, customer_id: str, k: int) -> list[RecommendationCandidate]:
        history_ids = set(
            self.transactions_df[self.transactions_df["customer_id"] == str(customer_id)]["article_id"].astype(str).tolist()
        )

        candidates: list[RecommendationCandidate] = []
        for rank, row in enumerate(self.article_records, start=1):
            article_id = str(row["article_id"])
            if article_id in history_ids:
                continue
            score = 1.0 / rank
            candidates.append(
                RecommendationCandidate(
                    article_id=article_id,
                    score=score,
                    source="fallback",
                    metadata=row,
                    features={"item_global_popularity": score},
                )
            )
        return candidates[:k]

    def _profile_text(self, customer_id: str) -> str:
        history = self.transactions_df[self.transactions_df["customer_id"] == str(customer_id)]
        if history.empty:
            return "new customer fashion profile"

        merged = history.merge(self.articles_df, on="article_id", how="left")
        parts = []
        for _, row in merged.tail(5).iterrows():
            prod_name = str(row.get("prod_name", "")).strip()
            color = str(row.get("colour_group_name", "")).strip()
            category = str(row.get("product_group_name", "")).strip()
            parts.append(" ".join(part for part in [prod_name, color, category] if part))
        return ". ".join(parts) or "fashion shopper"

    def recommend(self, customer_id: str, k: int = 12, mode: str = "hybrid") -> list[dict[str, Any]]:
        profile_text = self._profile_text(customer_id)

        if self.gnn_engine:
            try:
                candidates = self.hybrid.recommend(customer_id=customer_id, k=k, mode=mode, profile_text=profile_text)
            except Exception:
                candidates = self._fallback_recommend(customer_id, k)
        else:
            candidates = self._fallback_recommend(customer_id, k)

        return [self._candidate_to_dict(candidate) for candidate in candidates[:k]]

    def search(self, query: str, k: int = 12) -> list[dict[str, Any]]:
        if self.semantic_engine:
            try:
                results = self.semantic_engine.search(query, k=k)
            except Exception:
                results = self._fallback_search(query, k)
        else:
            results = self._fallback_search(query, k)

        return [self._candidate_to_dict(candidate) for candidate in results[:k]]

    def related(self, article_id: str, k: int = 12, mode: str = "hybrid") -> list[dict[str, Any]]:
        semantic_candidates = self._semantic_related(article_id, k=max(k * 6, 48))
        graph_candidates = self._graph_related(article_id, k=max(k * 8, 64))
        co_purchase_candidates = self._co_purchase_related(article_id, k=max(k * 4, 24))
        metadata_candidates = self._metadata_related(article_id, k=max(k * 4, 24))

        aligned_graph_candidates = self._filter_aligned_candidates(
            article_id,
            graph_candidates,
            minimum_similarity=1.0,
            min_keep=max(6, min(k, 10)),
        )

        if mode == "gnn":
            merged = self._rank_weighted_candidates(
                article_id,
                weighted_groups=[
                    (aligned_graph_candidates, 0.8),
                    (co_purchase_candidates, 0.2),
                ],
                metadata_weight=0.2,
                backfill_groups=[metadata_candidates],
            )
        elif mode == "semantic":
            merged = self._rank_weighted_candidates(
                article_id,
                weighted_groups=[
                    (semantic_candidates, 0.9),
                ],
                metadata_weight=0.1,
                backfill_groups=[metadata_candidates],
            )
        else:
            merged = self._rank_weighted_candidates(
                article_id,
                weighted_groups=[
                    (semantic_candidates, 0.45),
                    (aligned_graph_candidates, 0.35),
                    (co_purchase_candidates, 0.2),
                ],
                metadata_weight=0.1,
                backfill_groups=[metadata_candidates],
            )

        normalized_article_id = self._normalize_article_id(article_id)
        filtered = [candidate for candidate in merged if candidate.article_id != normalized_article_id]
        return [self._candidate_to_dict(candidate) for candidate in filtered[:k]]

    def discover(self, query: str, k: int = 12, mode: str = "hybrid") -> dict[str, Any]:
        search_results = self.search(query, k=max(k, 8))
        if not search_results:
            return {"anchor": None, "recommendations": []}

        anchor = search_results[0]
        recommendations = self.related(anchor["article_id"], k=k, mode=mode)
        return {
            "anchor": anchor,
            "recommendations": recommendations,
        }

    def explain(self, customer_id: str, article_id: str) -> list[str]:
        reasons: list[str] = []
        history = self.transactions_df[self.transactions_df["customer_id"] == str(customer_id)]
        target = self._catalog_lookup(article_id)

        if not history.empty:
            recent_ids = history["article_id"].astype(str).tail(3).tolist()
            recent_names = [self._catalog_lookup(item_id).get("prod_name", item_id) for item_id in recent_ids]
            if recent_names:
                reasons.append(f"Because you recently bought items like {', '.join(map(str, recent_names))}.")

        color = target.get("colour_group_name")
        category = target.get("product_group_name")
        if color:
            reasons.append(f"It matches your recent interest in {color} tones.")
        if category:
            reasons.append(f"It belongs to the {category} category seen in your shopping history.")

        reasons.append("Its hybrid score combines collaborative signals with semantic similarity.")
        return reasons

    def explain_related(self, anchor_article_id: str, article_id: str) -> list[str]:
        anchor = self._catalog_lookup(anchor_article_id)
        target = self._catalog_lookup(article_id)
        reasons: list[str] = []

        anchor_name = str(anchor.get("prod_name", anchor_article_id))
        target_name = str(target.get("prod_name", article_id))

        anchor_category = str(anchor.get("product_group_name", "")).strip()
        target_category = str(target.get("product_group_name", "")).strip()
        if anchor_category and anchor_category == target_category:
            reasons.append(f"{target_name} shares the same product group as {anchor_name}: {anchor_category}.")

        anchor_color = str(anchor.get("colour_group_name", "")).strip()
        target_color = str(target.get("colour_group_name", "")).strip()
        if anchor_color and anchor_color == target_color:
            reasons.append(f"Both items sit in the same color family: {anchor_color}.")

        buyers = self.transactions_df[self.transactions_df["article_id"].astype(str) == self._normalize_article_id(anchor_article_id)]
        if not buyers.empty:
            related_buyers = self.transactions_df[
                self.transactions_df["customer_id"].isin(buyers["customer_id"].astype(str))
                & (self.transactions_df["article_id"].astype(str) == self._normalize_article_id(article_id))
            ]
            if not related_buyers.empty:
                reasons.append("The items appear together in customer purchase histories.")

        reasons.append("The recommendation score is built from model similarity plus catalog-level matching signals.")
        return reasons
