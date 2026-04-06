from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

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
        self.transactions_df = self._load_transactions()
        self.gnn_engine = self._load_gnn_engine()
        self.semantic_engine = self._load_semantic_engine()
        self.reranker = self._load_reranker()
        self.hybrid = HybridRecommender(
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
        reranker_ready = bool(self.reranker and getattr(self.reranker, "is_trained", False))
        article_count = int(len(self.articles_df))
        interaction_count = int(len(self.transactions_df))
        customer_count = (
            int(self.transactions_df["customer_id"].astype(str).nunique())
            if not self.transactions_df.empty and "customer_id" in self.transactions_df.columns
            else 0
        )
        image_count = (
            int(self.articles_df["image_available"].fillna(False).astype(bool).sum())
            if "image_available" in self.articles_df.columns
            else 0
        )
        sample_data_active = not self.settings.articles_path.exists() or not self.settings.transactions_path.exists()

        return {
            "artifacts": self.artifact_status(),
            "engines": {
                "graph_ready": self.gnn_engine is not None,
                "semantic_ready": self.semantic_engine is not None,
                "reranker_ready": reranker_ready,
                "fallback_active": sample_data_active or self.gnn_engine is None or self.semantic_engine is None,
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

        if "image_available" not in enriched.columns:
            enriched["image_available"] = enriched["image_local_path"].astype(str).str.len() > 0

        return enriched

    @staticmethod
    def _normalize_article_id(article_id: str | int) -> str:
        return str(article_id).strip().zfill(10)

    @classmethod
    def _relative_image_path(cls, article_id: str | int) -> str:
        normalized_article_id = cls._normalize_article_id(article_id)
        return f"{normalized_article_id[:3]}/{normalized_article_id}.jpg"

    def _load_transactions(self) -> pd.DataFrame:
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
                interactions_path=self.settings.transactions_path if self.settings.transactions_path.exists() else None,
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
        matches = self.articles_df[self.articles_df["article_id"] == normalized_article_id]
        if matches.empty:
            return {
                "article_id": normalized_article_id,
                "image_relative_path": self._relative_image_path(normalized_article_id),
                "image_available": False,
            }
        row = matches.iloc[0].to_dict()
        return {key: value for key, value in row.items() if pd.notna(value)}

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

    def _fallback_search(self, query: str, k: int) -> list[RecommendationCandidate]:
        query_tokens = set(query.lower().split())
        candidates: list[RecommendationCandidate] = []

        for row in self.articles_df.to_dict(orient="records"):
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
        for rank, row in enumerate(self.articles_df.to_dict(orient="records"), start=1):
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

        return [asdict(candidate) for candidate in candidates[:k]]

    def search(self, query: str, k: int = 12) -> list[dict[str, Any]]:
        if self.semantic_engine:
            try:
                results = self.semantic_engine.search(query, k=k)
            except Exception:
                results = self._fallback_search(query, k)
        else:
            results = self._fallback_search(query, k)

        return [asdict(candidate) for candidate in results[:k]]

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
