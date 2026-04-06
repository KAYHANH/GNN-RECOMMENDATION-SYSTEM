from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RecommendationMode(str, Enum):
    HYBRID = "hybrid"
    GNN = "gnn"
    SEMANTIC = "semantic"


class ApiInfo(BaseModel):
    name: str
    version: str
    environment: str
    docs_url: str | None = None
    redoc_url: str | None = None


class ArtifactStatus(BaseModel):
    articles_ready: bool
    transactions_ready: bool
    semantic_index_ready: bool
    semantic_ids_ready: bool
    user_embeddings_ready: bool
    item_embeddings_ready: bool
    user_mapping_ready: bool
    item_mapping_ready: bool
    reranker_ready: bool


class EngineStatus(BaseModel):
    graph_ready: bool
    semantic_ready: bool
    reranker_ready: bool
    fallback_active: bool


class CatalogStatus(BaseModel):
    article_count: int
    interaction_count: int
    customer_count: int
    sample_data_active: bool


class ServiceSnapshot(BaseModel):
    artifacts: ArtifactStatus
    engines: EngineStatus
    catalog: CatalogStatus


class ResponseMeta(BaseModel):
    generated_at: datetime = Field(default_factory=utc_now)
    environment: str
    snapshot: ServiceSnapshot


class RootResponse(BaseModel):
    message: str
    api: ApiInfo


class HealthResponse(BaseModel):
    status: str
    api: ApiInfo
    snapshot: ServiceSnapshot | None = None


class RecommendationItem(BaseModel):
    article_id: str
    score: float
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    customer_id: str
    mode: RecommendationMode
    recommendations: list[RecommendationItem]
    meta: ResponseMeta


class SearchResponse(BaseModel):
    query: str
    results: list[RecommendationItem]
    meta: ResponseMeta


class ExplainResponse(BaseModel):
    customer_id: str
    article_id: str
    reasons: list[str]
    meta: ResponseMeta
