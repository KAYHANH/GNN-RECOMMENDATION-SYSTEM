from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _get_csv(name: str, default: str) -> tuple[str, ...]:
    raw_value = os.getenv(name, default)
    values = [item.strip() for item in raw_value.split(",")]
    return tuple(value for value in values if value)


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    environment: str
    log_level: str
    artifacts_dir: Path
    data_dir: Path
    raw_images_dir: Path
    external_image_base_url: str
    semantic_model_name: str
    cors_origins: tuple[str, ...]
    docs_enabled: bool
    load_transactions_at_runtime: bool

    @property
    def docs_url(self) -> str | None:
        return "/docs" if self.docs_enabled else None

    @property
    def redoc_url(self) -> str | None:
        return "/redoc" if self.docs_enabled else None

    @property
    def articles_path(self) -> Path:
        return self.data_dir / "articles_cleaned.csv"

    @property
    def transactions_path(self) -> Path:
        return self.data_dir / "transactions_cleaned.csv"

    @property
    def semantic_index_path(self) -> Path:
        return self.artifacts_dir / "semantic_faiss_index.bin"

    @property
    def semantic_ids_path(self) -> Path:
        return self.artifacts_dir / "article_ids.csv"

    @property
    def runtime_stats_path(self) -> Path:
        return self.artifacts_dir / "runtime_stats.json"

    @property
    def user_embeddings_path(self) -> Path:
        return self.artifacts_dir / "user_embeddings.npy"

    @property
    def item_embeddings_path(self) -> Path:
        return self.artifacts_dir / "item_embeddings.npy"

    @property
    def user_mapping_path(self) -> Path:
        return self.artifacts_dir / "user_mapping.json"

    @property
    def item_mapping_path(self) -> Path:
        return self.artifacts_dir / "item_mapping.json"

    @property
    def reranker_path(self) -> Path:
        return self.artifacts_dir / "lightgbm_reranker.pkl"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "Fashion Recommender API"),
        app_version=os.getenv("APP_VERSION", "0.2.0"),
        environment=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        artifacts_dir=PROJECT_ROOT / os.getenv("ARTIFACTS_DIR", "artifacts"),
        data_dir=PROJECT_ROOT / os.getenv("DATA_DIR", "data"),
        raw_images_dir=PROJECT_ROOT / os.getenv("RAW_IMAGES_DIR", "data/raw/images"),
        external_image_base_url=os.getenv(
            "EXTERNAL_IMAGE_BASE_URL",
            "https://qdrant-nextjs-demo-product-images.s3.us-east-1.amazonaws.com/images",
        ).rstrip("/"),
        semantic_model_name=os.getenv("SEMANTIC_MODEL_NAME", "all-MiniLM-L6-v2"),
        cors_origins=_get_csv(
            "CORS_ORIGINS",
            "*",
        ),
        docs_enabled=_get_bool("ENABLE_DOCS", True),
        load_transactions_at_runtime=_get_bool(
            "LOAD_TRANSACTIONS_AT_RUNTIME",
            os.getenv("APP_ENV", "development").lower() != "production",
        ),
    )
