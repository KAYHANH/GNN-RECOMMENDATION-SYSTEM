from __future__ import annotations

from functools import lru_cache

from api.app.config import Settings, get_settings
from api.app.services.recommender_service import RecommenderService


@lru_cache(maxsize=1)
def get_recommender_service() -> RecommenderService:
    return RecommenderService(get_settings())


def get_app_settings() -> Settings:
    return get_settings()

