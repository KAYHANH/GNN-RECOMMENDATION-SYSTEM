from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from time import perf_counter
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from api.app.config import Settings, get_settings
from api.app.dependencies import get_app_settings, get_recommender_service
from api.app.observability import configure_logging
from api.app.schemas import (
    ApiInfo,
    DiscoverResponse,
    ExplainResponse,
    HealthResponse,
    RelatedExplainResponse,
    RelatedResponse,
    RecommendationMode,
    RecommendationResponse,
    ResponseMeta,
    RootResponse,
    SearchResponse,
    ServiceSnapshot,
)
from api.app.services.recommender_service import RecommenderService


settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("fashion.api")


def build_api_info(app_settings: Settings) -> ApiInfo:
    return ApiInfo(
        name=app_settings.app_name,
        version=app_settings.app_version,
        environment=app_settings.environment,
        docs_url=app_settings.docs_url,
        redoc_url=app_settings.redoc_url,
    )


def build_response_meta(app_settings: Settings, service: RecommenderService) -> ResponseMeta:
    return ResponseMeta(
        environment=app_settings.environment,
        snapshot=ServiceSnapshot.model_validate(service.service_snapshot()),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "Starting %s v%s in %s mode",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Recommendation, search, and explanation endpoints for the fashion hybrid recommender.",
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", uuid4().hex)
    started_at = perf_counter()

    response = await call_next(request)

    duration_ms = int((perf_counter() - started_at) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(duration_ms)

    logger.info(
        "%s %s -> %s (%sms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/", response_model=RootResponse, tags=["Meta"])
async def root(app_settings: Annotated[Settings, Depends(get_app_settings)]) -> RootResponse:
    return RootResponse(
        message="Fashion Recommender API is running.",
        api=build_api_info(app_settings),
    )


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health(app_settings: Annotated[Settings, Depends(get_app_settings)]) -> HealthResponse:
    return HealthResponse(
        status="ok",
        api=build_api_info(app_settings),
    )


@app.get("/health/ready", response_model=HealthResponse, tags=["Meta"])
async def readiness(
    app_settings: Annotated[Settings, Depends(get_app_settings)],
    service: Annotated[RecommenderService, Depends(get_recommender_service)],
) -> HealthResponse:
    return HealthResponse(
        status=service.readiness_status(),
        api=build_api_info(app_settings),
        snapshot=ServiceSnapshot.model_validate(service.service_snapshot()),
    )


@app.get("/catalog/images/{article_id}", tags=["Catalog"])
async def catalog_image(
    article_id: str,
    service: Annotated[RecommenderService, Depends(get_recommender_service)],
):
    image_path = service.article_image_path(article_id)
    if image_path is not None:
        return FileResponse(image_path, media_type="image/jpeg")

    external_url = service.article_image_url(article_id)
    if external_url is not None:
        return RedirectResponse(url=external_url, status_code=307)

    raise HTTPException(status_code=404, detail="Image not found for the requested article.")


@app.get("/recommend/{customer_id}", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend(
    customer_id: str,
    k: Annotated[int, Query(ge=1, le=100)] = 12,
    mode: RecommendationMode = RecommendationMode.HYBRID,
    app_settings: Annotated[Settings, Depends(get_app_settings)] = None,
    service: Annotated[RecommenderService, Depends(get_recommender_service)] = None,
) -> RecommendationResponse:
    recommendations = service.recommend(customer_id=customer_id, k=k, mode=mode.value)
    return RecommendationResponse(
        customer_id=customer_id,
        mode=mode,
        recommendations=recommendations,
        meta=build_response_meta(app_settings, service),
    )


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search(
    q: Annotated[str, Query(min_length=1)],
    k: Annotated[int, Query(ge=1, le=100)] = 12,
    app_settings: Annotated[Settings, Depends(get_app_settings)] = None,
    service: Annotated[RecommenderService, Depends(get_recommender_service)] = None,
) -> SearchResponse:
    results = service.search(query=q, k=k)
    return SearchResponse(
        query=q,
        results=results,
        meta=build_response_meta(app_settings, service),
    )


@app.get("/discover", response_model=DiscoverResponse, tags=["Discovery"])
async def discover(
    q: Annotated[str, Query(min_length=1)],
    k: Annotated[int, Query(ge=1, le=100)] = 12,
    mode: RecommendationMode = RecommendationMode.HYBRID,
    app_settings: Annotated[Settings, Depends(get_app_settings)] = None,
    service: Annotated[RecommenderService, Depends(get_recommender_service)] = None,
) -> DiscoverResponse:
    payload = service.discover(query=q, k=k, mode=mode.value)
    return DiscoverResponse(
        query=q,
        anchor=payload["anchor"],
        mode=mode,
        recommendations=payload["recommendations"],
        meta=build_response_meta(app_settings, service),
    )


@app.get("/related/{article_id}", response_model=RelatedResponse, tags=["Discovery"])
async def related(
    article_id: str,
    k: Annotated[int, Query(ge=1, le=100)] = 12,
    mode: RecommendationMode = RecommendationMode.HYBRID,
    app_settings: Annotated[Settings, Depends(get_app_settings)] = None,
    service: Annotated[RecommenderService, Depends(get_recommender_service)] = None,
) -> RelatedResponse:
    anchor = service.get_article(article_id)

    return RelatedResponse(
        anchor_article_id=article_id,
        anchor=anchor,
        mode=mode,
        recommendations=service.related(article_id=article_id, k=k, mode=mode.value),
        meta=build_response_meta(app_settings, service),
    )


@app.get("/explain/{customer_id}/{article_id}", response_model=ExplainResponse, tags=["Explainability"])
async def explain(
    customer_id: str,
    article_id: str,
    app_settings: Annotated[Settings, Depends(get_app_settings)] = None,
    service: Annotated[RecommenderService, Depends(get_recommender_service)] = None,
) -> ExplainResponse:
    reasons = service.explain(customer_id=customer_id, article_id=article_id)
    return ExplainResponse(
        customer_id=customer_id,
        article_id=article_id,
        reasons=reasons,
        meta=build_response_meta(app_settings, service),
    )


@app.get("/explain-related/{anchor_article_id}/{article_id}", response_model=RelatedExplainResponse, tags=["Explainability"])
async def explain_related(
    anchor_article_id: str,
    article_id: str,
    app_settings: Annotated[Settings, Depends(get_app_settings)] = None,
    service: Annotated[RecommenderService, Depends(get_recommender_service)] = None,
) -> RelatedExplainResponse:
    return RelatedExplainResponse(
        anchor_article_id=anchor_article_id,
        article_id=article_id,
        reasons=service.explain_related(anchor_article_id=anchor_article_id, article_id=article_id),
        meta=build_response_meta(app_settings, service),
    )
