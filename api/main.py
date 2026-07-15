"""Production API for feedback ingestion and ML operations."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import hmac
import importlib
import logging
import math
import os
from typing import Any, Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from feedback.consumer import FeedbackConsumer
from feedback.event_handlers import FeedbackHandler
from feedback.interactions import INTERACTIONS, get_interaction, normalize_interaction
from feedback.producer import FeedbackProducer, create_redis_client
from feedback.settings import FeedbackSettings

logger = logging.getLogger("pipeline.api")

producer: FeedbackProducer | None = None
consumer: FeedbackConsumer | None = None
feedback_handler: FeedbackHandler | None = None
retrieval_engine: Any | None = None
onboarding_pipeline: Any | None = None
repo_embedding_pipeline: Any | None = None


class FeedbackRequest(BaseModel):
    event_id: str = Field(..., min_length=1, max_length=128)
    user_id: UUID
    repo_id: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1)
    occurred_at: datetime
    schema_version: Literal[1] = 1
    dwell_seconds: float | None = Field(default=None, ge=0)


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    is_cold_start: bool = False


class OnboardingRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    github_username: str | None = None
    username: str | None = None
    full_name: str | None = None
    bio: str | None = None
    interests: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    avatar_url: str | None = None


class EmbedRepoRequest(BaseModel):
    repo_id: str = Field(..., min_length=1)
    github_repo: str = Field(..., min_length=1)
    github_repo_url: str | None = None
    description: str | None = None
    primary_language: str | None = None
    languages: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    readme_summary: str | None = None
    star_count: int = Field(default=0, ge=0)
    fork_count: int = Field(default=0, ge=0)
    open_issues_count: int = Field(default=0, ge=0)
    created_at: str | None = None
    updated_at: str | None = None


def _build_feedback_runtime(
    settings: FeedbackSettings,
) -> tuple[FeedbackProducer, FeedbackConsumer, FeedbackHandler]:
    redis_client = create_redis_client(settings)
    qdrant = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=30.0,
    )
    handler = FeedbackHandler(qdrant_client=qdrant, settings=settings)
    stream_producer = FeedbackProducer(redis_client=redis_client, settings=settings)
    stream_consumer = FeedbackConsumer(
        handler=handler, redis_client=redis_client, settings=settings
    )
    return stream_producer, stream_consumer, handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start validated feedback dependencies and exactly one consumer task."""
    global producer, consumer, feedback_handler
    settings = FeedbackSettings.from_env()
    if settings.production and not os.getenv("INTERNAL_API_SECRET"):
        raise RuntimeError("INTERNAL_API_SECRET is required in production")
    producer, consumer, feedback_handler = _build_feedback_runtime(settings)
    await producer.start()
    await asyncio.to_thread(feedback_handler.healthy)
    await consumer.start()
    app.state.feedback_settings = settings
    try:
        yield
    finally:
        if consumer:
            await consumer.stop()
        redis_client = producer.redis_client if producer else None
        close = getattr(redis_client, "close", None)
        if close:
            await asyncio.to_thread(close)
        qdrant_client = feedback_handler.qdrant if feedback_handler else None
        close_qdrant = getattr(qdrant_client, "close", None)
        if close_qdrant:
            await asyncio.to_thread(close_qdrant)
        producer = None
        consumer = None
        feedback_handler = None


app = FastAPI(
    title="Git Social ML API",
    description="Authenticated ML operations and durable feedback ingestion.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def authenticate_non_health_routes(request: Request, call_next):
    """Fail closed for every route except the single health endpoint."""
    if request.url.path == "/api/v1/health":
        return await call_next(request)
    secret = os.getenv("INTERNAL_API_SECRET")
    if not secret:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Internal API authentication is not configured."},
        )
    header_name = os.getenv("INTERNAL_API_HEADER", "x-internal-secret").lower()
    supplied = request.headers.get(header_name)
    if not supplied or not hmac.compare_digest(supplied, secret):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Unauthorized."},
        )
    return await call_next(request)


@app.post("/api/v1/feedback", status_code=status.HTTP_202_ACCEPTED)
async def submit_feedback(request: FeedbackRequest):
    action = normalize_interaction(request.action)
    try:
        definition = get_interaction(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if request.dwell_seconds is not None and not math.isfinite(request.dwell_seconds):
        raise HTTPException(status_code=422, detail="dwell_seconds must be finite")
    if request.occurred_at.tzinfo is None:
        raise HTTPException(status_code=422, detail="occurred_at must include a timezone")
    if action == "dwell" and request.dwell_seconds is None:
        raise HTTPException(status_code=422, detail="dwell_seconds is required for dwell")
    if action != "dwell" and request.dwell_seconds is not None:
        raise HTTPException(status_code=422, detail="dwell_seconds is only valid for dwell")

    queued = False
    if definition.realtime:
        if producer is None:
            raise HTTPException(status_code=503, detail="Feedback stream is unavailable.")
        try:
            await producer.submit_feedback(
                user_id=str(request.user_id),
                repo_id=request.repo_id,
                action=action,
                event_id=str(request.event_id),
                occurred_at=request.occurred_at.isoformat(),
                schema_version=request.schema_version,
                dwell_seconds=request.dwell_seconds,
            )
            queued = True
        except Exception:
            logger.exception("Unable to publish feedback event %s", request.event_id)
            raise HTTPException(status_code=503, detail="Feedback stream is unavailable.")

    return {
        "status": "accepted",
        "data": {
            "event_id": str(request.event_id),
            "user_id": str(request.user_id),
            "repo_id": request.repo_id,
            "action": action,
            "reference_score": definition.reference_score,
            "queued_for_realtime_ml": queued,
        },
    }


def _load_service(module_name: str, class_name: str) -> Any:
    """Load other owned workstreams only when their endpoint is called."""
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()


@app.post("/api/v1/recommendations/generate")
async def generate_recommendations(request: RecommendationRequest):
    global retrieval_engine
    try:
        retrieval_engine = retrieval_engine or _load_service("retrieval_engine", "RetrievalEngine")
        batches = await run_in_threadpool(
            retrieval_engine.fetch_onboarding_batches,
            request.user_id,
            is_cold_start=request.is_cold_start,
        )
        return {"status": "success", "user_id": request.user_id, "data": batches}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception:
        logger.exception("Recommendation generation failed for %s", request.user_id)
        raise HTTPException(status_code=500, detail="Recommendation generation failed.")


@app.post("/api/v1/onboard")
async def onboard_user(request: OnboardingRequest):
    global onboarding_pipeline
    onboarding_pipeline = onboarding_pipeline or _load_service(
        "scripts.user_onboarding", "UserOnboardingPipeline"
    )
    data = request.model_dump(exclude_none=True)
    user_id = data.pop("user_id")
    try:
        success = await run_in_threadpool(onboarding_pipeline.onboard_user, user_id, data)
        if not success:
            raise HTTPException(status_code=500, detail="User onboarding failed.")
        return {"status": "success", "user_id": user_id}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("User onboarding failed for %s", user_id)
        raise HTTPException(status_code=500, detail="User onboarding failed.")


@app.post("/api/v1/embed-repo")
async def embed_repo(request: EmbedRepoRequest):
    global repo_embedding_pipeline
    repo_embedding_pipeline = repo_embedding_pipeline or _load_service(
        "embedding.embedding_pipeline", "RepositoryEmbeddingPipeline"
    )
    payload = {
        "id": request.repo_id,
        "repo_id": request.repo_id,
        "full_name": request.github_repo,
        "html_url": request.github_repo_url,
        "description": request.description or "",
        "primary_language": request.primary_language or "Unknown",
        "languages": request.languages,
        "topics": request.topics,
        "extracted_paragraphs": [request.readme_summary] if request.readme_summary else [],
        "readme_length": len(request.readme_summary or ""),
        "star_count": request.star_count,
        "fork_count": request.fork_count,
        "open_issues_count": request.open_issues_count,
        "created_at": request.created_at,
        "updated_at": request.updated_at,
    }
    try:
        results = await run_in_threadpool(repo_embedding_pipeline.index_batch, [payload])
        result = results[0] if results else None
        return {
            "status": "success",
            "repo_id": request.repo_id,
            "github_repo": request.github_repo,
            "embedding_version": getattr(result, "embedding_version", None),
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Repository embedding failed for %s", request.github_repo)
        raise HTTPException(status_code=500, detail="Repository embedding failed.")


@app.get("/api/v1/health")
async def health_check():
    checks = {"lifecycle": producer is not None and feedback_handler is not None}
    checks["consumer"] = bool(consumer and consumer.healthy)
    try:
        checks["redis"] = bool(
            producer and producer.redis_client
            and await asyncio.to_thread(producer.redis_client.ping)
        )
    except Exception:
        checks["redis"] = False
    try:
        checks["qdrant"] = bool(
            feedback_handler and await asyncio.to_thread(feedback_handler.healthy)
        )
    except Exception:
        checks["qdrant"] = False
    healthy = all(checks.values())
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "healthy" if healthy else "unhealthy", "checks": checks},
    )
