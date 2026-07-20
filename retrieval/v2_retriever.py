from __future__ import annotations

import logging
import math
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import QDRANT_API_KEY, QDRANT_COLLECTION_NAME, QDRANT_URL, QDRANT_VECTOR_NAME
from embedding.vector_contract import (
    REPOSITORY_DISCOVERY_CHANNELS,
    legacy_repository_point_id,
    user_point_ids,
)
from inference.feed_assembly import FeedAssemblySystem
from inference.ranker_service import RankerService
from retrieval.config import TOTAL_CANDIDATE_POOL
from scripts.user_onboarding import TARGET_VECTOR_NAME, USER_PROFILES_COLLECTION

logger = logging.getLogger("pipeline.v2_retrieval")

_DISCOVERY_SOURCES = {
    "trend": "trending",
    "activity": "active",
    "popularity": "popular",
    "freshness": "fresh",
}


@dataclass(frozen=True, slots=True)
class RankedRepository:
    repo_id: str
    score: float
    source: str


@dataclass(frozen=True, slots=True)
class RecommendationBatch:
    items: list[RankedRepository]
    model_version: str
    embedding_version: str
    ranker_applied: bool
    fallback_reason: str | None = None


class QdrantV2Retriever:
    """V2 candidate retrieval plus fail-safe heavy-ranker serving."""

    def __init__(
        self,
        *,
        client: QdrantClient | None = None,
        repository_collection: str = QDRANT_COLLECTION_NAME,
        user_collection: str = USER_PROFILES_COLLECTION,
        max_candidates: int = TOTAL_CANDIDATE_POOL,
        assembler: FeedAssemblySystem | None = None,
        ranker: RankerService | None = None,
        heavy_ranker_enabled: bool | None = None,
    ) -> None:
        self.client = client or QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=float(os.getenv("QDRANT_TIMEOUT_SECONDS", "10")),
        )
        self.repository_collection = repository_collection
        self.user_collection = user_collection
        self.max_candidates = max(50, min(max_candidates, 2_000))
        self.assembler = assembler or FeedAssemblySystem()
        self.hybrid_model_version = os.getenv("ML_MODEL_VERSION", "qdrant-hybrid-v2")
        self.embedding_version = os.getenv(
            "REPOSITORY_EMBEDDING_VERSION", "repo-embedding-v2"
        )

        if heavy_ranker_enabled is None:
            heavy_ranker_enabled = self._boolean_env(
                "V2_HEAVY_RANKER_ENABLED", default=True
            )
        self.heavy_ranker_enabled = heavy_ranker_enabled
        self.heavy_ranker_required = self.heavy_ranker_enabled and self._boolean_env(
            "V2_HEAVY_RANKER_REQUIRED",
            default=(
                os.getenv("APP_ENV", "development").strip().casefold()
                == "production"
            ),
        )
        self.ranker: RankerService | None = None
        self.ranker_error: str | None = None
        self.heavy_model_version: str | None = None
        if self.heavy_ranker_enabled:
            try:
                self.ranker = ranker or self._load_ranker()
                if not getattr(self.ranker, "ready", True):
                    raise RuntimeError("heavy ranker artifact is not loaded")
                ranker_version = str(
                    getattr(self.ranker, "model_version", "heavy-ranker-unknown")
                )
                self.heavy_model_version = f"{ranker_version}-v2-adapter"
            except Exception as exc:
                self.ranker = None
                self.ranker_error = str(exc)
                logger.exception(
                    "V2 heavy ranker initialization failed; hybrid fallback remains active"
                )

        # Compatibility for callers that inspect the configured primary model.
        self.model_version = self.heavy_model_version or self.hybrid_model_version

    @staticmethod
    def _boolean_env(name: str, *, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        normalized = raw.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name} must be a boolean; got {raw!r}")

    def _load_ranker(self) -> RankerService:
        inference_dir = Path(__file__).resolve().parents[1] / "inference"
        return RankerService(
            model_path=str(inference_dir / "heavy_ranker.pt"),
            scaler_path=str(inference_dir / "feature_scaler.json"),
            manifest_path=str(inference_dir / "model_manifest.json"),
            expected_embedding_version=self.embedding_version,
        )

    @staticmethod
    def _canonical_id(point: Any) -> str | None:
        payload = point.payload or {}
        candidate = str(payload.get("repo_id") or point.id)
        try:
            canonical = str(uuid.UUID(candidate))
        except (ValueError, AttributeError):
            return None
        valid_point_ids = {canonical, legacy_repository_point_id(canonical)}
        has_canonical_payload = str(payload.get("repo_id")) == canonical
        return canonical if str(point.id) in valid_point_ids and has_canonical_payload else None

    @staticmethod
    def _vector(value: Any, preferred: str | None = None) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            if preferred and preferred in value:
                return list(value[preferred])
            if len(value) == 1:
                return list(next(iter(value.values())))
            return None
        return list(value)

    @staticmethod
    def _eligible_filter() -> models.Filter:
        # Filtering in Qdrant happens before the retrieval limit is applied, so
        # stale pre-contract points cannot crowd valid backend repositories out.
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="content_version",
                    range=models.Range(gte=1),
                )
            ]
        )

    def _user_profile(self, user_id: str) -> tuple[list[float] | None, dict[str, Any]]:
        canonical, legacy = user_point_ids(user_id)
        points = self.client.retrieve(
            collection_name=self.user_collection,
            ids=[canonical, legacy],
            with_vectors=True,
            with_payload=True,
        )
        if not points:
            return None, {}
        by_id = {str(point.id): point for point in points}
        point = by_id.get(canonical) or by_id.get(legacy)
        if not point:
            return None, {}
        return (
            self._vector(point.vector, TARGET_VECTOR_NAME),
            dict(point.payload or {}),
        )

    def _user_vector(self, user_id: str) -> list[float] | None:
        """Compatibility wrapper retained for existing callers and tests."""
        vector, _ = self._user_profile(user_id)
        return vector

    @staticmethod
    def _profile_terms(payload: dict[str, Any]) -> list[str]:
        terms: list[str] = []
        for field_name in ("skills", "tech_stack", "interests", "topics"):
            raw = payload.get(field_name)
            values = raw if isinstance(raw, (list, tuple, set)) else [raw]
            for value in values:
                if value is not None and (text := str(value).strip()):
                    terms.append(text)
        return list(dict.fromkeys(terms))

    def _semantic(self, vector: list[float], limit: int) -> list[tuple[Any, float]]:
        response = self.client.query_points(
            collection_name=self.repository_collection,
            query=vector,
            using=QDRANT_VECTOR_NAME,
            query_filter=self._eligible_filter(),
            limit=min(limit, self.max_candidates),
            with_payload=True,
            with_vectors=[QDRANT_VECTOR_NAME],
        )
        return [(point, float(point.score)) for point in response.points]

    def _discovery(self, limit: int) -> list[tuple[Any, str]]:
        target = min(limit, self.max_candidates)
        channels = list(_DISCOVERY_SOURCES)
        per_channel = min(
            self.max_candidates,
            max(25, math.ceil((target * 2) / len(channels))),
        )
        channel_points: list[list[tuple[Any, str]]] = []
        for channel in channels:
            try:
                records, _ = self.client.scroll(
                    collection_name=self.repository_collection,
                    scroll_filter=self._eligible_filter(),
                    limit=per_channel,
                    order_by=models.OrderBy(
                        key=REPOSITORY_DISCOVERY_CHANNELS[channel],
                        direction=models.Direction.DESC,
                    ),
                    with_payload=True,
                    with_vectors=[QDRANT_VECTOR_NAME],
                )
            except Exception:
                logger.exception("V2 discovery channel %s failed", channel)
                records = []
            channel_points.append(
                [(point, _DISCOVERY_SOURCES[channel]) for point in records]
            )

        # Round-robin prevents one discovery signal from monopolizing the
        # ranker pool before cross-channel deduplication.
        combined: list[tuple[Any, str]] = []
        seen_repo_ids: set[str] = set()
        for index in range(max((len(points) for points in channel_points), default=0)):
            for points in channel_points:
                if index < len(points):
                    point, source = points[index]
                    repo_id = self._canonical_id(point)
                    if not repo_id or repo_id in seen_repo_ids:
                        continue
                    seen_repo_ids.add(repo_id)
                    combined.append((point, source))
                    if len(combined) >= target:
                        return combined
        return combined

    @staticmethod
    def _number(value: Any, *, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default

    @staticmethod
    def _discovery_score(payload: dict[str, Any]) -> tuple[float, str]:
        stars = max(0, int(QdrantV2Retriever._number(payload.get("star_count"))))
        velocity = max(
            0.0,
            QdrantV2Retriever._number(
                payload.get("trend_velocity") or payload.get("delta_7d")
            ),
        )
        activity = max(
            0.0, QdrantV2Retriever._number(payload.get("activity_score"))
        )
        raw_pushed_days = payload.get("pushed_days_ago")
        pushed_days = (
            999
            if raw_pushed_days is None
            else max(0, int(QdrantV2Retriever._number(raw_pushed_days, default=999)))
        )
        freshness = math.exp(-pushed_days / 60)
        score = (
            0.35 * math.log1p(stars)
            + 0.35 * math.log1p(velocity)
            + 0.2 * activity
            + 0.1 * freshness
        )
        source = (
            "trending"
            if velocity > 0
            else "fresh"
            if pushed_days <= 30
            else "popular"
        )
        return score, source

    def _heavy_rank(
        self,
        *,
        user_vector: list[float] | None,
        user_profile: dict[str, Any],
        ranked: list[RankedRepository],
        metadata: dict[str, dict[str, Any]],
        vectors: dict[str, list[float] | None],
    ) -> tuple[list[RankedRepository] | None, str | None]:
        if self.ranker is None:
            return None, self.ranker_error or "heavy ranker is disabled"
        if user_vector is None:
            return None, "user profile vector is unavailable"
        if not ranked:
            return None, "no eligible candidates were retrieved"

        ranker_candidates: list[dict[str, Any]] = []
        for item in ranked:
            payload = metadata.get(item.repo_id, {})
            ranker_candidates.append(
                {
                    **payload,
                    "id": item.repo_id,
                    "embedding": vectors.get(item.repo_id),
                    "languages": payload.get("languages")
                    or ([payload["primary_language"]] if payload.get("primary_language") else []),
                    "topics": payload.get("topics") or [],
                    "tags": payload.get("tags") or [],
                }
            )

        try:
            results = self.ranker.score_batch(
                user_vector,
                self._profile_terms(user_profile),
                ranker_candidates,
            )
            expected_ids = {item.repo_id for item in ranked}
            returned_ids = [str(result.get("repo_id")) for result in results]
            if (
                len(results) != len(ranked)
                or len(set(returned_ids)) != len(returned_ids)
                or set(returned_ids) != expected_ids
                or any(
                    not math.isfinite(self._number(result.get("final_score"), default=math.nan))
                    for result in results
                )
            ):
                raise ValueError("heavy ranker returned an incomplete or invalid candidate set")
            sources = {item.repo_id: item.source for item in ranked}
            return (
                [
                    RankedRepository(
                        repo_id=str(result["repo_id"]),
                        score=float(result["final_score"]),
                        source=sources[str(result["repo_id"])],
                    )
                    for result in results
                ],
                None,
            )
        except Exception as exc:
            logger.exception(
                "V2 heavy ranking failed for one request; using hybrid fallback"
            )
            return None, str(exc)

    def recommend_batch(
        self,
        user_id: str,
        limit: int,
        exclude_repo_ids: list[str],
        generation_seed: str | None = None,
    ) -> RecommendationBatch:
        excluded = {str(uuid.UUID(item)) for item in exclude_repo_ids}
        candidates: dict[str, RankedRepository] = {}
        metadata: dict[str, dict[str, Any]] = {}
        vectors: dict[str, list[float] | None] = {}
        user_vector, user_profile = self._user_profile(user_id)
        if user_vector:
            try:
                semantic = self._semantic(user_vector, max(limit * 5, 100))
            except Exception:
                logger.exception(
                    "V2 semantic retrieval failed; continuing with discovery"
                )
                semantic = []
            for point, score in semantic:
                repo_id = self._canonical_id(point)
                if repo_id and repo_id not in excluded and math.isfinite(score):
                    candidates[repo_id] = RankedRepository(repo_id, score, "semantic")
                    metadata[repo_id] = dict(point.payload or {})
                    vectors[repo_id] = self._vector(
                        getattr(point, "vector", None), QDRANT_VECTOR_NAME
                    )

        discovery: list[tuple[str, float, str, dict[str, Any], list[float] | None]] = []
        for point, channel_source in self._discovery(max(limit * 8, 200)):
            repo_id = self._canonical_id(point)
            if not repo_id or repo_id in excluded:
                continue
            payload = dict(point.payload or {})
            score, inferred_source = self._discovery_score(payload)
            if math.isfinite(score):
                discovery.append(
                    (
                        repo_id,
                        score,
                        channel_source or inferred_source,
                        payload,
                        self._vector(
                            getattr(point, "vector", None), QDRANT_VECTOR_NAME
                        ),
                    )
                )

        maximum_discovery = max((item[1] for item in discovery), default=0.0)
        for repo_id, score, source, payload, vector in discovery:
            normalized_discovery = (
                score / maximum_discovery if maximum_discovery > 0 else 0.0
            )
            current = candidates.get(repo_id)
            combined = 0.15 * normalized_discovery
            if current is not None:
                combined += current.score
            if math.isfinite(combined):
                candidates[repo_id] = RankedRepository(
                    repo_id, combined, current.source if current else source
                )
                metadata[repo_id] = payload
                vectors[repo_id] = vector or vectors.get(repo_id)

        hybrid_ranked = sorted(
            candidates.values(), key=lambda item: (-item.score, item.repo_id)
        )[: self.max_candidates]
        heavy_ranked, fallback_reason = self._heavy_rank(
            user_vector=user_vector,
            user_profile=user_profile,
            ranked=hybrid_ranked,
            metadata=metadata,
            vectors=vectors,
        )
        ranker_applied = heavy_ranked is not None
        ranked = heavy_ranked if heavy_ranked is not None else hybrid_ranked
        model_version = (
            self.heavy_model_version
            if ranker_applied and self.heavy_model_version
            else self.hybrid_model_version
        )

        assembly_input = [
            {
                "repo_id": item.repo_id,
                "score": item.score,
                "final_score": item.score,
                "source": item.source,
                "primary_language": metadata.get(item.repo_id, {}).get(
                    "primary_language"
                ),
                "created_at": metadata.get(item.repo_id, {}).get("created_at"),
            }
            for item in ranked
        ]
        shaped = self.assembler.shape_batch(
            assembly_input,
            seen_repo_ids=excluded,
            randomizer=random.Random(generation_seed),
        )
        items = [
            RankedRepository(
                repo_id=str(item["repo_id"]),
                score=round(float(item["final_score"]), 6),
                source=str(item["source"]),
            )
            for item in shaped[:limit]
        ]
        return RecommendationBatch(
            items=items,
            model_version=model_version,
            embedding_version=self.embedding_version,
            ranker_applied=ranker_applied,
            fallback_reason=None if ranker_applied else fallback_reason,
        )

    def recommend(
        self,
        user_id: str,
        limit: int,
        exclude_repo_ids: list[str],
        generation_seed: str | None = None,
    ) -> list[RankedRepository]:
        """Compatibility wrapper returning only the recommendation items."""
        return self.recommend_batch(
            user_id,
            limit,
            exclude_repo_ids,
            generation_seed,
        ).items

    def health(self) -> dict[str, Any]:
        if self.heavy_ranker_required and self.ranker is None:
            raise RuntimeError(
                "V2 heavy ranker is required but unavailable: "
                f"{self.ranker_error or 'unknown initialization failure'}"
            )
        info = self.client.get_collection(self.repository_collection)
        health: dict[str, Any] = {
            "qdrant": "healthy",
            "repository_points": int(info.points_count or 0),
            "heavy_ranker_enabled": self.heavy_ranker_enabled,
            "heavy_ranker_required": self.heavy_ranker_required,
            "heavy_ranker_ready": self.ranker is not None,
            "heavy_ranker_model_version": self.heavy_model_version,
            "hybrid_fallback_model_version": self.hybrid_model_version,
        }
        if self.ranker_error:
            health["heavy_ranker_error"] = self.ranker_error
        try:
            eligible = self.client.count(
                collection_name=self.repository_collection,
                count_filter=self._eligible_filter(),
                exact=True,
            )
            health["eligible_repository_points"] = int(eligible.count or 0)
        except Exception as exc:
            logger.warning("Could not count eligible V2 repository points: %s", exc)
        return health
