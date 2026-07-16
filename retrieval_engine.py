"""Integrated feed assembly engine.

This module wires together the complete post-onboarding recommendation pipeline:

  User Profile (Qdrant) → CandidateRetriever → RankerService (MMoE) → Ranked Batches

Usage::

    from retrieval_engine import RetrievalEngine

    engine = RetrievalEngine()
    result = engine.generate_recommendations(
        schema_version=2,
        generation_id="00000000-0000-4000-8000-000000000001",
        user_id="00000000-0000-4000-8000-000000000002",
        feed_version=1,
    )
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from typing import Any
from uuid import uuid4

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from qdrant_client import QdrantClient
from embedding import (
    canonical_backend_uuid,
    user_point_id,
)
from inference.feed_assembly import FeedAssemblySystem
from inference.feature_spec import RANKER_MODEL_VERSION
from retrieval.candidate_retriever import CandidateRetriever

from config import (  # type: ignore
    QDRANT_API_KEY,
    QDRANT_URL,
    REPOSITORY_EMBEDDING_VERSION,
)
from scripts.user_onboarding import USER_PROFILES_COLLECTION, TARGET_VECTOR_NAME  # type: ignore

logger = logging.getLogger("pipeline.retrieval")

BATCH_SIZE = 15
RECOMMENDATION_SCHEMA_VERSION = 2
MAX_RECOMMENDATION_ITEMS = BATCH_SIZE * 3


# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class RetrievalEngine:
    """Integrated feed assembler for Qdrant retrieval and MMoE ranking.

    Pipeline
    --------
    1. Load user interest embedding from Qdrant ``user_profiles``.
    2. Pull semantic and discovery candidates via ``CandidateRetriever``.
    3. Score every candidate with ``RankerService`` (MMoE heavy ranker).
    4. Shape and slice the ranked candidates into three batches of 15.
    """

    def __init__(
        self,
        *,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        self._url = qdrant_url or QDRANT_URL
        self._api_key = qdrant_api_key or QDRANT_API_KEY

        # Direct client for user_profiles (unnamed-vector collection)
        self._client = QdrantClient(url=self._url, api_key=self._api_key, timeout=30.0)

        # Lazy-loaded sub-components
        self._candidate_retriever: Any = None
        self._ranker: Any = None
        self._ranker_failed = False
        self.model_version = RANKER_MODEL_VERSION
        self.embedding_version = REPOSITORY_EMBEDDING_VERSION

    # ── Lazy sub-component accessors ──────────────────────────────────────────

    @property
    def candidate_retriever(self):
        """Lazy-load Person 3's Qdrant-only candidate retriever."""
        if self._candidate_retriever is None:
            try:
                self._candidate_retriever = CandidateRetriever(
                    qdrant_url=self._url,
                    qdrant_api_key=self._api_key,
                )
            except Exception as exc:
                logger.warning("Could not initialize CandidateRetriever: %s", exc)
                self._candidate_retriever = False
        return (
            self._candidate_retriever
            if self._candidate_retriever is not False
            else None
        )

    @property
    def ranker(self):
        """Lazy-load the RankerService (MMoE heavy ranker)."""
        if self._ranker is None and not self._ranker_failed:
            try:
                # Resolve paths relative to the inference/ directory
                _base = os.path.join(os.path.dirname(__file__), "inference")
                model_path = os.path.join(_base, "heavy_ranker.pt")
                scaler_path = os.path.join(_base, "feature_scaler.json")

                sys.path.insert(0, _base)
                from ranker_service import RankerService  # type: ignore
                self._ranker = RankerService(
                    model_path=model_path,
                    scaler_path=scaler_path,
                )
                self.model_version = self._ranker.model_version
                self.embedding_version = self._ranker.embedding_version
            except Exception as exc:
                logger.warning("Could not initialize RankerService: %s", exc)
                self._ranker_failed = True
        return self._ranker

    def _sanitize_batch_item(self, item: dict) -> dict:
        """Return a JSON-safe, public projection of an internal ranked item."""
        allowed_keys = (
            "repo_id",
            "full_name",
            "repo_name",
            "github_repo_url",
            "description",
            "primary_language",
            "languages",
            "topics",
            "star_count",
            "forks_count",
            "final_score",
            "predictions",
            "score_source",
            "category",
            "created_at",
            "updated_at",
        )
        drop_value = object()

        def to_plain_value(value):
            if isinstance(value, np.ndarray):
                return drop_value
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                cleaned = {}
                for key, nested_value in value.items():
                    plain_value = to_plain_value(nested_value)
                    if plain_value is not drop_value:
                        cleaned[key] = plain_value
                return cleaned
            if isinstance(value, (list, tuple, set)):
                cleaned = []
                for nested_value in value:
                    plain_value = to_plain_value(nested_value)
                    if plain_value is not drop_value:
                        cleaned.append(plain_value)
                return cleaned
            return value

        sanitized = {}
        for key in allowed_keys:
            if key not in item:
                continue

            value = item[key]
            if key == "predictions":
                if not isinstance(value, dict):
                    continue
                predictions = {}
                for prediction_name, prediction_value in value.items():
                    if isinstance(prediction_value, np.ndarray):
                        continue
                    try:
                        predictions[prediction_name] = float(prediction_value)
                    except (TypeError, ValueError):
                        continue
                sanitized[key] = predictions
                continue

            plain_value = to_plain_value(value)
            if plain_value is not drop_value:
                sanitized[key] = plain_value

        return sanitized

    # ── Core public API ───────────────────────────────────────────────────────

    def generate_recommendations(
        self,
        *,
        schema_version: int,
        generation_id: str,
        user_id: str,
        feed_version: int,
        limit: int = MAX_RECOMMENDATION_ITEMS,
        is_cold_start: bool = False,
        seen_repo_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """Return the canonical backend-to-ML recommendation v2 response.

        ``seen_repo_ids`` is the backend-owned exact-exclusion snapshot. It
        should include previously served repositories and durable interaction
        exclusions such as explicit dislikes or already-consumed saves. The
        user's Qdrant vector already reflects feedback directionally, so this
        online path must not query ``FeedbackStore`` and apply it a second time.
        """
        if schema_version != RECOMMENDATION_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {RECOMMENDATION_SCHEMA_VERSION}"
            )
        generation_id = canonical_backend_uuid(
            generation_id,
            field_name="generation_id",
        )
        user_id = canonical_backend_uuid(user_id, field_name="user_id")
        if isinstance(feed_version, bool) or not isinstance(feed_version, int):
            raise TypeError("feed_version must be an integer")
        if feed_version < 0:
            raise ValueError("feed_version must be non-negative")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_RECOMMENDATION_ITEMS
        ):
            raise ValueError(
                f"limit must be between 1 and {MAX_RECOMMENDATION_ITEMS}"
            )

        batches = self.fetch_onboarding_batches(
            user_id,
            is_cold_start=is_cold_start,
            seen_repo_ids=seen_repo_ids,
        )
        ranked_items = (
            batches["batch_1"]
            + batches["batch_2"]
            + batches["batch_3"]
        )

        items: list[dict[str, Any]] = []
        emitted_repo_ids: set[str] = set()
        for item in ranked_items:
            repo_id = canonical_backend_uuid(
                item.get("repo_id"),
                field_name="repo_id",
            )
            if repo_id in emitted_repo_ids:
                continue

            score = float(item.get("final_score") or 0.0)
            if not math.isfinite(score):
                raise ValueError(f"Recommendation score for {repo_id} must be finite")

            score_source = str(item.get("score_source") or "")
            if score_source == "cold_start":
                source = "cold_start"
            elif score_source == "cosine_fallback":
                source = "retrieval_fallback"
            else:
                source = "personalized"

            emitted_repo_ids.add(repo_id)
            items.append(
                {
                    "repo_id": repo_id,
                    "score": score,
                    "source": source,
                }
            )
            if len(items) >= limit:
                break

        return {
            "schema_version": RECOMMENDATION_SCHEMA_VERSION,
            "generation_id": generation_id,
            "user_id": user_id,
            "feed_version": feed_version,
            "model_version": self.model_version,
            "embedding_version": self.embedding_version,
            "items": items,
        }

    def fetch_onboarding_batches(
        self,
        user_id: str,
        *,
        is_cold_start: bool = False,
        seen_repo_ids: set[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Generate legacy internal batches used by the existing v1 adapter.

        The caller owns exact exclusions through ``seen_repo_ids``; online ML
        intentionally performs no PostgreSQL feedback lookup.

        Returns
        -------
        dict with keys ``"batch_1"``, ``"batch_2"``, ``"batch_3"``, each a
        list of up to ``BATCH_SIZE`` ranked repository dicts.
        """
        import time

        user_id = canonical_backend_uuid(user_id, field_name="user_id")
        if seen_repo_ids is not None:
            seen_repo_ids = {
                canonical_backend_uuid(repo_id, field_name="seen_repo_id")
                for repo_id in seen_repo_ids
            }

        # ── 1. Get user profile from Qdrant ───────────────────────────────────
        try:
            user_vector, user_skills = self._get_user_profile(user_id)
        except ValueError:
            if is_cold_start:
                logger.info(
                    "Cold start user '%s' vector not present; using empty skills.",
                    user_id,
                )
                user_vector = []
                user_skills = []
            else:
                raise
        except Exception as exc:
            # Catch connection errors (Qdrant down or network failure)
            logger.warning(
                "User '%s' Qdrant lookup failed (%s); using cold start.",
                user_id,
                type(exc).__name__,
            )
            user_vector = []
            user_skills = []
            # Force cold start pipeline if we completely lose Qdrant,
            # as semantic ranking requires a valid vector anyway.
            is_cold_start = True

        if is_cold_start:
            return self._cold_start_pipeline(
                user_id,
                user_vector,
                user_skills,
                seen_repo_ids=seen_repo_ids,
            )

        # ── 2. Retrieve candidate pool ────────────────────────────────────────
        start_retrieval = time.time()
        candidates = self._retrieve_candidates(user_vector, user_skills)
        retrieval_latency = (time.time() - start_retrieval) * 1000.0

        # ── 3. Rank and shape the candidate pool ──────────────────────────────
        start_ranking = time.time()
        ranked = self._rank_candidates(user_vector, user_skills, candidates)
        # Do not re-query FeedbackStore here: feedback is already represented
        # in the Qdrant user vector, while exact exclusions are supplied by the
        # backend in seen_repo_ids.
        ranked = FeedAssemblySystem().shape_batch(
            ranked,
            seen_repo_ids=seen_repo_ids,
        )
        ranking_latency = (time.time() - start_ranking) * 1000.0

        # ── 4. Slice into 3 batches of BATCH_SIZE ─────────────────────────────
        batches = {
            "batch_1": [
                self._sanitize_batch_item(item)
                for item in ranked[0:BATCH_SIZE]
            ],
            "batch_2": [
                self._sanitize_batch_item(item)
                for item in ranked[BATCH_SIZE: BATCH_SIZE * 2]
            ],
            "batch_3": [
                self._sanitize_batch_item(item)
                for item in ranked[BATCH_SIZE * 2: BATCH_SIZE * 3]
            ],
        }

        logger.info(
            "Generated onboarding batches for '%s': %d / %d / %d items.",
            user_id,
            len(batches["batch_1"]),
            len(batches["batch_2"]),
            len(batches["batch_3"]),
        )
        logger.info(
            "Latency Profile: Candidate Retrieval = %.2fms, MMoE Ranking = %.2fms (Total = %.2fms)",
            retrieval_latency,
            ranking_latency,
            retrieval_latency + ranking_latency,
        )
        return batches

    # ── Cold Start ────────────────────────────────────────────────────────────

    def _cold_start_pipeline(
        self,
        user_id: str,
        user_vector: list[float],
        user_skills: list[str],
        *,
        seen_repo_ids: set[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Dedicated retrieval and ranking pathway for new users with 0 interactions."""
        import time

        logger.info("Executing Cold Start pipeline for user '%s'", user_id)
        start_retrieval = time.time()
        
        candidates = self._retrieve_cold_start_candidates(user_skills)
        retrieval_latency = (time.time() - start_retrieval) * 1000.0

        start_ranking = time.time()
        ranked = self._score_cold_start_candidates(user_skills, candidates)
        ranked = FeedAssemblySystem().shape_batch(
            ranked,
            seen_repo_ids=seen_repo_ids,
        )
        ranking_latency = (time.time() - start_ranking) * 1000.0

        batches = {
            "batch_1": [
                self._sanitize_batch_item(item)
                for item in ranked[0:BATCH_SIZE]
            ],
            "batch_2": [
                self._sanitize_batch_item(item)
                for item in ranked[BATCH_SIZE: BATCH_SIZE * 2]
            ],
            "batch_3": [
                self._sanitize_batch_item(item)
                for item in ranked[BATCH_SIZE * 2: BATCH_SIZE * 3]
            ],
        }

        logger.info(
            "Generated Cold Start batches for '%s': %d / %d / %d items.",
            user_id,
            len(batches["batch_1"]),
            len(batches["batch_2"]),
            len(batches["batch_3"]),
        )
        logger.info(
            "Cold Start Latency: Retrieval = %.2fms, Scoring = %.2fms (Total = %.2fms)",
            retrieval_latency,
            ranking_latency,
            retrieval_latency + ranking_latency,
        )
        return batches

    def _retrieve_cold_start_candidates(self, user_skills: list[str]) -> list[dict[str, Any]]:
        """Retrieve Person 3 discovery candidates for a cold-start user."""
        retriever = self.candidate_retriever
        if retriever is None:
            logger.warning(
                "CandidateRetriever unavailable. Cold start retrieval returning empty."
            )
            return []

        try:
            records = retriever.retrieve_candidates(
                user_embedding=[],
                user_interests=user_skills,
            )
        except Exception as exc:
            logger.warning("Cold-start candidate retrieval failed: %s", exc)
            raise
        return self._normalize_retriever_candidates(records)

    @staticmethod
    def _normalize_retriever_candidate(record: dict) -> dict | None:
        """Validate Person 3's candidate and preserve the ranker contract."""
        payload = record.get("payload")
        if not isinstance(payload, dict):
            payload = {}

        repo_id = record.get("repo_id") or payload.get("repo_id")
        try:
            repo_id = canonical_backend_uuid(repo_id, field_name="repo_id")
        except (TypeError, ValueError) as exc:
            logger.warning("Skipping Qdrant repository record: %s", exc)
            return None

        candidate = dict(payload)
        candidate.update(record)
        candidate["repo_id"] = repo_id
        candidate["full_name"] = (
            record.get("full_name")
            or payload.get("full_name")
            or str(repo_id)
        )
        candidate["repo_embedding"] = (
            record.get("repo_embedding")
            or record.get("vector")
            or []
        )
        candidate["retrieval_score"] = float(
            record.get("retrieval_score")
            or record.get("score")
            or 0.0
        )
        candidate["retrieval_source"] = str(
            record.get("retrieval_source") or "unknown"
        )

        if "github_repo_url" not in candidate and payload.get("html_url") is not None:
            candidate["github_repo_url"] = payload["html_url"]
        if "forks_count" not in candidate and payload.get("fork_count") is not None:
            candidate["forks_count"] = payload["fork_count"]
        return candidate

    def _normalize_retriever_candidates(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter invalid IDs and deduplicate candidates at the v2 boundary."""
        candidates: list[dict[str, Any]] = []
        seen_repo_ids: set[str] = set()
        fallback_seen = False
        for record in records:
            fallback_seen = fallback_seen or record.get("retrieval_source") == "fallback"
            candidate = self._normalize_retriever_candidate(record)
            if candidate is None or candidate["repo_id"] in seen_repo_ids:
                continue
            seen_repo_ids.add(candidate["repo_id"])
            candidates.append(candidate)

        if records and fallback_seen and not candidates:
            raise RuntimeError(
                "Qdrant retrieval failed and static fallback candidates do not "
                "satisfy the backend UUID contract"
            )
        return candidates

    def _score_cold_start_candidates(
        self, user_skills: list[str], candidates: list[dict]
    ) -> list[dict]:
        """Deterministically score candidates based on skill match and popularity."""
        import math

        if not candidates:
            return []

        skill_weight = 0.6
        stars_weight = 0.4
        max_log_stars = math.log1p(500_000)  # normalisation ceiling
        user_set = {s.lower() for s in user_skills}

        for c in candidates:
            # --- Skill match ratio (0.0 to 1.0) ---
            repo_signals = set()
            lang = c.get("primary_language", "")
            if lang and lang != "Unknown":
                repo_signals.add(lang.lower())
            
            for t in (c.get("topics") or []):
                repo_signals.add(str(t).lower())
            
            for l in (c.get("languages") or []):
                # if language_used was a dict mapped to bytes, handle appropriately, 
                # but typically frontend/backend uses strings or lists
                repo_signals.add(str(l).lower())

            overlap = len(repo_signals & user_set)
            skill_match = overlap / max(len(user_set), 1)

            # --- Normalised star popularity (0.0 to 1.0) ---
            stars = int(c.get("star_count") or 0)
            norm_stars = min(math.log1p(stars) / max_log_stars, 1.0)

            # --- Final cold-start score ---
            c["final_score"] = (
                skill_weight * skill_match
                + stars_weight * norm_stars
            )
            c["score_source"] = "cold_start"
            # MMoE fields fallback so UI doesn't break
            c["predictions"] = {
                "p_ctr": skill_match,
                "p_save": norm_stars,
                "p_follow": 0.0,
                "pred_dwell_fraction": 0.5,
            }

        candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return candidates

    # ── User profile retrieval ────────────────────────────────────────────────

    def _get_user_profile(self, user_id: str) -> tuple[list[float], list[str]]:
        """Return (interest_vector, skills_list) for a user from Qdrant.

        The point ID is a deterministic UUID5 matching the scheme in
        ``user_onboarding.py:save_to_qdrant``.
        """
        point_uuid = user_point_id(user_id)

        response = self._client.retrieve(
            collection_name=USER_PROFILES_COLLECTION,
            ids=[point_uuid],
            with_vectors=True,
            with_payload=True,
        )

        if not response:
            raise ValueError(
                f"User '{user_id}' (point {point_uuid}) not found in "
                f"Qdrant collection '{USER_PROFILES_COLLECTION}'."
            )

        point = response[0]

        # Extract vector
        if isinstance(point.vector, dict):
            if TARGET_VECTOR_NAME and TARGET_VECTOR_NAME in point.vector:
                user_vector = list(point.vector[TARGET_VECTOR_NAME])
            else:
                vectors = list(point.vector.values())
                if not vectors:
                    raise ValueError(f"User '{user_id}' has an empty named-vector dict.")
                user_vector = list(vectors[0])
        else:
            user_vector = list(point.vector)

        # Extract skills from payload (used by the ranker's skill_match feature)
        payload = point.payload or {}
        skills_raw = payload.get("skills") or []
        tech_raw = payload.get("tech_stack") or []
        if isinstance(skills_raw, str):
            skills_raw = [skills_raw]
        if not isinstance(skills_raw, list):
            skills_raw = list(skills_raw) if isinstance(skills_raw, (tuple, set)) else []
        if isinstance(tech_raw, str):
            tech_raw = [tech_raw]
        if not isinstance(tech_raw, list):
            tech_raw = list(tech_raw) if isinstance(tech_raw, (tuple, set)) else []
        skills = skills_raw + tech_raw

        return user_vector, skills

    def _get_user_data(self, user_id: str) -> tuple[list[float], dict[str, Any]]:
        """Retrieve both the vector and payload for a user deterministic UUID."""
        point_uuid = user_point_id(user_id)

        response = self._client.retrieve(
            collection_name=USER_PROFILES_COLLECTION,
            ids=[point_uuid],
            with_vectors=True,
            with_payload=True,
        )

        if not response:
            raise ValueError(
                f"User '{user_id}' (point {point_uuid}) not found in "
                f"Qdrant collection '{USER_PROFILES_COLLECTION}'."
            )

        point = response[0]
        payload = point.payload or {}

        if isinstance(point.vector, dict):
            if TARGET_VECTOR_NAME and TARGET_VECTOR_NAME in point.vector:
                return list(point.vector[TARGET_VECTOR_NAME]), payload
            
            vectors = list(point.vector.values())
            if not vectors:
                raise ValueError(f"User '{user_id}' has an empty named-vector dict.")
            return list(vectors[0]), payload

        return list(point.vector), payload

    def _get_user_vector(self, user_id: str) -> list[float]:
        """Retrieve the user's interest embedding from the user_profiles collection."""
        vector, _ = self._get_user_data(user_id)
        return vector

    # ── Candidate retrieval ───────────────────────────────────────────────────

    def _retrieve_candidates(
        self,
        user_vector: list[float],
        user_skills: list[str],
    ) -> list[dict[str, Any]]:
        """Pull Person 3's Qdrant-only semantic and discovery candidate pool."""
        retriever = self.candidate_retriever
        if retriever is None:
            logger.warning(
                "CandidateRetriever unavailable. No candidates to rank."
            )
            return []

        try:
            records = retriever.retrieve_candidates(
                user_embedding=user_vector,
                user_interests=user_skills,
            )
        except Exception as exc:
            logger.warning("Candidate retrieval failed: %s", exc)
            raise

        candidates = self._normalize_retriever_candidates(records)
        logger.info("Qdrant retrieval returned %d unique candidates.", len(candidates))
        return candidates

    # ── MMoE Ranking ──────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_fallback(
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return copies ordered by their Qdrant retrieval scores."""
        fallback: list[dict[str, Any]] = []
        for candidate in candidates:
            item = dict(candidate)
            item["final_score"] = float(item.get("retrieval_score") or 0.0)
            item["predictions"] = {}
            item["score_source"] = "cosine_fallback"
            fallback.append(item)
        fallback.sort(key=lambda item: item["final_score"], reverse=True)
        return fallback

    def _rank_candidates(
        self,
        user_vector: list[float],
        user_skills: list[str],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score and sort candidates with the MMoE heavy ranker.

        All candidates have repository vectors supplied by Person 3's public
        retriever and are passed through the MMoE network uniformly.

        Each candidate dict is enriched with:
        - ``final_score``   — raw weighted value-function output (up to 28.1)
        - ``predictions``   — raw per-task probabilities (p_ctr, p_save, …)
        - ``score_source``  — "mmoe_{source}" or "cosine_fallback" (if ranker unavailable)
        """
        if not candidates:
            return []

        ranker = self.ranker

        if ranker is None:
            logger.warning(
                "RankerService unavailable.  Returning candidates in "
                "retrieval order (cosine score)."
            )
            return self._cosine_fallback(candidates)

        user_emb = np.array(user_vector, dtype=np.float32)

        # ── Build ranker inputs for all candidates ────────────────────────────
        ranker_inputs: list[dict] = []
        for c in candidates:
            topics = c.get("topics") or []
            if isinstance(topics, str):
                try:
                    topics = json.loads(topics)
                except Exception:
                    topics = []

            languages: list[str] = []
            lang = c.get("primary_language")
            if lang:
                languages.append(str(lang))
            payload_languages = c.get("languages") or []
            if isinstance(payload_languages, dict):
                languages += [str(value) for value in payload_languages.keys()]
            elif isinstance(payload_languages, (list, tuple, set)):
                languages += [str(value) for value in payload_languages]
            elif isinstance(payload_languages, str):
                languages.append(payload_languages)
            lang_used = c.get("language_used") or {}
            if isinstance(lang_used, dict):
                languages += list(lang_used.keys())
            elif isinstance(lang_used, list):
                languages += [str(l) for l in lang_used]
            languages = list(dict.fromkeys(languages))

            repo_emb_raw = c.get("repo_embedding") or []
            repo_emb = np.array(repo_emb_raw, dtype=np.float32) if repo_emb_raw else np.zeros(ranker.emb_dim, dtype=np.float32)
            norm = np.linalg.norm(repo_emb)
            if norm > 1e-6:
                repo_emb = repo_emb / norm

            import math
            daily_stars = float(c.get("daily_stars") or 0.0)
            if daily_stars > 0:
                trend_vel = min(math.log1p(daily_stars) / math.log1p(500.0), 1.0)
            else:
                trend_vel = float(c.get("trend_velocity") or 0.0)

            readme_length = c.get("readme_length")
            if readme_length is None:
                readme_length = len(c.get("readme_summary") or "") or 1000
            pushed_days_ago = c.get("pushed_days_ago")
            if pushed_days_ago is None:
                pushed_days_ago = 365

            ranker_inputs.append({
                "id":                c.get("repo_id") or c.get("full_name", "unknown"),
                "embedding":         repo_emb,
                "doc_quality":       c.get("doc_quality", 0.5),
                "code_health":       c.get("code_health", 0.5),
                "readme_length":     int(readme_length),
                "star_count":        int(c.get("star_count") or 0),
                "fork_count":        int(c.get("forks_count") or c.get("fork_count") or 0),
                "open_issues_count": int(c.get("open_issues_count") or 0),
                "pushed_days_ago":   int(pushed_days_ago),
                "activity_score":    float(c.get("activity_score") or 0.0),
                "trend_velocity":    trend_vel,
                "languages":         languages,
                "topics":            topics,
                "tags":              topics,
            })

        # ── Run MMoE on all candidates ────────────────────────────────────────
        try:
            scored = ranker.score_batch(user_emb, user_skills, ranker_inputs)
        except Exception as exc:
            logger.error("RankerService.score_batch failed: %s. Falling back to cosine order.", exc)
            return self._cosine_fallback(candidates)

        id_to_score: dict[str, dict] = {s["repo_id"]: s for s in scored}
        expected_repo_ids = {str(item["id"]) for item in ranker_inputs}
        if set(id_to_score) != expected_repo_ids:
            logger.warning(
                "Ranker returned %d/%d candidate scores; falling back to "
                "Qdrant retrieval order.",
                len(id_to_score),
                len(expected_repo_ids),
            )
            return self._cosine_fallback(candidates)

        # ── Merge scores back ─────────────────────────────────────────────────
        enriched: list[dict[str, Any]] = []
        for c, inp in zip(candidates, ranker_inputs):
            c_copy = dict(c)
            score_entry = id_to_score.get(inp["id"], {})
            preds = score_entry.get("predictions", {})

            source = c.get("retrieval_source", "unknown")
            c_copy["final_score"] = score_entry.get("final_score", 0.0)
            c_copy["predictions"] = preds
            c_copy["score_source"] = f"mmoe_{source}"
            c_copy["languages"] = inp.get("languages", [])
            enriched.append(c_copy)

        enriched.sort(key=lambda x: x["final_score"], reverse=True)

        logger.info(
            "RankerService scored %d candidates. Top score: %.4f",
            len(enriched),
            enriched[0]["final_score"] if enriched else 0.0,
        )
        return enriched

    # ── Utility: list onboarded users ─────────────────────────────────────────

    def list_onboarded_users(self, batch_size: int = 100) -> list[dict[str, Any]]:
        """Scroll the user_profiles collection and return all user metadata."""
        users = []
        next_offset = None

        while True:
            try:
                records, next_offset = self._client.scroll(
                    collection_name=USER_PROFILES_COLLECTION,
                    limit=batch_size,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                if "Not found" in str(exc) or "doesn't exist" in str(exc):
                    return users
                logger.error("Qdrant scroll failed: %s", exc)
                raise

            for record in records:
                payload = record.payload or {}
                users.append({
                    "point_id": str(record.id),
                    "user_id": payload.get("user_id", "unknown"),
                    "skills": payload.get("skills", []),
                    "interests": payload.get("interests", []),
                })

            if next_offset is None:
                break

        return users


# ══════════════════════════════════════════════════════════════════════════════
#  MANUAL TEST
# ══════════════════════════════════════════════════════════════════════════════

def _print_batch(name: str, batch: list[dict[str, Any]]) -> None:
    """Pretty-print v2 items or a legacy sanitized batch for inspection."""
    if not batch:
        print(f"  {name}: (empty)")
        return
    print(f"  {name}  ({len(batch)} repos)")
    print(f"  {'#':<3} {'Score':>8}  {'Source':<18} {'Repo':<42} {'Category'}")
    print(f"  {'-'*3} {'-'*8}  {'-'*18} {'-'*42} {'-'*28}")
    for i, item in enumerate(batch, 1):
        raw_score = item.get("score")
        if raw_score is None:
            raw_score = item.get("final_score")
        score = float(raw_score or 0.0)
        source = str(item.get("source") or item.get("score_source") or "?")[:18]
        print(
            f"  {i:<3} {score:>8.4f}  {source:<18} "
            f"{(item.get('full_name') or item.get('repo_id') or '?'):<42} "
            f"{item.get('category') or item.get('primary_language') or ''}"
        )
    print()


def main() -> None:
    """Run the Qdrant-to-ranker pipeline and print backend v2 responses."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    engine = RetrievalEngine()
    users = engine.list_onboarded_users()

    if not users:
        print("\nNo onboarded users found in Qdrant. Please onboard users first.")
        return

    print(f"\nFound {len(users)} onboarded user(s).  Running retrieval + ranking...\n")
    print("=" * 80)

    for user_info in users:
        user_id = user_info["user_id"]
        interests = ", ".join(user_info.get("interests", [])) or "(none)"
        print(f"\n{'=' * 80}")
        print(f"  User: {user_id}")
        print(f"  Interests: {interests}")
        print(f"{'=' * 80}\n")

        try:
            response = engine.generate_recommendations(
                schema_version=RECOMMENDATION_SCHEMA_VERSION,
                generation_id=str(uuid4()),
                user_id=user_id,
                feed_version=0,
            )
            print(
                f"  Contract: schema={response['schema_version']} "
                f"model={response['model_version']} "
                f"embedding={response['embedding_version']}"
            )
            _print_batch("items", response["items"])

        except Exception as exc:
            print(f"  [FAIL]  Pipeline failed for '{user_id}': {exc}")

    print(f"\n{'=' * 80}")
    print("Done.")


if __name__ == "__main__":
    main()
