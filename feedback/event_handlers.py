import logging
import uuid
import numpy as np
import math
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from database.connector import PostgreSQLConnector
from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIM,
    MIN_DWELL_SECONDS,
    MAX_DWELL_SECONDS,
    DWELL_BASE_ALPHA,
)
from scripts.user_onboarding import USER_PROFILES_COLLECTION, TARGET_VECTOR_NAME

logger = logging.getLogger("pipeline.feedback")

# Action-weights for vector adjustment (learning rate \alpha)
ACTION_WEIGHTS = {
    "click": 0.05,
    "like": 0.15,
    "save": 0.20,
    "skip": -0.10,
    "dwell": None,   # dynamic: computed by _dwell_alpha(dwell_seconds)
}

# PostgreSQL column mapping for repository engagement stats.
# "dwell" maps to None — it only updates the Qdrant embedding, not a Postgres counter.
METRIC_COLUMNS = {
    "click": "views_count",
    "like": "likes_count",
    "save": "saves_count",
    "dwell": None,   # no Postgres column — embedding-only signal
}


def _dwell_alpha(dwell_seconds: float) -> Optional[float]:
    """Map raw dwell time to an embedding shift strength (alpha).

    Uses log-linear scaling so that short dwells produce small shifts
    and long engaged reads approach DWELL_BASE_ALPHA.

    Returns
    -------
    float  — learning rate to pass to shift_vector
    None   — dwell is below MIN_DWELL_SECONDS (accidental scroll); skip update
    """
    # The below threshold is for filtering out accidental card flicks that
    # should not influence the interest vector at all.
    if dwell_seconds < MIN_DWELL_SECONDS:
        return None
    # Log-linear: consistent with how trend_velocity is clamped in retrieval_engine.py.
    ratio = math.log1p(dwell_seconds) / math.log1p(MAX_DWELL_SECONDS)
    return DWELL_BASE_ALPHA * min(ratio, 1.0)


def shift_vector(user_vec: List[float], repo_vec: List[float], alpha: float) -> List[float]:
    """Shift user vector towards (or away from) repo vector and L2 normalize."""
    u = np.array(user_vec, dtype=np.float32)
    r = np.array(repo_vec, dtype=np.float32)

    # Shift formula: V_user_new = V_user + \alpha * V_repo
    updated = u + alpha * r

    # Re-normalize to unit length (L2 norm)
    norm = np.linalg.norm(updated)
    if norm > 0:
        updated = updated / norm

    return updated.tolist()


class FeedbackHandler:
    def __init__(
        self,
        db_connector: PostgreSQLConnector | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        self.db = db_connector or PostgreSQLConnector()
        self.qdrant_url = qdrant_url or QDRANT_URL
        self.qdrant_api_key = qdrant_api_key or QDRANT_API_KEY

        self._qdrant_client: QdrantClient | None = None
        if self.qdrant_url:
            try:
                self._qdrant_client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    timeout=30.0,
                )
            except Exception as exc:
                logger.error("Failed to connect to Qdrant inside FeedbackHandler: %s", exc)

    @property
    def qdrant(self) -> QdrantClient | None:
        return self._qdrant_client

    def handle_feedback(
        self,
        user_id: str,
        repo_id: str,
        action: str,
        *,
        dwell_seconds: Optional[float] = None,
    ) -> bool:
        """Process a single feedback event: update Postgres counters and shift Qdrant embedding.

        Parameters
        ----------
        user_id       : Unique user identifier.
        repo_id       : Repository full_name or UUID.
        action        : One of click | like | save | skip | dwell.
        dwell_seconds : Required when action == 'dwell'. Observed time the user
                        spent on the repository card, in seconds.
        """
        action = action.lower()
        if action not in ACTION_WEIGHTS:
            logger.error("Unknown feedback action: %s", action)
            return False

        # Resolve the embedding learning rate (alpha) for this event.
        if action == "dwell":
            if dwell_seconds is None:
                logger.warning(
                    "'dwell' action received without dwell_seconds for user '%s'. Skipping.",
                    user_id,
                )
                return False
            resolved_alpha = _dwell_alpha(float(dwell_seconds))
            if resolved_alpha is None:
                # The below early return is for discarding sub-threshold dwells cleanly
                # without touching Postgres or Qdrant — accidental scroll, not real interest.
                logger.debug(
                    "Dwell %.1fs below MIN_DWELL_SECONDS=%.1fs for user '%s'. Ignored.",
                    dwell_seconds, MIN_DWELL_SECONDS, user_id,
                )
                return True   # not an error — just a no-op
        else:
            resolved_alpha = ACTION_WEIGHTS[action]

        logger.info(
            "Processing feedback: User '%s' -> Repo '%s' [%s] alpha=%.4f",
            user_id, repo_id, action,
            resolved_alpha if resolved_alpha is not None else 0.0,
        )

        # 1. Update PostgreSQL engagement counts (dwell has no column — no-op)
        db_success = self.update_postgres_metrics(repo_id, action)
        if not db_success:
            logger.warning("Failed to update engagement metrics in Postgres for '%s'", repo_id)

        # 2. Update Qdrant user embedding vector using the resolved alpha
        qdrant_success = self.update_user_embedding(user_id, repo_id, resolved_alpha)
        if not qdrant_success:
            logger.warning("Failed to adjust Qdrant profile embedding for user '%s'", user_id)

        # 3. Invalidate the cached feed batches for this user in PostgreSQL
        cache_success = self.invalidate_user_feed_cache(user_id)
        if not cache_success:
            logger.warning("Failed to invalidate feed cache for user '%s'", user_id)

        return db_success and qdrant_success and cache_success

    def update_postgres_metrics(self, repo_id: str, action: str) -> bool:
        """Increment the metric count inside the Repo PostgreSQL table."""
        column = METRIC_COLUMNS.get(action)
        if column is None or not self.db.enabled:
            # 'skip' and 'dwell' have no Postgres counter — treat as success
            return True

        # Guard against SQL injection via strict whitelist validation (defense-in-depth)
        if column not in {"views_count", "likes_count", "saves_count"}:
            logger.error("Forbidden database column update: '%s'", column)
            return False

        conn = None
        try:
            conn = self.db.connect()
            cursor = conn.cursor()

            # Increment count defensively handling NULL values using COALESCE
            query = f"""
            UPDATE Repo
            SET {column} = COALESCE({column}, 0) + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE full_name = %s OR repo_id::text = %s;
            """
            cursor.execute(query, (repo_id, repo_id))
            conn.commit()

            logger.info("Successfully incremented %s count for repo '%s'", column, repo_id)
            return True
        except Exception as exc:
            logger.error("Error updating metrics in Postgres for repo '%s': %s", repo_id, exc)
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return False
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def update_user_embedding(self, user_id: str, repo_id: str, alpha: float) -> bool:
        """Shift the user's Qdrant embedding towards (or away from) a repository vector.

        Parameters
        ----------
        user_id : Unique user identifier.
        repo_id : Repository full_name or UUID.
        alpha   : Signed learning rate.  Positive → shift toward repo (interest).
                  Negative → shift away (disinterest, e.g. skip).
        """
        if not self.qdrant:
            logger.warning("Qdrant client not configured; skipping vector shift.")
            return False

        user_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"user:{user_id}"))
        repo_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"github:{repo_id}"))

        try:
            # 1. Fetch user vector and payload
            user_points = self.qdrant.retrieve(
                collection_name=USER_PROFILES_COLLECTION,
                ids=[user_uuid],
                with_vectors=True,
            )
            if not user_points:
                logger.warning("User '%s' profiles not found in collection '%s'", user_id, USER_PROFILES_COLLECTION)
                return False

            user_point = user_points[0]
            user_payload = user_point.payload or {}

            # Extract user vector
            user_vector = None
            vector_name = None
            if isinstance(user_point.vector, dict):
                if TARGET_VECTOR_NAME and TARGET_VECTOR_NAME in user_point.vector:
                    vector_name = TARGET_VECTOR_NAME
                    user_vector = list(user_point.vector[vector_name])
                else:
                    vectors = list(user_point.vector.values())
                    if not vectors:
                        logger.error("User '%s' has an empty named-vector dict in Qdrant.", user_id)
                        return False
                    vector_name = list(user_point.vector.keys())[0]
                    user_vector = list(vectors[0])
            else:
                user_vector = list(user_point.vector)

            # 2. Fetch repository vector
            repo_points = self.qdrant.retrieve(
                collection_name=QDRANT_COLLECTION_NAME,
                ids=[repo_uuid],
                with_vectors=True,
            )
            if not repo_points:
                logger.warning("Repository '%s' not found in collection '%s'. Skipping embedding adjustment.", repo_id, QDRANT_COLLECTION_NAME)
                return False

            repo_point = repo_points[0]
            
            # Repository vectors are named 'repo_embedding'
            repo_vector = None
            if isinstance(repo_point.vector, dict):
                repo_vector = list(repo_point.vector.get("repo_embedding", []))
            else:
                repo_vector = list(repo_point.vector)

            if not repo_vector or len(repo_vector) != EMBEDDING_DIM:
                logger.error("Repository '%s' embedding dimension mismatch or missing.", repo_id)
                return False

            # 3. Calculate shifted vector
            updated_vector = shift_vector(user_vector, repo_vector, alpha)

            # 4. Save updated vector back to Qdrant, preserving metadata payload
            final_vector = {vector_name: updated_vector} if vector_name is not None else updated_vector
            self.qdrant.upsert(
                collection_name=USER_PROFILES_COLLECTION,
                points=[
                    PointStruct(
                        id=user_uuid,
                        vector=final_vector,
                        payload=user_payload,
                    )
                ]
            )

            logger.info("Successfully adjusted and upserted user '%s' embedding vector in Qdrant.", user_id)
            return True
        except Exception as exc:
            logger.error("Failed to update user vector in Qdrant: %s", exc)
            return False

    def invalidate_user_feed_cache(self, user_id: str) -> bool:
        """Invalidate the cached recommendation batches for this user in PostgreSQL."""
        if not self.db.enabled:
            return True

        conn = None
        try:
            conn = self.db.connect()
            cursor = conn.cursor()

            # Delete cache row for user
            query = "DELETE FROM user_recommendation_batches WHERE user_id = %s;"
            cursor.execute(query, (user_id,))
            conn.commit()

            logger.info("Invalidated recommendation cache for user '%s'", user_id)
            return True
        except Exception as exc:
            logger.error("Failed to delete cache in PostgreSQL for user '%s': %s", user_id, exc)
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return False
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
