from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from embedding.qdrant_store import QdrantRepositoryStore


def build_report(limit: int = 100_000) -> dict:
    store = QdrantRepositoryStore()
    points = store.list_points(limit=limit, with_vectors=False)
    canonical = []
    legacy = []
    duplicates: dict[str, list[str]] = {}
    for point in points:
        payload_id = str(point.get("repo_id") or "")
        point_id = str(point.get("id") or "")
        try:
            normalized = str(uuid.UUID(payload_id))
        except ValueError:
            legacy.append({"point_id": point_id, "repo_id": payload_id, "reason": "payload_not_uuid"})
            continue
        duplicates.setdefault(normalized, []).append(point_id)
        if point_id == normalized:
            canonical.append(point_id)
        else:
            legacy.append({"point_id": point_id, "repo_id": normalized, "reason": "point_id_mismatch"})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scanned": len(points),
        "canonical": len(canonical),
        "legacy": legacy,
        "duplicate_identities": {key: value for key, value in duplicates.items() if len(value) > 1},
        "safe_to_cut_over": not legacy and all(len(value) == 1 for value in duplicates.values()),
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
