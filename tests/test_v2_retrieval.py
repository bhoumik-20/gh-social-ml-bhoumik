import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from embedding.vector_contract import legacy_repository_point_id, legacy_user_point_id
from inference.feed_assembly import FeedAssemblySystem
from retrieval.v2_retriever import QdrantV2Retriever


class FakeQdrant:
    def __init__(self):
        self.user_id = str(uuid.uuid4())
        self.repo_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    def retrieve(self, collection_name, ids, with_vectors, with_payload=True):
        return [SimpleNamespace(id=ids[0], vector=[1.0, 0.0], payload={"last_feedback_version": 0})]

    def query_points(self, **_kwargs):
        point = SimpleNamespace(id=self.repo_ids[0], score=0.9, payload={"repo_id": self.repo_ids[0], "star_count": 50})
        return SimpleNamespace(points=[point])

    def scroll(self, **_kwargs):
        points = [
            SimpleNamespace(id=self.repo_ids[0], payload={"repo_id": self.repo_ids[0], "star_count": 50}),
            SimpleNamespace(id=self.repo_ids[1], payload={"repo_id": self.repo_ids[1], "star_count": 1000, "delta_7d": 20}),
            SimpleNamespace(id=str(uuid.uuid4()), payload={"repo_id": "owner/legacy"}),
        ]
        return points, None


class HeavyRankerQdrant:
    def __init__(self):
        self.user_id = str(uuid.uuid4())
        self.repo_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        self.query_calls = []
        self.scroll_calls = []
        self.vector = [1.0] + [0.0] * 383

    def retrieve(self, **_kwargs):
        return [
            SimpleNamespace(
                id=self.user_id,
                vector=self.vector,
                payload={"skills": ["Python"], "topics": ["recommendation"]},
            )
        ]

    def _point(self, repo_id, *, score=None, stars=0):
        values = {
            "id": repo_id,
            "vector": {"repo_embedding": self.vector},
            "payload": {
                "repo_id": repo_id,
                "content_version": 1,
                "primary_language": "Python",
                "languages": ["Python"],
                "topics": ["recommendation"],
                "star_count": stars,
            },
        }
        if score is not None:
            values["score"] = score
        return SimpleNamespace(**values)

    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        return SimpleNamespace(
            points=[
                self._point(self.repo_ids[0], score=0.95, stars=100),
                self._point(self.repo_ids[1], score=0.50, stars=10),
            ]
        )

    def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        return [
            self._point(self.repo_ids[0], stars=100),
            self._point(self.repo_ids[1], stars=10),
        ], None


class DeterministicHeavyRanker:
    ready = True
    model_version = "heavy-ranker-v1"

    def __init__(self):
        self.calls = []

    def score_batch(self, user_vector, user_skills, candidates):
        self.calls.append((user_vector, user_skills, candidates))
        return [
            {
                "repo_id": candidates[1]["id"],
                "final_score": 0.9,
            },
            {
                "repo_id": candidates[0]["id"],
                "final_score": 0.1,
            },
        ]


def test_qdrant_only_retrieval_deduplicates_and_rejects_legacy_identity():
    client = FakeQdrant()
    retriever = QdrantV2Retriever(client=client)
    items = retriever.recommend(client.user_id, 10, [])
    assert {item.repo_id for item in items} == set(client.repo_ids)
    assert len(items) == 2


def test_qdrant_only_retrieval_reads_pre_v2_uuid5_points():
    client = FakeQdrant()
    user_id = client.user_id
    repo_id = client.repo_ids[0]

    def retrieve(collection_name, ids, with_vectors, with_payload=True):
        assert legacy_user_point_id(user_id) in ids
        return [
            SimpleNamespace(
                id=legacy_user_point_id(user_id),
                vector=[1.0, 0.0],
                payload={"user_id": user_id, "last_feedback_version": 4},
            )
        ]

    def query_points(**_kwargs):
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=legacy_repository_point_id(repo_id),
                    score=0.9,
                    payload={"repo_id": repo_id, "star_count": 50},
                )
            ]
        )

    client.retrieve = retrieve
    client.query_points = query_points
    client.scroll = lambda **_kwargs: ([], None)

    items = QdrantV2Retriever(client=client).recommend(user_id, 10, [])
    assert [item.repo_id for item in items] == [repo_id]


def test_discovery_score_treats_zero_pushed_days_as_fresh():
    today_score, today_source = QdrantV2Retriever._discovery_score(
        {"pushed_days_ago": 0}
    )
    missing_score, missing_source = QdrantV2Retriever._discovery_score({})

    assert today_source == "fresh"
    assert today_score == pytest.approx(0.1)
    assert missing_source == "popular"
    assert missing_score < today_score


def test_v2_retrieval_applies_diversity_freshness_and_deterministic_exploration():
    client = FakeQdrant()
    client.repo_ids = [str(uuid.uuid4()) for _ in range(8)]
    points = []
    for index, repo_id in enumerate(client.repo_ids):
        points.append(
            SimpleNamespace(
                id=repo_id,
                payload={
                    "repo_id": repo_id,
                    "star_count": 100 - index,
                    "primary_language": "Python" if index < 6 else "Rust",
                    "created_at": datetime.now(timezone.utc).isoformat() if index == 7 else None,
                },
            )
        )
    client.retrieve = lambda **_kwargs: []
    client.scroll = lambda **_kwargs: (points, None)

    retriever = QdrantV2Retriever(
        client=client,
        assembler=FeedAssemblySystem(max_same_language=2),
    )
    first = retriever.recommend(client.user_id, 8, [], "fixed-generation")
    second = retriever.recommend(client.user_id, 8, [], "fixed-generation")

    assert first == second
    assert {item.repo_id for item in first} == set(client.repo_ids)
    assert first.index(next(item for item in first if item.repo_id == client.repo_ids[6])) < 6
    assert first[0].repo_id == client.repo_ids[7]


def test_v2_heavy_ranker_reranks_full_vector_candidates_and_reports_served_model():
    client = HeavyRankerQdrant()
    ranker = DeterministicHeavyRanker()
    retriever = QdrantV2Retriever(
        client=client,
        ranker=ranker,
        heavy_ranker_enabled=True,
    )

    batch = retriever.recommend_batch(client.user_id, 2, [], "generation")

    assert batch.ranker_applied is True
    assert batch.model_version == "heavy-ranker-v1-v2-adapter"
    assert [item.repo_id for item in batch.items] == [
        client.repo_ids[1],
        client.repo_ids[0],
    ]
    assert ranker.calls[0][1] == ["Python", "recommendation"]
    assert all(len(candidate["embedding"]) == 384 for candidate in ranker.calls[0][2])


def test_v2_invalid_heavy_output_falls_back_for_only_that_request():
    client = HeavyRankerQdrant()
    ranker = DeterministicHeavyRanker()
    ranker.score_batch = lambda *_args: []
    retriever = QdrantV2Retriever(
        client=client,
        ranker=ranker,
        heavy_ranker_enabled=True,
    )

    batch = retriever.recommend_batch(client.user_id, 2, [])

    assert batch.ranker_applied is False
    assert batch.model_version == "qdrant-hybrid-v2"
    assert "incomplete or invalid" in batch.fallback_reason
    assert [item.repo_id for item in batch.items] == client.repo_ids


def test_v2_filters_eligibility_before_limits_and_uses_ordered_discovery():
    client = HeavyRankerQdrant()
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)

    retriever.recommend(client.user_id, 2, [])

    assert client.query_calls[0]["with_vectors"] == ["repo_embedding"]
    semantic_filter = client.query_calls[0]["query_filter"]
    assert semantic_filter.must[0].key == "content_version"
    assert semantic_filter.must[0].range.gte == 1
    assert {call["order_by"].key for call in client.scroll_calls} == {
        "trend_velocity",
        "activity_score",
        "star_count",
        "pushed_at",
    }
    assert all(
        call["scroll_filter"].must[0].key == "content_version"
        for call in client.scroll_calls
    )
    assert all(
        call["with_vectors"] == ["repo_embedding"] for call in client.scroll_calls
    )


def test_production_health_rejects_a_required_unavailable_ranker(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    retriever = QdrantV2Retriever(
        client=HeavyRankerQdrant(),
        ranker=SimpleNamespace(ready=False),
        heavy_ranker_enabled=True,
    )

    with pytest.raises(RuntimeError, match="required but unavailable"):
        retriever.health()


def test_one_failed_discovery_channel_does_not_fail_recommendations():
    client = HeavyRankerQdrant()
    original_scroll = client.scroll
    calls = 0

    def partially_failing_scroll(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary channel failure")
        return original_scroll(**kwargs)

    client.scroll = partially_failing_scroll
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)

    items = retriever.recommend(client.user_id, 2, [])

    assert len(items) == 2
    assert calls == 4
