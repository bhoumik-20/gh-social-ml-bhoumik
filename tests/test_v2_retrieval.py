import uuid
from types import SimpleNamespace

import pytest

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


def test_qdrant_only_retrieval_deduplicates_and_rejects_legacy_identity():
    client = FakeQdrant()
    retriever = QdrantV2Retriever(client=client)
    items = retriever.recommend(client.user_id, 10, [])
    assert {item.repo_id for item in items} == set(client.repo_ids)
    assert len(items) == 2


def test_discovery_score_treats_zero_pushed_days_as_fresh():
    today_score, today_source = QdrantV2Retriever._discovery_score(
        {"pushed_days_ago": 0}
    )
    missing_score, missing_source = QdrantV2Retriever._discovery_score({})

    assert today_source == "fresh"
    assert today_score == pytest.approx(0.1)
    assert missing_source == "popular"
    assert missing_score < today_score
