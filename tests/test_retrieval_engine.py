from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config import REPOSITORY_EMBEDDING_VERSION
from retrieval_engine import RetrievalEngine


USER_ID = "11111111-1111-4111-8111-111111111111"
MISSING_USER_ID = "11111111-1111-4111-8111-111111111112"
GENERATION_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
REPO_1_ID = "22222222-2222-4222-8222-222222222221"
REPO_2_ID = "22222222-2222-4222-8222-222222222222"
REPO_3_ID = "22222222-2222-4222-8222-222222222223"
REPO_4_ID = "22222222-2222-4222-8222-222222222224"
REPO_5_ID = "22222222-2222-4222-8222-222222222225"


@pytest.fixture
def mock_retrieval_dependencies():
    """Mock the user-profile client and Person 3 candidate retriever."""
    with patch("retrieval_engine.QdrantClient") as mock_qdrant_cls, patch(
        "retrieval_engine.CandidateRetriever"
    ) as mock_retriever_cls:
        mock_qdrant = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant

        mock_retriever = MagicMock()
        mock_retriever.retrieve_candidates.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        yield mock_qdrant, mock_retriever, mock_retriever_cls


def _configure_user_profile(mock_qdrant):
    mock_user_point = MagicMock()
    mock_user_point.vector = [0.1] * 384
    mock_user_point.payload = {
        "user_id": USER_ID,
        "skills": ["Python", "AI/ML"],
    }
    mock_qdrant.retrieve.return_value = [mock_user_point]


def _candidate_pool():
    return [
        {
            "repo_id": REPO_1_ID,
            "full_name": "owner/repo1",
            "retrieval_source": "semantic",
            "retrieval_score": 0.85,
            "repo_embedding": [0.2] * 384,
            "embedding": np.array([0.2] * 384, dtype=np.float32),
            "star_count": 100,
            "fork_count": 5,
            "readme_length": 0,
            "pushed_days_ago": 0,
            "primary_language": "Python",
            "languages": ["Python", "C"],
        },
        {
            "repo_id": REPO_2_ID,
            "full_name": "owner/repo2",
            "retrieval_source": "semantic",
            "retrieval_score": 0.80,
            "repo_embedding": [0.3] * 384,
            "star_count": 200,
            "primary_language": "Python",
        },
        {
            "repo_id": REPO_3_ID,
            "full_name": "owner/repo3",
            "retrieval_source": "trending",
            "retrieval_score": 0.75,
            "repo_embedding": [0.4] * 384,
            "star_count": 300,
            "primary_language": "JavaScript",
        },
    ]


def _mock_ranker():
    ranker = MagicMock()
    ranker.emb_dim = 384
    ranker.score_batch.return_value = [
        {
            "repo_id": REPO_2_ID,
            "final_score": np.float32(10.5),
            "predictions": {"p_follow": np.float32(0.525)},
        },
        {
            "repo_id": REPO_3_ID,
            "final_score": np.float32(5.2),
            "predictions": {"p_follow": np.float32(0.520)},
        },
        {
            "repo_id": REPO_1_ID,
            "final_score": np.float32(1.1),
            "predictions": {"p_follow": np.float32(0.055)},
        },
    ]
    return ranker


def _ranked_candidates(count):
    return [
        {
            "repo_id": f"33333333-3333-4333-8333-{index:012d}",
            "final_score": float(count - index),
            "primary_language": f"language-{index}",
        }
        for index in range(count)
    ]


def _fetch_with_ranked_candidates(engine, ranked):
    with patch.object(
        engine,
        "_get_user_profile",
        return_value=([0.1] * 384, ["Python"]),
    ), patch.object(
        engine,
        "_retrieve_candidates",
        return_value=ranked,
    ), patch.object(engine, "_rank_candidates", return_value=ranked):
        return engine.fetch_onboarding_batches(USER_ID)


def test_retrieval_engine_lazy_loading(mock_retrieval_dependencies):
    """Verify the candidate retriever and ranker references start unloaded."""
    _, mock_retriever, mock_retriever_cls = mock_retrieval_dependencies
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")

    assert engine._candidate_retriever is None
    assert engine._ranker is None
    assert engine.candidate_retriever is mock_retriever

    mock_retriever_cls.assert_called_once_with(
        qdrant_url="http://localhost:6333",
        qdrant_api_key=engine._api_key,
    )


def test_retrieval_engine_fetch_and_rank(mock_retrieval_dependencies):
    """Fetch and rank using Person 3's retriever and RankerService."""
    mock_qdrant, mock_retriever, _ = mock_retrieval_dependencies
    _configure_user_profile(mock_qdrant)
    mock_retriever.retrieve_candidates.return_value = _candidate_pool()

    mock_ranker = _mock_ranker()
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")
    engine._ranker = mock_ranker

    batches = engine.fetch_onboarding_batches(USER_ID)

    mock_retriever.retrieve_candidates.assert_called_once_with(
        user_embedding=[0.1] * 384,
        user_interests=["Python", "AI/ML"],
    )

    assert mock_ranker.score_batch.call_count == 1
    call_args = mock_ranker.score_batch.call_args.args
    np.testing.assert_array_almost_equal(
        call_args[0],
        np.array([0.1] * 384, dtype=np.float32),
    )
    assert call_args[1] == ["Python", "AI/ML"]
    first_ranker_candidate = call_args[2][0]
    assert first_ranker_candidate["languages"] == ["Python", "C"]
    assert first_ranker_candidate["fork_count"] == 5
    assert first_ranker_candidate["readme_length"] == 0
    assert first_ranker_candidate["pushed_days_ago"] == 0

    batch_1 = batches["batch_1"]
    assert len(batch_1) == 3
    assert [item["repo_id"] for item in batch_1] == [
        REPO_2_ID,
        REPO_3_ID,
        REPO_1_ID,
    ]
    assert batch_1[0]["final_score"] == pytest.approx(10.5)
    assert batch_1[1]["final_score"] == pytest.approx(5.2)
    assert batch_1[2]["final_score"] == pytest.approx(1.1)


def test_fetch_onboarding_batches_filters_seen_repos(mock_retrieval_dependencies):
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")
    ranked = [
        {
            "repo_id": REPO_1_ID,
            "final_score": 3.0,
            "primary_language": "Python",
        },
        {
            "repo_id": REPO_2_ID,
            "final_score": 2.0,
            "primary_language": "Go",
        },
        {
            "repo_id": REPO_3_ID,
            "final_score": 1.0,
            "primary_language": "Rust",
        },
    ]

    with patch.object(
        engine,
        "_get_user_profile",
        return_value=([0.1] * 384, ["Python"]),
    ), patch.object(
        engine,
        "_retrieve_candidates",
        return_value=[{"repo_id": "candidate"}],
    ), patch.object(engine, "_rank_candidates", return_value=ranked):
        batches = engine.fetch_onboarding_batches(
            USER_ID,
            seen_repo_ids={REPO_2_ID},
        )

    all_repos = [
        item["repo_id"]
        for batch in batches.values()
        for item in batch
    ]
    assert REPO_2_ID not in all_repos
    assert set(all_repos) == {REPO_1_ID, REPO_3_ID}


def test_engine_works_without_database_url(
    mock_retrieval_dependencies,
    monkeypatch,
):
    mock_qdrant, mock_retriever, _ = mock_retrieval_dependencies
    monkeypatch.delenv("DATABASE_URL", raising=False)
    _configure_user_profile(mock_qdrant)
    mock_retriever.retrieve_candidates.return_value = _candidate_pool()

    engine = RetrievalEngine(qdrant_url="http://localhost:6333")
    engine._ranker = _mock_ranker()
    batches = engine.fetch_onboarding_batches(USER_ID)

    assert len(batches["batch_1"]) == 3
    assert not hasattr(engine, "_db")
    assert not hasattr(engine, "db")


def test_cold_start_uses_retriever_not_postgres(mock_retrieval_dependencies):
    _, mock_retriever, _ = mock_retrieval_dependencies
    candidates = [
        {
            "repo_id": REPO_4_ID,
            "full_name": "owner/python-project",
            "retrieval_score": 0.0,
            "repo_embedding": [0.2] * 384,
            "primary_language": "Python",
            "languages": ["Python"],
            "topics": ["machine-learning"],
            "star_count": 1_000,
        },
        {
            "repo_id": REPO_5_ID,
            "full_name": "owner/go-project",
            "retrieval_score": 0.0,
            "repo_embedding": [0.3] * 384,
            "primary_language": "Go",
            "languages": ["Go"],
            "topics": [],
            "star_count": 500,
        },
    ]
    mock_retriever.retrieve_candidates.return_value = candidates
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")

    batches = engine._cold_start_pipeline(
        USER_ID,
        [],
        ["Python"],
    )

    mock_retriever.retrieve_candidates.assert_called_once_with(
        user_embedding=[],
        user_interests=["Python"],
    )
    assert len(batches["batch_1"]) == 2
    assert batches["batch_1"][0]["repo_id"] == REPO_4_ID
    assert all(
        item["score_source"] == "cold_start"
        for item in batches["batch_1"]
    )


def test_empty_candidate_pool_returns_empty_batches(mock_retrieval_dependencies):
    mock_qdrant, mock_retriever, _ = mock_retrieval_dependencies
    _configure_user_profile(mock_qdrant)
    mock_retriever.retrieve_candidates.return_value = []
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")

    batches = engine.fetch_onboarding_batches(USER_ID)

    assert batches == {
        "batch_1": [],
        "batch_2": [],
        "batch_3": [],
    }


def test_batch_slicing_45_candidates(mock_retrieval_dependencies):
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")

    batches = _fetch_with_ranked_candidates(engine, _ranked_candidates(45))

    assert len(batches["batch_1"]) == 15
    assert len(batches["batch_2"]) == 15
    assert len(batches["batch_3"]) == 15


def test_batch_slicing_30_candidates(mock_retrieval_dependencies):
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")

    batches = _fetch_with_ranked_candidates(engine, _ranked_candidates(30))

    assert len(batches["batch_1"]) == 15
    assert len(batches["batch_2"]) == 15
    assert batches["batch_3"] == []


def test_batch_slicing_10_candidates(mock_retrieval_dependencies):
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")

    batches = _fetch_with_ranked_candidates(engine, _ranked_candidates(10))

    assert len(batches["batch_1"]) == 10
    assert batches["batch_2"] == []
    assert batches["batch_3"] == []


def test_ranker_unavailable_returns_cosine_order(mock_retrieval_dependencies):
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")
    engine._ranker_failed = True
    candidates = [
        {"repo_id": REPO_1_ID, "retrieval_score": 0.2},
        {"repo_id": REPO_2_ID, "retrieval_score": 0.9},
        {"repo_id": REPO_3_ID, "retrieval_score": 0.4},
        {"repo_id": REPO_4_ID, "retrieval_score": 0.7},
        {"repo_id": REPO_5_ID, "retrieval_score": 0.1},
    ]

    ranked = engine._rank_candidates([0.0] * 384, [], candidates)

    assert [item["repo_id"] for item in ranked] == [
        REPO_2_ID,
        REPO_4_ID,
        REPO_3_ID,
        REPO_1_ID,
        REPO_5_ID,
    ]
    assert all(item["score_source"] == "cosine_fallback" for item in ranked)


def test_ranker_without_loaded_model_returns_cosine_order(
    mock_retrieval_dependencies,
):
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")
    ranker = MagicMock()
    ranker.emb_dim = 384
    ranker.score_batch.return_value = []
    engine._ranker = ranker
    candidates = [
        {
            "repo_id": REPO_1_ID,
            "repo_embedding": [0.1] * 384,
            "retrieval_score": 0.2,
        },
        {
            "repo_id": REPO_2_ID,
            "repo_embedding": [0.2] * 384,
            "retrieval_score": 0.9,
        },
        {
            "repo_id": REPO_3_ID,
            "repo_embedding": [0.3] * 384,
            "retrieval_score": 0.4,
        },
    ]

    ranked = engine._rank_candidates([0.0] * 384, [], candidates)

    assert [item["repo_id"] for item in ranked] == [
        REPO_2_ID,
        REPO_3_ID,
        REPO_1_ID,
    ]
    assert all(item["score_source"] == "cosine_fallback" for item in ranked)


def test_qdrant_user_profile_not_found_raises_valueerror(
    mock_retrieval_dependencies,
):
    mock_qdrant, _, _ = mock_retrieval_dependencies
    mock_qdrant.retrieve.return_value = []
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")

    with pytest.raises(ValueError, match="not found"):
        engine.fetch_onboarding_batches(MISSING_USER_ID)


def test_cold_start_flag_bypasses_missing_profile(mock_retrieval_dependencies):
    mock_qdrant, mock_retriever, _ = mock_retrieval_dependencies
    mock_qdrant.retrieve.return_value = []
    mock_retriever.retrieve_candidates.return_value = []
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")

    batches = engine.fetch_onboarding_batches(
        USER_ID,
        is_cold_start=True,
    )

    assert batches == {
        "batch_1": [],
        "batch_2": [],
        "batch_3": [],
    }


def test_batch_items_do_not_contain_embeddings(mock_retrieval_dependencies):
    mock_qdrant, mock_retriever, _ = mock_retrieval_dependencies
    _configure_user_profile(mock_qdrant)
    mock_retriever.retrieve_candidates.return_value = _candidate_pool()
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")
    engine._ranker = _mock_ranker()

    batches = engine.fetch_onboarding_batches(USER_ID)

    for batch in batches.values():
        for item in batch:
            assert "repo_embedding" not in item
            assert "embedding" not in item


def test_batch_predictions_are_plain_floats(mock_retrieval_dependencies):
    mock_qdrant, mock_retriever, _ = mock_retrieval_dependencies
    _configure_user_profile(mock_qdrant)
    mock_retriever.retrieve_candidates.return_value = _candidate_pool()
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")
    engine._ranker = _mock_ranker()

    batches = engine.fetch_onboarding_batches(USER_ID)

    predictions = [
        item["predictions"]
        for batch in batches.values()
        for item in batch
    ]
    assert predictions
    assert all(
        type(value) is float
        for prediction in predictions
        for value in prediction.values()
    )


def test_generate_recommendations_matches_backend_v2_schema(
    mock_retrieval_dependencies,
):
    mock_qdrant, mock_retriever, _ = mock_retrieval_dependencies
    _configure_user_profile(mock_qdrant)
    mock_retriever.retrieve_candidates.return_value = _candidate_pool()
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")
    engine._ranker = _mock_ranker()

    response = engine.generate_recommendations(
        schema_version=2,
        generation_id=GENERATION_ID,
        user_id=USER_ID,
        feed_version=12,
        limit=2,
    )

    assert response == {
        "schema_version": 2,
        "generation_id": GENERATION_ID,
        "user_id": USER_ID,
        "feed_version": 12,
        "model_version": "heavy-ranker-v1",
        "embedding_version": REPOSITORY_EMBEDDING_VERSION,
        "items": [
            {
                "repo_id": REPO_2_ID,
                "score": pytest.approx(10.5),
                "source": "personalized",
            },
            {
                "repo_id": REPO_3_ID,
                "score": pytest.approx(5.2),
                "source": "personalized",
            },
        ],
    }
    assert all(set(item) == {"repo_id", "score", "source"} for item in response["items"])
    assert all(type(item["score"]) is float for item in response["items"])


def test_generate_recommendations_applies_backend_feedback_exclusions(
    mock_retrieval_dependencies,
):
    mock_qdrant, mock_retriever, _ = mock_retrieval_dependencies
    _configure_user_profile(mock_qdrant)
    mock_retriever.retrieve_candidates.return_value = _candidate_pool()
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")
    engine._ranker = _mock_ranker()

    response = engine.generate_recommendations(
        schema_version=2,
        generation_id=GENERATION_ID,
        user_id=USER_ID,
        feed_version=12,
        limit=3,
        seen_repo_ids={REPO_2_ID},
    )

    assert [item["repo_id"] for item in response["items"]] == [
        REPO_3_ID,
        REPO_1_ID,
    ]


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"schema_version": 1}, "schema_version"),
        ({"generation_id": "not-a-uuid"}, "generation_id"),
        ({"user_id": "owner/name"}, "user_id"),
        ({"feed_version": -1}, "feed_version"),
        ({"limit": 46}, "limit"),
    ],
)
def test_generate_recommendations_rejects_non_v2_contract(
    mock_retrieval_dependencies,
    overrides,
    error,
):
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")
    request = {
        "schema_version": 2,
        "generation_id": GENERATION_ID,
        "user_id": USER_ID,
        "feed_version": 1,
    }
    request.update(overrides)

    with pytest.raises((TypeError, ValueError), match=error):
        engine.generate_recommendations(**request)


def test_retriever_candidate_requires_backend_repo_uuid():
    record = {
        "repo_id": "owner/repository",
        "full_name": "owner/repository",
        "payload": {"repo_id": "owner/repository"},
        "vector": [0.1] * 384,
        "score": 0.9,
    }

    assert RetrievalEngine._normalize_retriever_candidate(record) is None


def test_invalid_static_fallback_signals_retrieval_failure(
    mock_retrieval_dependencies,
):
    _, mock_retriever, _ = mock_retrieval_dependencies
    mock_retriever.retrieve_candidates.return_value = [
        {
            "repo_id": "facebook/react",
            "full_name": "facebook/react",
            "repo_embedding": [0.0] * 384,
            "retrieval_source": "fallback",
            "retrieval_score": 0.0,
        }
    ]
    engine = RetrievalEngine(qdrant_url="http://localhost:6333")

    with pytest.raises(RuntimeError, match="UUID contract"):
        engine._retrieve_candidates([0.1] * 384, ["Python"])
