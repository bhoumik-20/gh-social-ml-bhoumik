import os
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient
from app import app
from inference.feed_assembly import FeedAssemblySystem

client = TestClient(app)

@pytest.mark.unit
class TestFeedAssemblySystem:
    """Unit tests for the FeedAssemblySystem class."""

    def test_process_feed_assembly_preserves_zero_scores(self):
        candidates = [
            {"repo_id": "repo-zero", "final_score": 0.0},
            {"repo_id": "repo-positive", "final_score": 0.1},
        ]

        ordered_ids = FeedAssemblySystem.process_feed_assembly(
            candidates,
            target_size=2,
        )

        assert ordered_ids == ["repo-positive", "repo-zero"]

    def test_shape_batch_empty_input_returns_empty(self):
        assert FeedAssemblySystem().shape_batch([]) == []

    def test_shape_batch_with_no_seen_repos(self):
        ranked = [
            {"repo_id": "repo-1", "final_score": 2.0},
            {"repo_id": "repo-2", "final_score": 1.0},
        ]

        result = FeedAssemblySystem().shape_batch(ranked)

        assert [item["repo_id"] for item in result] == ["repo-1", "repo-2"]

    def test_shape_batch_all_repos_seen_returns_empty(self):
        ranked = [
            {"repo_id": "repo-1", "final_score": 2.0},
            {"repo_id": "repo-2", "final_score": 1.0},
        ]

        result = FeedAssemblySystem().shape_batch(
            ranked,
            seen_repo_ids={"repo-1", "repo-2"},
        )

        assert result == []

    def test_shape_batch_removes_seen_repos(self):
        ranked = [
            {"repo_id": "repo-1", "final_score": 2.0},
            {"repo_id": "repo-2", "final_score": 1.0},
        ]

        result = FeedAssemblySystem().shape_batch(
            ranked,
            seen_repo_ids={"repo-1"},
        )

        assert [item["repo_id"] for item in result] == ["repo-2"]

    def test_shape_batch_diversity_cap(self):
        ranked = [
            {
                "repo_id": f"repo-{i}",
                "final_score": 10.0 - i,
                "primary_language": "Python",
            }
            for i in range(10)
        ]

        with patch("random.shuffle"):
            result = FeedAssemblySystem().shape_batch(ranked)

        assert [item["repo_id"] for item in result[:5]] == [
            f"repo-{i}" for i in range(5)
        ]
        assert {item["repo_id"] for item in result[5:]} == {
            f"repo-{i}" for i in range(5, 10)
        }

    def test_shape_batch_freshness_boost_promotes_fresh_repo(self):
        now = datetime.now(timezone.utc)
        ranked = [
            {
                "repo_id": "repo-old",
                "final_score": 10.1,
                "created_at": now - timedelta(days=10),
            },
            {
                "repo_id": "repo-fresh",
                "final_score": 10.0,
                "created_at": now - timedelta(hours=1),
            },
        ]

        result = FeedAssemblySystem().shape_batch(ranked)

        assert result[0]["repo_id"] == "repo-fresh"
        assert result[0]["final_score"] > result[1]["final_score"]

    def test_shape_batch_exploration_shuffles_tail(self):
        now = datetime.now(timezone.utc)
        ranked = [
            {
                "repo_id": f"repo-{i}",
                "final_score": 100.0 - i,
                "created_at": now - timedelta(days=10),
                "primary_language": f"language-{i}",
            }
            for i in range(15)
        ]

        with patch("random.shuffle") as mock_shuffle:
            result = FeedAssemblySystem().shape_batch(ranked)

        mock_shuffle.assert_called_once()
        shuffled_tail = mock_shuffle.call_args.args[0]
        assert [item["repo_id"] for item in shuffled_tail] == [
            f"repo-{i}" for i in range(10, 15)
        ]
        assert [item["repo_id"] for item in result[:10]] == [
            f"repo-{i}" for i in range(10)
        ]

    def test_process_feed_assembly_empty(self):
        """Test that passing an empty list of candidates returns an empty list."""
        assert FeedAssemblySystem.process_feed_assembly([], target_size=15) == []
        assert FeedAssemblySystem.process_feed_assembly([{"repo_id": "1"}], target_size=0) == []

    def test_process_feed_assembly_freshness_boost(self):
        """Test that fresh repositories get a score boost and rise in rank."""
        now = datetime.now(timezone.utc)
        
        # We create 15 repos.
        # Repo 1 has base score 10.0 and was created 1 hour ago (gets high boost).
        # Repo 2 has base score 10.1 and was created 10 days ago (no boost).
        candidates = [
            {
                "repo_id": "repo-fresh",
                "final_score": 10.0,
                "created_at": now - timedelta(hours=1)
            },
            {
                "repo_id": "repo-old",
                "final_score": 10.1,
                "created_at": now - timedelta(days=10)
            }
        ]
        # Pad up to 15 repositories to meet the explore-split minimum checks
        for i in range(13):
            candidates.append({
                "repo_id": f"pad-{i}",
                "final_score": 5.0,
                "created_at": now - timedelta(days=10)
            })

        # Process assembly
        ordered_ids = FeedAssemblySystem.process_feed_assembly(candidates, target_size=15)
        
        # repo-fresh (10.0 base + boost ~0.25) should exceed repo-old (10.1 base)
        # and end up as the first repository in the ordered list!
        assert ordered_ids[0] == "repo-fresh"
        assert ordered_ids[1] == "repo-old"

    def test_process_feed_assembly_exploration_injection(self):
        """Test that exploration injection shuffles the bottom-tier repositories."""
        now = datetime.now(timezone.utc)
        candidates = []
        for i in range(15):
            candidates.append({
                "repo_id": f"repo-{i}",
                "final_score": 100.0 - i, # strictly descending
                "created_at": now - timedelta(days=10)
            })

        # If exploration count is target // 3 (which is 15 // 3 = 5 repos),
        # then the top 10 repos (0 to 9) should remain stable in their exploit tier,
        # while the bottom 5 repos (10 to 14) are shuffled.
        # Let's mock random.shuffle to verify it's called on the last 5 repos.
        with patch("random.shuffle") as mock_shuffle:
            ordered_ids = FeedAssemblySystem.process_feed_assembly(candidates, target_size=15)
            
            # Shuffling should be called on the tail of 5 repos
            mock_shuffle.assert_called_once()
            called_args = mock_shuffle.call_args[0][0]
            assert len(called_args) == 5
            assert {item["repo_id"] for item in called_args} == {f"repo-{i}" for i in range(10, 15)}
            
            # The top 10 positions must be preserved exactly
            assert ordered_ids[:10] == [f"repo-{i}" for i in range(10)]


@pytest.mark.unit
class TestFeedAssemblyApi:
    """Integration/API tests for the /api/internal/ml/assemble-feed endpoint."""

    def test_assemble_feed_endpoint_success(self):
        """Test that a valid 15-candidate request returns 200 OK and ranked IDs."""
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "candidates": [
                {
                    "repo_id": f"repo-{i}",
                    "final_score": 10.0 - i,
                    "created_at": now
                }
                for i in range(15)
            ]
        }
        response = client.post("/api/internal/ml/assemble-feed", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "rankedRepoIds" in data
        assert len(data["rankedRepoIds"]) == 15

    def test_assemble_feed_endpoint_invalid_payload_length(self):
        """Test that requests with less or more than 15 candidates are rejected."""
        now = datetime.now(timezone.utc).isoformat()
        # 14 candidates (invalid length)
        payload = {
            "candidates": [
                {
                    "repo_id": f"repo-{i}",
                    "final_score": 10.0,
                    "created_at": now
                }
                for i in range(14)
            ]
        }
        response = client.post("/api/internal/ml/assemble-feed", json=payload)
        assert response.status_code == 422
