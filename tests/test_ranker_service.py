import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from inference.feature_spec import (
    EMBEDDING_DIM,
    FEATURE_COUNT,
    FEATURE_SPEC_VERSION,
    INPUT_DIM,
)
from inference.ranker_service import MMoEHeavyRanker, RankerService


def _write_scaler(path, feature_count=FEATURE_COUNT):
    path.write_text(
        json.dumps(
            {
                "mean": [0.0] * feature_count,
                "scale": [1.0] * feature_count,
            }
        )
    )


def _write_manifest(path, **overrides):
    manifest = {
        "model_version": "heavy-ranker-test",
        "embedding_version": "repo-embedding-test",
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "input_dim": INPUT_DIM,
        "embedding_dim": EMBEDDING_DIM,
        "feature_count": FEATURE_COUNT,
    }
    manifest.update(overrides)
    path.write_text(json.dumps(manifest))


@pytest.fixture
def loaded_ranker_service(tmp_path):
    scaler_path = tmp_path / "feature_scaler.json"
    model_path = tmp_path / "heavy_ranker.pt"
    _write_scaler(scaler_path)
    model_path.touch()

    state_dict = MMoEHeavyRanker(INPUT_DIM).state_dict()
    with patch("inference.ranker_service.torch.load", return_value=state_dict):
        return RankerService(
            model_path=str(model_path),
            scaler_path=str(scaler_path),
        )


def test_ranker_service_rejects_scaler_with_wrong_feature_count(tmp_path):
    scaler_path = tmp_path / "feature_scaler.json"
    _write_scaler(scaler_path, feature_count=FEATURE_COUNT - 1)

    with pytest.raises(ValueError, match="expected 10"):
        RankerService(
            model_path=str(tmp_path / "missing_model.pt"),
            scaler_path=str(scaler_path),
        )


def test_missing_model_disables_scoring(tmp_path, caplog):
    scaler_path = tmp_path / "feature_scaler.json"
    _write_scaler(scaler_path)

    service = RankerService(
        model_path=str(tmp_path / "missing_model.pt"),
        scaler_path=str(scaler_path),
    )

    assert service._model_loaded is False
    with caplog.at_level("WARNING", logger="pipeline.ranker"):
        assert service.score_batch(np.zeros(384), [], [{}]) == []
    assert "model is not loaded" in caplog.text


def test_ranker_service_exposes_manifest_versions(tmp_path):
    scaler_path = tmp_path / "feature_scaler.json"
    manifest_path = tmp_path / "model_manifest.json"
    _write_scaler(scaler_path)
    _write_manifest(manifest_path)

    service = RankerService(
        model_path=str(tmp_path / "missing_model.pt"),
        scaler_path=str(scaler_path),
        manifest_path=str(manifest_path),
    )

    assert service.model_version == "heavy-ranker-test"
    assert service.embedding_version == "repo-embedding-test"


def test_ranker_service_rejects_incompatible_manifest(tmp_path):
    scaler_path = tmp_path / "feature_scaler.json"
    manifest_path = tmp_path / "model_manifest.json"
    _write_scaler(scaler_path)
    _write_manifest(manifest_path, embedding_dim=768)

    with pytest.raises(ValueError, match="Model manifest is incompatible"):
        RankerService(
            model_path=str(tmp_path / "missing_model.pt"),
            scaler_path=str(scaler_path),
            manifest_path=str(manifest_path),
        )


def test_score_batch_empty_candidates_returns_empty(loaded_ranker_service):
    assert loaded_ranker_service.score_batch(
        np.zeros(384, dtype=np.float32),
        ["Python"],
        [],
    ) == []


def test_score_batch_single_candidate_shape(loaded_ranker_service):
    candidate = {
        "id": "repo-1",
        "embedding": np.zeros(384, dtype=np.float32),
        "languages": ["Python"],
        "topics": [],
        "tags": [],
    }
    results = loaded_ranker_service.score_batch(
        np.zeros(384, dtype=np.float32),
        ["Python"],
        [candidate],
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert set(results[0]) == {
        "repo_id",
        "final_score",
        "skill_match",
        "predictions",
    }
    assert results[0]["repo_id"] == "repo-1"
    assert isinstance(results[0]["final_score"], float)
    assert set(results[0]["predictions"]) == {
        "p_ctr",
        "p_save",
        "p_gh",
        "pred_dwell_fraction",
        "p_follow",
    }


def test_score_batch_candidate_missing_optional_fields(loaded_ranker_service):
    candidate = {
        "id": "minimal-repo",
        "embedding": np.zeros(384, dtype=np.float32),
    }

    results = loaded_ranker_service.score_batch(
        np.zeros(384, dtype=np.float32),
        [],
        [candidate],
    )

    assert len(results) == 1
    assert results[0]["repo_id"] == "minimal-repo"


def test_score_batch_preserves_repo_id_order_mapping(loaded_ranker_service):
    candidates = [
        {
            "id": f"repo-{index}",
            "embedding": np.zeros(384, dtype=np.float32),
        }
        for index in range(3)
    ]
    loaded_ranker_service.model = MagicMock(
        return_value=(
            torch.tensor([0.9, 0.8, 0.7]),
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(3),
        )
    )

    results = loaded_ranker_service.score_batch(
        np.zeros(384, dtype=np.float32),
        [],
        candidates,
    )

    assert [result["repo_id"] for result in results] == [
        candidate["id"] for candidate in candidates
    ]
