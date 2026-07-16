import pytest

from inference.value_function import (
    VALUE_SCORE_SUM,
    VALUE_WEIGHTS,
    compute_value_score,
)


def test_compute_value_score_with_full_predictions():
    predictions = {
        "p_ctr": 0.5,
        "p_save": 0.3,
        "p_gh": 0.4,
        "pred_dwell_fraction": 0.6,
        "p_follow": 0.2,
    }

    assert compute_value_score(predictions) == pytest.approx(6.86)


def test_compute_value_score_with_empty_predictions():
    assert compute_value_score({}) == 0.0


def test_compute_value_score_defaults_missing_predictions_to_zero():
    assert compute_value_score({"p_save": 0.4, "p_follow": 0.1}) == pytest.approx(4.0)


def test_value_weights_sum_matches_documented_total():
    assert sum(VALUE_WEIGHTS.values()) == VALUE_SCORE_SUM
    assert VALUE_SCORE_SUM == pytest.approx(28.1)
