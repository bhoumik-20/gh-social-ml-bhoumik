"""Business value scoring for heavy-ranker predictions."""

VALUE_WEIGHTS = {
    "ctr": 1.0,
    "save": 5.0,
    "gh_open": 2.0,
    "dwell": 0.1,
    "follow": 20.0,
}

VALUE_SCORE_SUM = sum(VALUE_WEIGHTS.values())


def compute_value_score(predictions: dict[str, float]) -> float:
    """Return the weighted business value of a prediction dictionary."""
    return (
        VALUE_WEIGHTS["ctr"] * predictions.get("p_ctr", 0.0)
        + VALUE_WEIGHTS["save"] * predictions.get("p_save", 0.0)
        + VALUE_WEIGHTS["gh_open"] * predictions.get("p_gh", 0.0)
        + VALUE_WEIGHTS["dwell"]
        * predictions.get("pred_dwell_fraction", 0.0)
        + VALUE_WEIGHTS["follow"] * predictions.get("p_follow", 0.0)
    )
