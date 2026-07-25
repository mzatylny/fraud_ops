import math

import pytest

from src.decision_engine import make_decision


def test_decision_thresholds():
    assert make_decision(0.10) == "APPROVE"
    assert make_decision(0.35) == "REVIEW"
    assert make_decision(0.84) == "REVIEW"
    assert make_decision(0.85) == "BLOCK"


@pytest.mark.parametrize("score", [-0.01, 1.01, math.nan, math.inf])
def test_invalid_risk_scores_fail_closed(score):
    with pytest.raises(ValueError):
        make_decision(score)


def test_non_numeric_risk_score_is_rejected():
    with pytest.raises(TypeError):
        make_decision("0.5")
