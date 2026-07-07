from src.decision_engine import make_decision


def test_decision_thresholds():
    assert make_decision(0.10) == "APPROVE"
    assert make_decision(0.35) == "REVIEW"
    assert make_decision(0.84) == "REVIEW"
    assert make_decision(0.85) == "BLOCK"
