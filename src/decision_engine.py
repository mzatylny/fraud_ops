"""Risk routing thresholds for the operational decision engine."""

from .config import BLOCK_THRESHOLD, REVIEW_THRESHOLD


def make_decision(risk_score: float) -> str:
    if risk_score >= BLOCK_THRESHOLD:
        return "BLOCK"
    if risk_score >= REVIEW_THRESHOLD:
        return "REVIEW"
    return "APPROVE"
