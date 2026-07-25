"""Risk routing thresholds for the operational decision engine."""

import math

from .config import BLOCK_THRESHOLD, REVIEW_THRESHOLD


def make_decision(risk_score: float) -> str:
    if isinstance(risk_score, bool) or not isinstance(risk_score, (int, float)):
        raise TypeError("risk_score must be a number between 0 and 1")
    if not math.isfinite(risk_score) or not 0 <= risk_score <= 1:
        raise ValueError("risk_score must be finite and between 0 and 1")
    if risk_score >= BLOCK_THRESHOLD:
        return "BLOCK"
    if risk_score >= REVIEW_THRESHOLD:
        return "REVIEW"
    return "APPROVE"
