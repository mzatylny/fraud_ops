"""Two-stage scoring service with latency measurement and explanations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, STAGE1_THRESHOLD

_STAGE1 = None
_STAGE2 = None


def load_models():
    global _STAGE1, _STAGE2
    if _STAGE1 is None:
        _STAGE1 = joblib.load(MODELS_DIR / "stage1_model.joblib")
    if _STAGE2 is None:
        _STAGE2 = joblib.load(MODELS_DIR / "stage2_model.joblib")
    return _STAGE1, _STAGE2


def explain(features: dict[str, float], score: float, layer: str) -> str:
    reasons = []
    if features["amount_ratio_customer"] >= 4.0:
        reasons.append("amount much higher than customer history")
    if features["customer_tx_count_1h"] >= 3:
        reasons.append("rapid customer velocity within one hour")
    if features["terminal_tx_count_24h"] >= 25:
        reasons.append("unusually busy terminal in last 24h")
    if features["is_new_terminal_for_customer"] == 1:
        reasons.append("first time at this terminal")
    if features["is_new_device"] == 1:
        reasons.append("new customer device")
    if features["is_foreign_country"] == 1:
        reasons.append("transaction country differs from customer home country")
    if features["terminal_risk_score"] >= 0.12:
        reasons.append("high-risk merchant category / terminal")
    if features["time_since_last_tx_seconds"] < 90:
        reasons.append("very short time since previous transaction")
    if score >= 0.85:
        reasons.append("advanced ensemble returned severe risk")
    if not reasons:
        reasons.append("normal behavioural profile")
    return f"{layer}: " + "; ".join(reasons)


def score_transaction(features_df: pd.DataFrame, raw_features: dict[str, float]) -> tuple[float, str, str, float]:
    stage1, stage2 = load_models()
    start = time.perf_counter()
    p1 = float(stage1.predict_proba(features_df)[0, 1])
    if p1 >= STAGE1_THRESHOLD:
        p2 = float(stage2.predict_proba(features_df)[0, 1])
        layer = "Stage 2 (Advanced)"
        score = p2
    else:
        layer = "Stage 1 (Fast Screen)"
        score = p1
    latency = (time.perf_counter() - start) * 1000
    return score, layer, explain(raw_features, score, layer), latency
