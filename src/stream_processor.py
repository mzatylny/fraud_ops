"""End-to-end transaction processing pipeline."""

from __future__ import annotations

import uuid
from typing import Any

from .database import insert_transaction
from .decision_engine import make_decision
from .detector import score_transaction
from .features import compute_online_features


def process_transaction(tx: dict[str, Any]) -> dict[str, Any]:
    features_df, raw_features = compute_online_features(tx)
    risk_score, layer, reasons, latency_ms = score_transaction(features_df, raw_features)
    action = make_decision(risk_score)
    record = dict(tx)
    record.update(
        {
            "tx_id": str(uuid.uuid4())[:10],
            "risk_score": risk_score,
            "layer": layer,
            "reasons": reasons,
            "latency_ms": latency_ms,
            "action": action,
        }
    )
    insert_transaction(record)
    return record
