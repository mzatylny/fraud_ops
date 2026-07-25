"""End-to-end transaction processing pipeline."""

from __future__ import annotations

import uuid
from math import isfinite
from typing import Any

import pandas as pd

from .database import insert_transaction
from .decision_engine import make_decision
from .detector import score_transaction
from .features import commit_online_transaction, preview_online_features

REQUIRED_FIELDS = {"tx_datetime", "customer_id", "terminal_id", "tx_amount"}


def validate_transaction(tx: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise the minimum event contract."""
    if not isinstance(tx, dict):
        raise TypeError("transaction must be a dictionary")
    missing = sorted(REQUIRED_FIELDS.difference(tx))
    if missing:
        raise ValueError("missing required transaction fields: " + ", ".join(missing))

    normalised = dict(tx)
    try:
        timestamp = pd.to_datetime(normalised["tx_datetime"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("tx_datetime must be a valid datetime") from exc
    if pd.isna(timestamp):
        raise ValueError("tx_datetime must be a valid datetime")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    normalised["tx_datetime"] = timestamp.isoformat()

    for field in ("customer_id", "terminal_id"):
        value = normalised[field]
        if isinstance(value, bool):
            raise ValueError(f"{field} must be an integer")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer") from exc
        if not isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{field} must be an integer")
        normalised[field] = int(numeric_value)

    try:
        amount = float(normalised["tx_amount"])
    except (TypeError, ValueError) as exc:
        raise ValueError("tx_amount must be a non-negative finite number") from exc
    if not isfinite(amount) or amount < 0:
        raise ValueError("tx_amount must be a non-negative finite number")
    normalised["tx_amount"] = amount
    return normalised


def process_transaction(tx: dict[str, Any]) -> dict[str, Any]:
    tx = validate_transaction(tx)
    features_df, raw_features = preview_online_features(tx)
    risk_score, layer, reasons, latency_ms = score_transaction(features_df, raw_features)
    action = make_decision(risk_score)
    record = dict(tx)
    record.update(
        {
            "tx_id": uuid.uuid4().hex,
            "risk_score": risk_score,
            "layer": layer,
            "reasons": reasons,
            "latency_ms": latency_ms,
            "action": action,
        }
    )
    insert_transaction(record)
    commit_online_transaction(tx)
    return record
