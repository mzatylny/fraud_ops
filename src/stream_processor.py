"""End-to-end transaction processing pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime
from math import isfinite
from numbers import Integral, Real
from typing import Any

import pandas as pd

from .database import insert_transaction
from .decision_engine import make_decision
from .detector import score_transaction
from .features import commit_online_transaction, preview_online_features

REQUIRED_FIELDS = {"tx_datetime", "customer_id", "terminal_id", "tx_amount"}


def _normalise_integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, str):
        candidate = value.strip()
        digits = candidate[1:] if candidate[:1] in {"+", "-"} else candidate
        if not digits.isdigit():
            raise ValueError(f"{field} must be an integer")
        result = int(candidate)
    elif isinstance(value, Real):
        numeric_value = float(value)
        if not isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"{field} must be an integer")
        result = int(numeric_value)
    else:
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _normalise_float(
    value: Any,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return result


def validate_transaction(tx: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise the minimum event contract."""
    if not isinstance(tx, dict):
        raise TypeError("transaction must be a dictionary")
    missing = sorted(REQUIRED_FIELDS.difference(tx))
    if missing:
        raise ValueError("missing required transaction fields: " + ", ".join(missing))

    normalised = dict(tx)
    raw_timestamp = normalised["tx_datetime"]
    if isinstance(raw_timestamp, bool) or not isinstance(raw_timestamp, (str, datetime)):
        raise ValueError("tx_datetime must be a valid datetime")
    try:
        timestamp = pd.Timestamp(pd.to_datetime(raw_timestamp, errors="raise"))
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("tx_datetime must be a valid datetime") from exc
    if pd.isna(timestamp):
        raise ValueError("tx_datetime must be a valid datetime")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    normalised["tx_datetime"] = timestamp.isoformat()

    for field in ("customer_id", "terminal_id"):
        normalised[field] = _normalise_integer(normalised[field], field, minimum=0)

    normalised["tx_amount"] = _normalise_float(
        normalised["tx_amount"], "tx_amount", minimum=0
    )
    if "transaction_id" in normalised:
        normalised["transaction_id"] = _normalise_integer(
            normalised["transaction_id"], "transaction_id", minimum=0
        )
    if "terminal_risk_score" in normalised:
        normalised["terminal_risk_score"] = _normalise_float(
            normalised["terminal_risk_score"],
            "terminal_risk_score",
            minimum=0,
            maximum=1,
        )
    if "distance_to_terminal" in normalised:
        normalised["distance_to_terminal"] = _normalise_float(
            normalised["distance_to_terminal"], "distance_to_terminal", minimum=0
        )
    for field in ("is_foreign_country", "tx_fraud"):
        if field in normalised:
            value = _normalise_integer(normalised[field], field)
            if value not in {0, 1}:
                raise ValueError(f"{field} must be 0 or 1")
            normalised[field] = value
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
