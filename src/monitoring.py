"""Deterministic data and prediction drift monitoring for fraud operations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import BLOCK_THRESHOLD, POLICY_VERSION, REVIEW_THRESHOLD

WARNING_PSI = 0.10
CRITICAL_PSI = 0.25


def action_for_score(score: float) -> str:
    if score >= BLOCK_THRESHOLD:
        return "BLOCK"
    if score >= REVIEW_THRESHOLD:
        return "REVIEW"
    return "APPROVE"


def _numeric_profile(values: np.ndarray, bins: int = 10) -> dict[str, list[float]]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        raise ValueError("monitoring baseline values must contain finite observations")
    edges = np.unique(np.quantile(clean, np.linspace(0, 1, bins + 1)))
    if len(edges) == 1:
        padding = max(abs(float(edges[0])) * 0.01, 1e-6)
        edges = np.array([edges[0] - padding, edges[0] + padding])
    else:
        padding = max(float(edges[-1] - edges[0]) * 0.01, 1e-6)
        edges[0] -= padding
        edges[-1] += padding
    clipped = np.clip(clean, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    proportions = counts / counts.sum()
    return {
        "edges": [float(value) for value in edges],
        "proportions": [float(value) for value in proportions],
    }


def create_monitoring_baseline(
    transactions: pd.DataFrame,
    risk_scores: np.ndarray,
    model_release: str,
) -> dict[str, Any]:
    if len(transactions) != len(risk_scores) or len(transactions) == 0:
        raise ValueError("transactions and risk_scores must be non-empty and aligned")
    actions = pd.Series([action_for_score(float(score)) for score in risk_scores])
    action_distribution = actions.value_counts(normalize=True).to_dict()
    return {
        "schema_version": 1,
        "reference_count": int(len(transactions)),
        "model_release": model_release,
        "policy_version": POLICY_VERSION,
        "numeric_profiles": {
            "tx_amount": _numeric_profile(transactions["tx_amount"].to_numpy()),
            "risk_score": _numeric_profile(np.asarray(risk_scores)),
        },
        "action_distribution": {
            action: float(action_distribution.get(action, 0.0))
            for action in ("APPROVE", "REVIEW", "BLOCK")
        },
    }


def population_stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.shape != actual.shape or expected.size == 0:
        raise ValueError("PSI distributions must have the same non-empty shape")
    epsilon = 1e-6
    expected = np.clip(expected, epsilon, None)
    actual = np.clip(actual, epsilon, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _numeric_psi(values: pd.Series, profile: dict[str, Any]) -> float:
    edges = np.asarray(profile["edges"], dtype=float)
    expected = np.asarray(profile["proportions"], dtype=float)
    clean = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("inf")
    counts, _ = np.histogram(np.clip(clean, edges[0], edges[-1]), bins=edges)
    return population_stability_index(expected, counts / counts.sum())


def _status_for_psi(value: float) -> str:
    if value >= CRITICAL_PSI:
        return "critical"
    if value >= WARNING_PSI:
        return "warning"
    return "healthy"


def evaluate_operations(
    current: pd.DataFrame,
    baseline: dict[str, Any],
    *,
    min_sample_size: int = 50,
    latency_slo_ms: float = 100.0,
) -> dict[str, Any]:
    if current.empty:
        return {
            "status": "insufficient_data",
            "sample_size": 0,
            "signals": [],
            "alerts": ["No scored transactions are available for monitoring."],
        }

    required = {"tx_amount", "risk_score", "latency_ms", "action"}
    missing = sorted(required.difference(current.columns))
    if missing:
        raise ValueError("monitoring data is missing columns: " + ", ".join(missing))

    invalid = (
        ~np.isfinite(pd.to_numeric(current["tx_amount"], errors="coerce"))
        | ~np.isfinite(pd.to_numeric(current["risk_score"], errors="coerce"))
        | ~np.isfinite(pd.to_numeric(current["latency_ms"], errors="coerce"))
        | (pd.to_numeric(current["tx_amount"], errors="coerce") < 0)
        | ~pd.to_numeric(current["risk_score"], errors="coerce").between(0, 1)
        | (pd.to_numeric(current["latency_ms"], errors="coerce") < 0)
        | ~current["action"].isin(["APPROVE", "REVIEW", "BLOCK"])
    )
    data_quality_error_rate = float(invalid.mean())
    signals = []
    for name, profile in baseline["numeric_profiles"].items():
        psi = _numeric_psi(current[name], profile)
        signals.append({"name": name, "psi": psi, "status": _status_for_psi(psi)})

    expected_actions = np.array(
        [baseline["action_distribution"].get(name, 0.0) for name in ("APPROVE", "REVIEW", "BLOCK")]
    )
    current_actions = current["action"].value_counts(normalize=True)
    actual_actions = np.array(
        [current_actions.get(name, 0.0) for name in ("APPROVE", "REVIEW", "BLOCK")]
    )
    action_psi = population_stability_index(expected_actions, actual_actions)
    signals.append(
        {"name": "action_distribution", "psi": action_psi, "status": _status_for_psi(action_psi)}
    )

    p95_latency_ms = float(pd.to_numeric(current["latency_ms"]).quantile(0.95))
    alerts = []
    if len(current) < min_sample_size:
        alerts.append(
            f"Only {len(current)} transactions are available; at least {min_sample_size} are required."
        )
    for signal in signals:
        if signal["status"] != "healthy":
            alerts.append(
                f"{signal['name']} drift is {signal['status']} (PSI {signal['psi']:.3f})."
            )
    if data_quality_error_rate > 0:
        alerts.append(f"Data-quality error rate is {data_quality_error_rate:.2%}.")
    if p95_latency_ms > latency_slo_ms:
        alerts.append(
            f"p95 scoring latency {p95_latency_ms:.2f} ms exceeds the {latency_slo_ms:.2f} ms SLO."
        )

    active_releases = (
        sorted(str(value) for value in current["model_release"].dropna().unique())
        if "model_release" in current
        else []
    )
    active_policies = (
        sorted(str(value) for value in current["policy_version"].dropna().unique())
        if "policy_version" in current
        else []
    )
    if len(active_releases) > 1:
        alerts.append("The monitoring window contains multiple model releases.")
    if len(active_policies) > 1:
        alerts.append("The monitoring window contains multiple policy versions.")

    if len(current) < min_sample_size:
        status = "insufficient_data"
    elif any(signal["status"] == "critical" for signal in signals) or data_quality_error_rate > 0:
        status = "critical"
    elif alerts:
        status = "warning"
    else:
        status = "healthy"
    return {
        "status": status,
        "sample_size": int(len(current)),
        "signals": signals,
        "data_quality_error_rate": data_quality_error_rate,
        "p95_latency_ms": p95_latency_ms,
        "latency_slo_ms": latency_slo_ms,
        "active_model_releases": active_releases,
        "active_policy_versions": active_policies,
        "alerts": alerts,
    }
