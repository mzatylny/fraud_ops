import numpy as np
import pandas as pd

from src.monitoring import create_monitoring_baseline, evaluate_operations


def reference_frame(size=200):
    return pd.DataFrame(
        {
            "tx_amount": np.linspace(5, 250, size),
            "risk_score": np.linspace(0.01, 0.95, size),
            "latency_ms": np.linspace(1, 10, size),
            "action": ["APPROVE"] * 70 + ["REVIEW"] * 30 + ["BLOCK"] * 100,
            "model_release": ["release-1"] * size,
            "policy_version": ["policy-1"] * size,
        }
    )


def test_monitoring_reports_healthy_reference_like_traffic():
    frame = reference_frame()
    baseline = create_monitoring_baseline(frame, frame["risk_score"].to_numpy(), "release-1")
    frame["action"] = frame["risk_score"].map(
        lambda score: "BLOCK" if score >= 0.85 else "REVIEW" if score >= 0.35 else "APPROVE"
    )
    report = evaluate_operations(frame, baseline)
    assert report["status"] == "healthy"
    assert report["data_quality_error_rate"] == 0
    assert report["active_model_releases"] == ["release-1"]


def test_monitoring_alerts_on_drift_and_latency():
    reference = reference_frame()
    baseline = create_monitoring_baseline(
        reference, reference["risk_score"].to_numpy(), "release-1"
    )
    current = pd.DataFrame(
        {
            "tx_amount": [10_000.0] * 100,
            "risk_score": [0.99] * 100,
            "latency_ms": [250.0] * 100,
            "action": ["BLOCK"] * 100,
            "model_release": ["release-2"] * 100,
            "policy_version": ["policy-1"] * 100,
        }
    )
    report = evaluate_operations(current, baseline)
    assert report["status"] == "critical"
    assert any("drift" in alert for alert in report["alerts"])
    assert any("latency" in alert for alert in report["alerts"])


def test_monitoring_requires_enough_traffic():
    reference = reference_frame()
    baseline = create_monitoring_baseline(
        reference, reference["risk_score"].to_numpy(), "release-1"
    )
    report = evaluate_operations(reference.head(10), baseline)
    assert report["status"] == "insufficient_data"
    assert any("at least 50" in alert for alert in report["alerts"])
