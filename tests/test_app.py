from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import config, database, features, stream_processor

APP_PATH = Path(__file__).parents[1] / "app.py"


@pytest.fixture
def isolated_app(tmp_path, monkeypatch):
    """Run the dashboard against disposable data and database paths."""
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "fraud_ops.db")
    monkeypatch.setattr(config, "TRANSACTIONS_CSV", tmp_path / "transactions.csv")
    monkeypatch.setattr(config, "METRICS_CSV", tmp_path / "metrics.csv")
    monkeypatch.setattr(config, "THRESHOLD_CSV", tmp_path / "thresholds.csv")
    monkeypatch.setattr(config, "MONITORING_BASELINE_JSON", tmp_path / "baseline.json")
    st.cache_data.clear()

    def run_app():
        return AppTest.from_file(str(APP_PATH)).run(timeout=20)

    return run_app


def button_with_label(app, label):
    return next(button for button in app.button if button.label == label)


def test_dashboard_renders_without_exceptions(isolated_app):
    app = isolated_app()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Live traffic",
        "Analyst review",
        "Monitoring",
        "Model evaluation",
        "Dataset explorer",
    ]
    assert "No processed transactions yet. Start the live stream." in [
        message.value for message in app.info
    ]


def test_start_stream_reports_missing_dataset(isolated_app):
    app = isolated_app()
    button_with_label(app, "Start live stream").click().run(timeout=20)

    assert not app.exception
    assert [message.value for message in app.error] == [
        "Dataset missing. Run `python train_models.py` first."
    ]


def test_reset_control_clears_stream_state(isolated_app, monkeypatch):
    resets = []
    monkeypatch.setattr(features, "reset_feature_store", lambda: resets.append(True))

    app = isolated_app()
    button_with_label(app, "Reset online feature store").click().run(timeout=20)

    assert not app.exception
    assert resets == [True]
    assert [message.value for message in app.success][-1] == "Online feature store reset."


def test_start_stream_processes_a_transaction(isolated_app, monkeypatch, tmp_path):
    pd.DataFrame(
        [
            {
                "transaction_id": 101,
                "tx_datetime": "2026-08-29 09:30:00",
                "customer_id": 7,
                "terminal_id": 3,
                "tx_fraud": 1,
                "fraud_scenario": "test_case",
            }
        ]
    ).to_csv(tmp_path / "transactions.csv", index=False)
    monkeypatch.setattr(features, "reset_feature_store", lambda: None)
    monkeypatch.setattr(
        stream_processor,
        "process_transaction",
        lambda _record: {
            "action": "REVIEW",
            "risk_score": 0.73,
            "layer": "Stage 2 (Advanced)",
            "latency_ms": 1.25,
            "reasons": "Integration-test decision",
        },
    )

    app = isolated_app()
    button_with_label(app, "Start live stream").click().run(timeout=20)

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Decision"] == "REVIEW"
    assert metrics["Risk score"] == "0.730"
    assert "Integration-test decision" in [caption.value for caption in app.caption]


def test_monitoring_dashboard_renders_an_operational_snapshot(isolated_app, monkeypatch):
    recent = pd.DataFrame(
        [
            {
                "tx_datetime": "2026-08-29 09:30:00",
                "tx_id": "tx-1",
                "customer_id": 7,
                "terminal_id": 3,
                "tx_amount": 125.0,
                "risk_score": 0.73,
                "action": "REVIEW",
                "layer": "Stage 2 (Advanced)",
                "model_release": "advanced-test",
                "policy_version": "policy-test",
                "latency_ms": 1.25,
                "status": "PENDING",
            }
        ]
    )
    monkeypatch.setattr(database, "get_transactions", lambda _limit: recent)
    monkeypatch.setattr(
        database,
        "analytics_snapshot",
        lambda _limit: {
            "total": 1,
            "review_rate": 1.0,
            "block_rate": 0.0,
            "stage2_rate": 1.0,
            "avg_latency_ms": 1.25,
        },
    )

    app = isolated_app()

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Processed"] == "1"
    assert metrics["Review rate"] == "100.00%"
    assert metrics["Average latency"] == "1.25 ms"


def test_review_action_updates_case_status(isolated_app, monkeypatch):
    cases = pd.DataFrame(
        [
            {
                "tx_id": "case-1",
                "tx_datetime": "2026-08-29 09:30:00",
                "customer_id": 7,
                "terminal_id": 3,
                "tx_amount": 125.0,
                "country": "GB",
                "device_id": "device-1",
                "channel": "web",
                "merchant_category": "travel",
                "fraud_scenario": "test_case",
                "status": "PENDING",
                "risk_score": 0.73,
                "layer": "Stage 2 (Advanced)",
                "ground_truth": 1,
                "reasons": "Integration-test review case",
                "analyst_notes": "",
            }
        ]
    )
    updates = []
    monkeypatch.setattr(database, "get_review_cases", lambda *_args, **_kwargs: cases)
    monkeypatch.setattr(database, "get_review_history", lambda _tx_id: pd.DataFrame())
    monkeypatch.setattr(
        database,
        "update_transaction_status",
        lambda tx_id, status, notes: updates.append((tx_id, status, notes)),
    )

    app = isolated_app()
    button_with_label(app, "Start investigation").click().run(timeout=20)

    assert not app.exception
    assert updates == [("case-1", "UNDER_REVIEW", "Investigation started")]
