from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def isolate_cached_data():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def test_dashboard_renders_without_exceptions():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Live traffic",
        "Analyst review",
        "Monitoring",
        "Model evaluation",
        "Dataset explorer",
    ]


def test_restarting_playback_preserves_history_and_uses_a_fresh_simulation(tmp_path, monkeypatch):
    import pandas as pd

    from src import config, database, stream_processor

    csv = tmp_path / "transactions.csv"
    pd.DataFrame([{
        "transaction_id": 42, "tx_datetime": "2024-01-01T12:00:00",
        "tx_fraud": 0, "fraud_scenario": "legitimate", "customer_id": 1, "terminal_id": 2, "tx_amount": 10, "device_id": "fixture",
    }]).to_csv(csv, index=False)
    monkeypatch.setattr(config, "TRANSACTIONS_CSV", csv)
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(stream_processor, "score_transaction", lambda *args: (0.1, "Stage 1", "fixture", 0.1))
    monkeypatch.setattr(stream_processor, "active_model_release", lambda: "fixture")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=20)
    assert not app.exception
    for _ in range(2):
        next(button for button in app.button if button.label == "Start live stream").click().run(timeout=20)
        assert not app.exception
        assert app.session_state.stream_error is None
        assert app.session_state.stream_feature_store.global_tx_count == 1
    rows = database.get_transactions()
    assert len(rows) == 2
    assert rows.transaction_id.tolist() == [42, 42]
    assert rows.simulation_id.nunique() == 2
