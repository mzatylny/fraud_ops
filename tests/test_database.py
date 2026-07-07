from pathlib import Path

import pandas as pd

from src import database


def test_database_insert_update(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db(reset=True)
    database.insert_transaction(
        {
            "tx_id": "abc",
            "transaction_id": 1,
            "tx_datetime": "2024-01-01 12:00:00",
            "customer_id": 1,
            "terminal_id": 2,
            "tx_amount": 12.5,
            "country": "US",
            "device_id": "D",
            "channel": "pos",
            "merchant_category": "grocery",
            "risk_score": 0.5,
            "action": "REVIEW",
            "layer": "Stage 2 (Advanced)",
            "latency_ms": 1.2,
            "reasons": "test reason",
            "tx_fraud": 1,
            "fraud_scenario": "unit_test",
        }
    )
    database.update_transaction_status("abc", "CONFIRMED_FRAUD", "checked")
    df = database.get_transactions()
    assert len(df) == 1
    assert df.iloc[0].status == "CONFIRMED_FRAUD"
