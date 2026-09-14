import sqlite3

import pytest

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
    history = database.get_review_history("abc")
    assert history.iloc[0].new_status == "CONFIRMED_FRAUD"


def _record(tx_id="abc", action="REVIEW"):
    return {
        "tx_id": tx_id,
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
        "action": action,
        "layer": "Stage 2 (Advanced)",
        "latency_ms": 1.2,
        "reasons": "test reason",
        "tx_fraud": 1,
        "fraud_scenario": "unit_test",
    }


def test_database_rejects_duplicates_and_invalid_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db(reset=True)
    database.insert_transaction(_record())
    with pytest.raises(sqlite3.IntegrityError):
        database.insert_transaction({**_record(), "transaction_id": None})
    with pytest.raises(database.DuplicateTransactionError, match="already processed"):
        database.insert_transaction(_record(tx_id="different"))
    with pytest.raises(KeyError):
        database.update_transaction_status("missing", "UNDER_REVIEW")
    with pytest.raises(ValueError):
        database.update_transaction_status("abc", "NOT_A_STATUS")


def test_auto_resolved_transaction_cannot_enter_review(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db(reset=True)
    database.insert_transaction(_record(action="APPROVE"))
    with pytest.raises(ValueError, match="routed to REVIEW"):
        database.update_transaction_status("abc", "UNDER_REVIEW")


@pytest.mark.parametrize(
    "change",
    [
        {"tx_id": ""},
        {"action": "ALLOW"},
        {"risk_score": float("nan")},
        {"risk_score": 1.1},
        {"latency_ms": -1},
        {"tx_fraud": 2},
    ],
)
def test_database_rejects_invalid_operational_records(tmp_path, monkeypatch, change):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db(reset=True)
    with pytest.raises(ValueError):
        database.insert_transaction({**_record(), **change})


def test_database_persists_model_and_policy_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db(reset=True)
    database.insert_transaction(
        {**_record(), "model_release": "release-123", "policy_version": "policy-7"}
    )
    row = database.get_transactions().iloc[0]
    assert row.model_release == "release-123"
    assert row.policy_version == "policy-7"
    assert database.transaction_exists(1)
    assert not database.transaction_exists(999)


def test_same_source_event_can_run_in_independent_simulations(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db()
    database.insert_transaction({**_record("first"), "simulation_id": "run-1"})
    database.insert_transaction({**_record("second"), "simulation_id": "run-2"})
    with pytest.raises(database.DuplicateTransactionError):
        database.insert_transaction({**_record("third"), "simulation_id": "run-1"})
    assert database.transaction_exists(1, "run-1")
    assert not database.transaction_exists(1)
    assert len(database.get_transactions()) == 2


def test_legacy_events_keep_duplicate_protection_after_migration(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE transactions (tx_id TEXT PRIMARY KEY, transaction_id INTEGER, "
            "tx_datetime TEXT, action TEXT, status TEXT, risk_score REAL);"
            "CREATE UNIQUE INDEX idx_transactions_source_event ON transactions(transaction_id);"
            "INSERT INTO transactions(tx_id, transaction_id) VALUES ('old', 42);"
        )
    database.init_db()
    database.init_db()
    assert database.transaction_exists(42)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT simulation_id FROM transactions").fetchone()[0] == "default"
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO transactions(tx_id, transaction_id) VALUES ('retry', 42)")


@pytest.mark.parametrize("simulation_id", [None, "", " " , 42, "x" * 129])
def test_invalid_simulation_names_are_rejected(tmp_path, monkeypatch, simulation_id):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db()
    with pytest.raises(ValueError, match="simulation_id"):
        database.insert_transaction({**_record(), "simulation_id": simulation_id})
