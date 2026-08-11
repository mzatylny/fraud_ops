import pytest

from src import stream_processor


def valid_transaction():
    return {
        "tx_datetime": "2024-01-01T12:00:00",
        "customer_id": 1,
        "terminal_id": 2,
        "tx_amount": 15.0,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"tx_amount": -1},
        {"tx_amount": float("nan")},
        {"tx_amount": True},
        {"tx_datetime": "bad-date"},
        {"tx_datetime": 1},
        {"customer_id": True},
        {"customer_id": -1},
        {"transaction_id": -1},
        {"terminal_risk_score": float("nan")},
        {"terminal_risk_score": 1.01},
        {"distance_to_terminal": -0.01},
        {"is_foreign_country": 2},
    ],
)
def test_validate_transaction_rejects_invalid_events(change):
    with pytest.raises(ValueError):
        stream_processor.validate_transaction({**valid_transaction(), **change})


def test_validate_transaction_normalises_types_and_timezone():
    result = stream_processor.validate_transaction(
        {
            **valid_transaction(),
            "tx_datetime": "2024-01-01T13:00:00+01:00",
            "customer_id": "1",
            "terminal_id": 2.0,
            "tx_amount": "15.5",
            "transaction_id": "42",
            "terminal_risk_score": "0.2",
            "distance_to_terminal": "3.5",
            "is_foreign_country": "1",
        }
    )
    assert result["tx_datetime"] == "2024-01-01T12:00:00"
    assert result["customer_id"] == 1
    assert result["terminal_id"] == 2
    assert result["tx_amount"] == 15.5
    assert result["transaction_id"] == 42
    assert result["terminal_risk_score"] == 0.2
    assert result["distance_to_terminal"] == 3.5
    assert result["is_foreign_country"] == 1


def test_failed_persistence_does_not_commit_feature_state(monkeypatch):
    committed = []
    monkeypatch.setattr(
        stream_processor,
        "preview_online_features",
        lambda tx: (object(), {"amount_ratio_customer": 1.0}),
    )
    monkeypatch.setattr(
        stream_processor,
        "score_transaction",
        lambda frame, raw: (0.2, "Stage 1 (Fast Screen)", "normal", 0.1),
    )
    monkeypatch.setattr(stream_processor, "active_model_release", lambda: "test-release")
    monkeypatch.setattr(stream_processor, "insert_transaction", lambda record: (_ for _ in ()).throw(RuntimeError("db")))
    monkeypatch.setattr(stream_processor, "commit_online_transaction", committed.append)

    with pytest.raises(RuntimeError, match="db"):
        stream_processor.process_transaction(valid_transaction())
    assert committed == []


def test_duplicate_source_event_is_rejected_before_feature_preview(monkeypatch):
    transaction = {**valid_transaction(), "transaction_id": 42}
    monkeypatch.setattr(stream_processor, "transaction_exists", lambda transaction_id: True)
    monkeypatch.setattr(
        stream_processor,
        "preview_online_features",
        lambda tx: pytest.fail("duplicate event reached feature computation"),
    )
    with pytest.raises(stream_processor.DuplicateTransactionError, match="already processed"):
        stream_processor.process_transaction(transaction)
