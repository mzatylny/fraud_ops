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
        {"tx_datetime": "bad-date"},
        {"customer_id": True},
    ],
)
def test_validate_transaction_rejects_invalid_events(change):
    with pytest.raises(ValueError):
        stream_processor.validate_transaction({**valid_transaction(), **change})


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
    monkeypatch.setattr(stream_processor, "insert_transaction", lambda record: (_ for _ in ()).throw(RuntimeError("db")))
    monkeypatch.setattr(stream_processor, "commit_online_transaction", committed.append)

    with pytest.raises(RuntimeError, match="db"):
        stream_processor.process_transaction(valid_transaction())
    assert committed == []
