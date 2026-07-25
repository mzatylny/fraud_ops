import pandas as pd
import pytest

from src.config import FEATURES
from src.features import OnlineFeatureStore, OutOfOrderTransactionError, build_batch_features


def base_tx(amount=10, minute=0, terminal=1, device="D1", country="US"):
    return {
        "tx_datetime": f"2024-01-01 12:{minute:02d}:00",
        "customer_id": 1,
        "terminal_id": terminal,
        "tx_amount": amount,
        "terminal_risk_score": 0.1,
        "distance_to_terminal": 2.0,
        "is_foreign_country": 0 if country == "US" else 1,
        "device_id": device,
    }


def test_online_features_are_stateful():
    store = OnlineFeatureStore()
    _, raw1 = store.compute(base_tx(amount=10, minute=0))
    _, raw2 = store.compute(base_tx(amount=50, minute=10))
    assert raw1["is_new_device"] == 1
    assert raw2["is_new_device"] == 0
    assert raw2["customer_tx_count_1h"] == 1
    assert raw2["amount_ratio_customer"] > 4


def test_preview_does_not_mutate_state():
    store = OnlineFeatureStore()
    tx = base_tx()
    _, first = store.preview(tx)
    _, repeated = store.preview(tx)
    assert first == repeated
    assert store.global_tx_count == 0
    store.commit(tx)
    assert store.global_tx_count == 1


def test_new_customer_uses_prior_global_average():
    store = OnlineFeatureStore()
    store.compute({**base_tx(amount=100), "customer_id": 1})
    _, raw = store.compute({**base_tx(amount=20, minute=5), "customer_id": 2})
    assert raw["customer_avg_amount"] == pytest.approx(100.0)


def test_out_of_order_event_is_rejected():
    store = OnlineFeatureStore()
    store.compute(base_tx(minute=10))
    with pytest.raises(OutOfOrderTransactionError):
        store.compute(base_tx(minute=5))


def test_batch_and_online_features_match_for_ordered_stream():
    rows = [
        {**base_tx(amount=10, minute=0, terminal=1, device="D1"), "transaction_id": 1, "tx_fraud": 0},
        {**base_tx(amount=50, minute=10, terminal=1, device="D1"), "transaction_id": 2, "tx_fraud": 1},
        {
            **base_tx(amount=20, minute=20, terminal=2, device="D2"),
            "customer_id": 2,
            "transaction_id": 3,
            "tx_fraud": 0,
        },
    ]
    frame = pd.DataFrame(rows)
    batch, _, _ = build_batch_features(frame)
    store = OnlineFeatureStore()
    online = pd.concat([store.compute(row)[0] for row in rows], ignore_index=True)
    pd.testing.assert_frame_equal(
        batch[FEATURES].astype(float),
        online[FEATURES].astype(float),
        check_dtype=False,
    )
