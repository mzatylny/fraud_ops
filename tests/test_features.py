from src.features import OnlineFeatureStore


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
