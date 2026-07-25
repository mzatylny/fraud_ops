import numpy as np
import pandas as pd
import pytest

from train_models import precision_at_k, temporal_split


def test_temporal_split_orders_and_aligns_inputs():
    enriched = pd.DataFrame(
        {
            "transaction_id": [3, 1, 2, 4, 8, 6, 5, 7],
            "tx_datetime": pd.to_datetime(
                [
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-04",
                    "2024-01-08",
                    "2024-01-06",
                    "2024-01-05",
                    "2024-01-07",
                ]
            ),
        }
    )
    X = pd.DataFrame({"row_id": enriched["transaction_id"]})
    y = pd.Series((enriched["transaction_id"] % 2).astype(int))
    X_train, X_test, y_train, y_test, test_rows = temporal_split(
        X, y, enriched, test_ratio=0.25
    )
    assert X_train["row_id"].tolist() == [1, 2, 3, 4, 5, 6]
    assert X_test["row_id"].tolist() == [7, 8]
    assert test_rows["transaction_id"].tolist() == [7, 8]
    assert y_test.tolist() == [1, 0]


def test_precision_at_k_validates_arguments():
    with pytest.raises(ValueError):
        precision_at_k(pd.Series(dtype=int), np.array([]))
    with pytest.raises(ValueError):
        precision_at_k(pd.Series([0, 1]), np.array([0.1, 0.9]), 0)
