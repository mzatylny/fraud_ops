"""Batch and online behavioural feature engineering."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .config import FEATURES


def build_batch_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create leakage-controlled features for model training.

    Each customer's historical features are shifted so the current transaction is
    not used to describe itself. This is important for an academically credible
    fraud-detection evaluation.
    """
    work = df.copy()
    work["tx_datetime"] = pd.to_datetime(work["tx_datetime"])
    work = work.sort_values(["tx_datetime", "transaction_id"]).reset_index(drop=True)
    work["hour"] = work["tx_datetime"].dt.hour
    work["is_night"] = ((work["hour"] < 6) | (work["hour"] >= 23)).astype(int)

    # Customer amount baseline using only previous transactions.
    prior_sum = work.groupby("customer_id")["tx_amount"].cumsum() - work["tx_amount"]
    prior_count = work.groupby("customer_id").cumcount()
    global_mean = work["tx_amount"].mean()
    work["customer_avg_amount"] = np.where(prior_count > 0, prior_sum / prior_count, global_mean)
    work["amount_ratio_customer"] = work["tx_amount"] / (work["customer_avg_amount"] + 1.0)

    # Time since last transaction.
    work["time_since_last_tx_seconds"] = (
        work.groupby("customer_id")["tx_datetime"].diff().dt.total_seconds().fillna(86400)
    )

    # Rolling counts over one hour and 24 hours.
    # Implemented with deques to avoid pandas duplicate-timestamp index issues.
    from collections import defaultdict, deque

    customer_windows = defaultdict(deque)
    terminal_windows = defaultdict(deque)
    count_1h = []
    count_24h = []
    term_count_24h = []
    for row in work.itertuples(index=False):
        dt = row.tx_datetime.to_pydatetime() if hasattr(row.tx_datetime, "to_pydatetime") else row.tx_datetime
        cwin = customer_windows[row.customer_id]
        twin = terminal_windows[row.terminal_id]
        while cwin and (dt - cwin[0]).total_seconds() > 86400:
            cwin.popleft()
        while twin and (dt - twin[0]).total_seconds() > 86400:
            twin.popleft()
        count_24h.append(len(cwin))
        count_1h.append(sum(1 for t in cwin if (dt - t).total_seconds() <= 3600))
        term_count_24h.append(len(twin))
        cwin.append(dt)
        twin.append(dt)
    work["customer_tx_count_1h"] = count_1h
    work["customer_tx_count_24h"] = count_24h
    work["terminal_tx_count_24h"] = term_count_24h

    # First-seen flags. cumcount()==0 means current transaction is first occurrence.
    work = work.sort_values(["tx_datetime", "transaction_id"]).reset_index(drop=True)
    work["is_new_terminal_for_customer"] = (
        work.groupby(["customer_id", "terminal_id"]).cumcount() == 0
    ).astype(int)
    work["is_new_device"] = (work.groupby(["customer_id", "device_id"]).cumcount() == 0).astype(int)

    for col in ["terminal_risk_score", "distance_to_terminal", "is_foreign_country"]:
        if col not in work.columns:
            work[col] = 0

    X = work[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = work["tx_fraud"].astype(int)
    return X, y, work


@dataclass
class OnlineFeatureStore:
    """Small in-memory feature store used by the live stream simulator."""

    customer_history: dict[Any, deque] = field(default_factory=lambda: defaultdict(deque))
    terminal_history: dict[Any, deque] = field(default_factory=lambda: defaultdict(deque))
    customer_amount_sum: dict[Any, float] = field(default_factory=lambda: defaultdict(float))
    customer_tx_count: dict[Any, int] = field(default_factory=lambda: defaultdict(int))
    seen_customer_terminals: set[tuple[Any, Any]] = field(default_factory=set)
    seen_customer_devices: set[tuple[Any, Any]] = field(default_factory=set)

    def compute(self, tx: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
        dt = pd.to_datetime(tx["tx_datetime"]).to_pydatetime()
        customer_id = tx["customer_id"]
        terminal_id = tx["terminal_id"]
        device_id = tx.get("device_id", "unknown")
        amount = float(tx["tx_amount"])
        hour = dt.hour

        history = self.customer_history[customer_id]
        term_history = self.terminal_history[terminal_id]

        while history and (dt - history[0]).total_seconds() > 86400:
            history.popleft()
        while term_history and (dt - term_history[0]).total_seconds() > 86400:
            term_history.popleft()

        count_24h = len(history)
        count_1h = sum(1 for t in history if (dt - t).total_seconds() <= 3600)
        terminal_count_24h = len(term_history)

        tx_count = self.customer_tx_count[customer_id]
        avg_amount = self.customer_amount_sum[customer_id] / tx_count if tx_count > 0 else amount
        amount_ratio = amount / (avg_amount + 1.0)
        time_since = (dt - history[-1]).total_seconds() if history else 86400

        pair_terminal = (customer_id, terminal_id)
        pair_device = (customer_id, device_id)
        is_new_terminal = 0 if pair_terminal in self.seen_customer_terminals else 1
        is_new_device = 0 if pair_device in self.seen_customer_devices else 1

        raw = {
            "tx_amount": amount,
            "hour": hour,
            "is_night": int(hour < 6 or hour >= 23),
            "customer_avg_amount": float(avg_amount),
            "amount_ratio_customer": float(amount_ratio),
            "time_since_last_tx_seconds": float(time_since),
            "customer_tx_count_1h": float(count_1h),
            "customer_tx_count_24h": float(count_24h),
            "terminal_tx_count_24h": float(terminal_count_24h),
            "terminal_risk_score": float(tx.get("terminal_risk_score", 0.0)),
            "distance_to_terminal": float(tx.get("distance_to_terminal", 0.0)),
            "is_new_terminal_for_customer": int(is_new_terminal),
            "is_foreign_country": int(tx.get("is_foreign_country", 0)),
            "is_new_device": int(is_new_device),
        }

        # Update state after features are computed to avoid leakage.
        history.append(dt)
        term_history.append(dt)
        self.customer_amount_sum[customer_id] += amount
        self.customer_tx_count[customer_id] += 1
        self.seen_customer_terminals.add(pair_terminal)
        self.seen_customer_devices.add(pair_device)

        return pd.DataFrame([raw], columns=FEATURES), raw


FEATURE_STORE = OnlineFeatureStore()


def compute_online_features(tx: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    return FEATURE_STORE.compute(tx)


def reset_feature_store() -> None:
    global FEATURE_STORE
    FEATURE_STORE = OnlineFeatureStore()
