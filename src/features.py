"""Batch and online behavioural feature engineering."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_CUSTOMER_AVG_AMOUNT, FEATURES


class OutOfOrderTransactionError(ValueError):
    """Raised when an online event is older than already committed state."""


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
    prior_global_count = np.arange(len(work))
    prior_global_sum = work["tx_amount"].cumsum() - work["tx_amount"]
    prior_global_mean = np.where(
        prior_global_count > 0,
        prior_global_sum / np.maximum(prior_global_count, 1),
        DEFAULT_CUSTOMER_AVG_AMOUNT,
    )
    work["customer_avg_amount"] = np.where(
        prior_count > 0,
        prior_sum / np.maximum(prior_count, 1),
        prior_global_mean,
    )
    work["amount_ratio_customer"] = work["tx_amount"] / (work["customer_avg_amount"] + 1.0)

    # Time since last transaction.
    work["time_since_last_tx_seconds"] = (
        work.groupby("customer_id")["tx_datetime"].diff().dt.total_seconds().fillna(86400)
    )

    # Rolling counts over one hour and 24 hours. Deques avoid duplicate-index issues.
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
    latest_customer_time: dict[Any, datetime] = field(default_factory=dict)
    latest_terminal_time: dict[Any, datetime] = field(default_factory=dict)
    global_amount_sum: float = 0.0
    global_tx_count: int = 0
    seen_customer_terminals: set[tuple[Any, Any]] = field(default_factory=set)
    seen_customer_devices: set[tuple[Any, Any]] = field(default_factory=set)

    def _normalise(self, tx: dict[str, Any]) -> tuple[datetime, Any, Any, Any, float]:
        dt = pd.to_datetime(tx["tx_datetime"]).to_pydatetime()
        if pd.isna(dt):
            raise ValueError("tx_datetime must be a valid datetime")
        customer_id = tx["customer_id"]
        terminal_id = tx["terminal_id"]
        device_id = tx.get("device_id", "unknown")
        amount = float(tx["tx_amount"])
        if not np.isfinite(amount) or amount < 0:
            raise ValueError("tx_amount must be a non-negative finite number")
        return dt, customer_id, terminal_id, device_id, amount

    def _validate_order(self, dt: datetime, customer_id: Any, terminal_id: Any) -> None:
        customer_latest = self.latest_customer_time.get(customer_id)
        terminal_latest = self.latest_terminal_time.get(terminal_id)
        if customer_latest is not None and dt < customer_latest:
            raise OutOfOrderTransactionError(
                f"customer {customer_id!r} event at {dt.isoformat()} precedes committed state"
            )
        if terminal_latest is not None and dt < terminal_latest:
            raise OutOfOrderTransactionError(
                f"terminal {terminal_id!r} event at {dt.isoformat()} precedes committed state"
            )

    def preview(self, tx: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
        """Compute features without mutating state.

        A caller can score and persist the event before calling :meth:`commit`, so
        failed scoring attempts do not contaminate future behavioural features.
        """
        dt, customer_id, terminal_id, device_id, amount = self._normalise(tx)
        self._validate_order(dt, customer_id, terminal_id)
        hour = dt.hour

        history = self.customer_history[customer_id]
        term_history = self.terminal_history[terminal_id]
        valid_history = [t for t in history if (dt - t).total_seconds() <= 86400]
        valid_term_history = [t for t in term_history if (dt - t).total_seconds() <= 86400]

        count_24h = len(valid_history)
        count_1h = sum(1 for t in valid_history if (dt - t).total_seconds() <= 3600)
        terminal_count_24h = len(valid_term_history)

        tx_count = self.customer_tx_count[customer_id]
        if tx_count > 0:
            avg_amount = self.customer_amount_sum[customer_id] / tx_count
        elif self.global_tx_count > 0:
            avg_amount = self.global_amount_sum / self.global_tx_count
        else:
            avg_amount = DEFAULT_CUSTOMER_AVG_AMOUNT
        amount_ratio = amount / (avg_amount + 1.0)
        previous_time = self.latest_customer_time.get(customer_id)
        time_since = (dt - previous_time).total_seconds() if previous_time else 86400

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

        return pd.DataFrame([raw], columns=FEATURES), raw

    def commit(self, tx: dict[str, Any]) -> None:
        """Commit an event after successful scoring and persistence."""
        dt, customer_id, terminal_id, device_id, amount = self._normalise(tx)
        self._validate_order(dt, customer_id, terminal_id)
        history = self.customer_history[customer_id]
        term_history = self.terminal_history[terminal_id]
        while history and (dt - history[0]).total_seconds() > 86400:
            history.popleft()
        while term_history and (dt - term_history[0]).total_seconds() > 86400:
            term_history.popleft()
        history.append(dt)
        term_history.append(dt)
        self.customer_amount_sum[customer_id] += amount
        self.customer_tx_count[customer_id] += 1
        self.global_amount_sum += amount
        self.global_tx_count += 1
        self.latest_customer_time[customer_id] = dt
        self.latest_terminal_time[terminal_id] = dt
        self.seen_customer_terminals.add((customer_id, terminal_id))
        self.seen_customer_devices.add((customer_id, device_id))

    def compute(self, tx: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
        features = self.preview(tx)
        self.commit(tx)
        return features


FEATURE_STORE = OnlineFeatureStore()


def compute_online_features(tx: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    return FEATURE_STORE.compute(tx)


def preview_online_features(tx: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    return FEATURE_STORE.preview(tx)


def commit_online_transaction(tx: dict[str, Any]) -> None:
    FEATURE_STORE.commit(tx)


def reset_feature_store() -> None:
    global FEATURE_STORE
    FEATURE_STORE = OnlineFeatureStore()
