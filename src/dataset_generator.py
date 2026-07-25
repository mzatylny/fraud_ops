"""Handbook-inspired synthetic transaction simulator.

The generator follows the Fraud Detection Handbook idea of customer profiles,
terminal profiles, terminal radius association, daily Poisson transaction counts,
and time-dependent fraud scenarios. It adds extra operational fields so the
project can support a dashboard and analyst workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import CUSTOMERS_CSV, DATA_DIR, TERMINALS_CSV, TRANSACTIONS_CSV


@dataclass(frozen=True)
class SimulationConfig:
    n_customers: int = 1800
    n_terminals: int = 3500
    n_days: int = 75
    radius: float = 5.0
    start_date: str = "2024-01-01"
    random_state: int = 42
    max_transactions: int | None = None
    target_max_fraud_rate: float = 0.01

    def __post_init__(self) -> None:
        integer_fields = {
            "n_customers": self.n_customers,
            "n_terminals": self.n_terminals,
            "n_days": self.n_days,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_days < 7:
            raise ValueError("n_days must be at least 7 to simulate temporal fraud scenarios")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be a positive finite number")
        if self.max_transactions is not None and (
            not isinstance(self.max_transactions, int)
            or isinstance(self.max_transactions, bool)
            or self.max_transactions <= 0
        ):
            raise ValueError("max_transactions must be a positive integer or None")
        if not 0 < self.target_max_fraud_rate < 1:
            raise ValueError("target_max_fraud_rate must be between 0 and 1")
        try:
            datetime.fromisoformat(self.start_date)
        except (TypeError, ValueError) as exc:
            raise ValueError("start_date must be an ISO-8601 date or datetime") from exc


COUNTRIES = ["US", "UK", "DE", "FR", "CH", "CA", "AU", "JP", "BR", "PL", "ES", "IT"]
COUNTRY_WEIGHTS = [0.34, 0.13, 0.10, 0.08, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03, 0.025, 0.025]
CATEGORIES = ["grocery", "fuel", "electronics", "travel", "restaurant", "digital", "luxury", "pharmacy"]
CHANNELS = ["pos", "ecommerce", "mobile_wallet"]


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_customer_profiles(n_customers: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for customer_id in range(n_customers):
        mean_amount = rng.uniform(5, 110)
        # A small number of high-spending customers makes amount-only detection harder.
        if rng.random() < 0.045:
            mean_amount = rng.uniform(180, 650)
        rows.append(
            {
                "customer_id": customer_id,
                "x_customer": rng.uniform(0, 100),
                "y_customer": rng.uniform(0, 100),
                "mean_amount": mean_amount,
                "std_amount": max(2.0, mean_amount / 2.0),
                "mean_nb_tx_per_day": rng.uniform(0.15, 4.3),
                "home_country": rng.choice(COUNTRIES, p=np.array(COUNTRY_WEIGHTS) / np.sum(COUNTRY_WEIGHTS)),
                "primary_device": f"DEV_C{customer_id}_0",
                "secondary_device": f"DEV_C{customer_id}_1" if rng.random() < 0.33 else "",
            }
        )
    return pd.DataFrame(rows)


def generate_terminal_profiles(n_terminals: int, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state + 17)
    categories = rng.choice(CATEGORIES, size=n_terminals, p=[0.26, 0.13, 0.12, 0.08, 0.15, 0.15, 0.04, 0.07])
    base_risk = []
    for cat in categories:
        risk = {
            "grocery": 0.02,
            "fuel": 0.04,
            "electronics": 0.09,
            "travel": 0.08,
            "restaurant": 0.03,
            "digital": 0.12,
            "luxury": 0.15,
            "pharmacy": 0.025,
        }[cat]
        base_risk.append(float(np.clip(rng.normal(risk, 0.015), 0.005, 0.3)))
    return pd.DataFrame(
        {
            "terminal_id": np.arange(n_terminals),
            "x_terminal": rng.uniform(0, 100, n_terminals),
            "y_terminal": rng.uniform(0, 100, n_terminals),
            "merchant_category": categories,
            "terminal_risk_score": base_risk,
            "country": rng.choice(COUNTRIES, size=n_terminals, p=np.array(COUNTRY_WEIGHTS) / np.sum(COUNTRY_WEIGHTS)),
        }
    )


def _terminal_candidates(customer: pd.Series, terminals: pd.DataFrame, radius: float) -> np.ndarray:
    dist = np.sqrt(
        (terminals["x_terminal"].to_numpy() - customer["x_customer"]) ** 2
        + (terminals["y_terminal"].to_numpy() - customer["y_customer"]) ** 2
    )
    candidates = terminals.loc[dist < radius, "terminal_id"].to_numpy()
    if len(candidates) == 0:
        # Fallback: nearest terminals. This avoids customers with no transactions.
        candidates = terminals.iloc[np.argsort(dist)[:20]]["terminal_id"].to_numpy()
    return candidates


def _normal_time_seconds(rng: np.random.Generator) -> int:
    # Similar idea to handbook: most transactions around daytime, but with noise.
    for _ in range(10):
        seconds = int(rng.normal(86400 / 2, 22000))
        if 0 <= seconds < 86400:
            return seconds
    return int(rng.uniform(0, 86400))


def generate_legitimate_transactions(
    customers: pd.DataFrame, terminals: pd.DataFrame, cfg: SimulationConfig
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state + 101)
    start = datetime.fromisoformat(cfg.start_date)
    terminal_lookup = terminals.set_index("terminal_id")
    rows = []
    transaction_id = 0

    candidate_cache = {
        int(row.customer_id): _terminal_candidates(row, terminals, cfg.radius)
        for _, row in customers.iterrows()
    }

    for _, customer in customers.iterrows():
        cid = int(customer.customer_id)
        candidates = candidate_cache[cid]
        for day in range(cfg.n_days):
            nb_tx = rng.poisson(customer.mean_nb_tx_per_day)
            if nb_tx <= 0:
                continue
            for _ in range(nb_tx):
                terminal_id = int(rng.choice(candidates))
                terminal = terminal_lookup.loc[terminal_id]
                dt = start + timedelta(days=day, seconds=_normal_time_seconds(rng))
                amount = rng.normal(customer.mean_amount, customer.std_amount)
                if amount <= 0:
                    amount = rng.uniform(1, customer.mean_amount * 2)
                amount = float(round(amount, 2))
                device = customer.primary_device
                if customer.secondary_device and rng.random() < 0.18:
                    device = customer.secondary_device
                rows.append(
                    {
                        "transaction_id": transaction_id,
                        "tx_datetime": dt,
                        "tx_time_seconds": int((dt - start).total_seconds()),
                        "tx_time_days": day,
                        "customer_id": cid,
                        "terminal_id": terminal_id,
                        "tx_amount": amount,
                        "country": terminal.country if rng.random() < 0.93 else rng.choice(COUNTRIES),
                        "device_id": device,
                        "channel": rng.choice(CHANNELS, p=[0.58, 0.32, 0.10]),
                        "merchant_category": terminal.merchant_category,
                        "terminal_risk_score": terminal.terminal_risk_score,
                        "home_country": customer.home_country,
                        "x_customer": customer.x_customer,
                        "y_customer": customer.y_customer,
                        "x_terminal": terminal.x_terminal,
                        "y_terminal": terminal.y_terminal,
                        "tx_fraud": 0,
                        "fraud_scenario": "legit",
                    }
                )
                transaction_id += 1
                if cfg.max_transactions and transaction_id >= cfg.max_transactions:
                    return pd.DataFrame(rows).sort_values("tx_datetime").reset_index(drop=True)

    return pd.DataFrame(rows).sort_values("tx_datetime").reset_index(drop=True)


def add_fraud_scenarios(
    df: pd.DataFrame,
    customers: pd.DataFrame,
    terminals: pd.DataFrame,
    cfg: SimulationConfig,
) -> pd.DataFrame:
    """Label time-dependent fraud scenarios.

    The first three scenarios mirror the handbook conceptually. Two extra subtle
    scenarios create richer streaming system behaviour.
    """
    rng = np.random.default_rng(cfg.random_state + 202)
    df = df.copy()

    # Scenario 1: obvious large amount validation pattern.
    # The threshold is adapted upward because this enhanced simulator contains legitimate high spenders.
    s1_mask = df["tx_amount"] > 950
    df.loc[s1_mask, ["tx_fraud", "fraud_scenario"]] = [1, "S1_large_amount"]

    # Scenario 2: compromised terminals over 28 days.
    terminal_ids = terminals["terminal_id"].to_numpy()
    compromised_per_event = max(1, int(len(terminal_ids) * 0.0006))
    for day in range(0, cfg.n_days, 5):
        compromised = rng.choice(terminal_ids, size=min(compromised_per_event, len(terminal_ids)), replace=False)
        mask = (
            (df["tx_time_days"] >= day)
            & (df["tx_time_days"] < day + 28)
            & (df["terminal_id"].isin(compromised))
        )
        df.loc[mask, ["tx_fraud", "fraud_scenario"]] = [1, "S2_compromised_terminal"]

    # Scenario 3: card-not-present/customer compromise over 14 days.
    customer_ids = customers["customer_id"].to_numpy()
    compromised_customers_per_event = max(1, int(len(customer_ids) * 0.0015))
    for day in range(0, cfg.n_days, 4):
        compromised_customers = rng.choice(customer_ids, size=min(compromised_customers_per_event, len(customer_ids)), replace=False)
        idx = df.index[
            (df["tx_time_days"] >= day)
            & (df["tx_time_days"] < day + 14)
            & (df["customer_id"].isin(compromised_customers))
        ].tolist()
        if idx:
            chosen = rng.choice(idx, size=max(1, len(idx) // 3), replace=False)
            df.loc[chosen, "tx_amount"] = (df.loc[chosen, "tx_amount"] * rng.uniform(3.5, 6.0)).round(2)
            df.loc[chosen, ["tx_fraud", "fraud_scenario"]] = [1, "S3_customer_compromise"]
            df.loc[chosen, "channel"] = "ecommerce"
            df.loc[chosen, "device_id"] = [f"DEV_ATO_{int(x)}" for x in rng.integers(10000, 99999, len(chosen))]

    # Scenario 4: card testing. Inject new tiny rapid transactions.
    extra_rows = []
    next_id = int(df["transaction_id"].max()) + 1 if not df.empty else 0
    start = datetime.fromisoformat(cfg.start_date)
    n_card_testing_customers = min(len(customer_ids), max(1, int(len(df) * 0.0006)))
    sample_customers = rng.choice(customer_ids, size=n_card_testing_customers, replace=False)
    high_risk_terminals = terminals.sort_values("terminal_risk_score", ascending=False).head(250)
    for cid in sample_customers:
        burst_day = int(rng.integers(1, cfg.n_days - 1))
        base_dt = start + timedelta(days=burst_day, seconds=int(rng.integers(0, 86400)))
        for j in range(int(rng.integers(3, 8))):
            terminal = high_risk_terminals.iloc[int(rng.integers(0, len(high_risk_terminals)))]
            dt = base_dt + timedelta(seconds=45 * j)
            extra_rows.append(
                {
                    "transaction_id": next_id,
                    "tx_datetime": dt,
                    "tx_time_seconds": int((dt - start).total_seconds()),
                    "tx_time_days": burst_day,
                    "customer_id": int(cid),
                    "terminal_id": int(terminal.terminal_id),
                    "tx_amount": float(round(rng.uniform(0.5, 4.5), 2)),
                    "country": str(rng.choice(["US", "BR", "JP", "PL"])),
                    "device_id": f"DEV_TEST_{int(rng.integers(1, 9999))}",
                    "channel": "ecommerce",
                    "merchant_category": "digital",
                    "terminal_risk_score": float(terminal.terminal_risk_score),
                    "home_country": customers.loc[customers.customer_id == cid, "home_country"].iloc[0],
                    "x_customer": customers.loc[customers.customer_id == cid, "x_customer"].iloc[0],
                    "y_customer": customers.loc[customers.customer_id == cid, "y_customer"].iloc[0],
                    "x_terminal": float(terminal.x_terminal),
                    "y_terminal": float(terminal.y_terminal),
                    "tx_fraud": 1,
                    "fraud_scenario": "S4_card_testing",
                }
            )
            next_id += 1

    # Scenario 5: stealthy account takeover, not always high amount.
    n_stealth_customers = min(len(customer_ids), max(1, int(len(customer_ids) * 0.006)))
    for cid in rng.choice(customer_ids, size=n_stealth_customers, replace=False):
        customer_mask = df["customer_id"] == cid
        idx = df.index[customer_mask & (df["tx_time_days"] > cfg.n_days // 3)].tolist()
        if not idx:
            continue
        chosen = rng.choice(idx, size=min(len(idx), int(rng.integers(1, 4))), replace=False)
        df.loc[chosen, "country"] = rng.choice(["RU", "CN", "NG", "BR"], size=len(chosen))
        df.loc[chosen, "device_id"] = [f"DEV_NEW_{int(x)}" for x in rng.integers(1000, 9999, len(chosen))]
        df.loc[chosen, "channel"] = "ecommerce"
        df.loc[chosen, "tx_amount"] = (df.loc[chosen, "tx_amount"] * rng.uniform(1.2, 2.2)).round(2)
        df.loc[chosen, ["tx_fraud", "fraud_scenario"]] = [1, "S5_stealth_ato"]

    if extra_rows:
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    # Calibrate to a strongly imbalanced fraud rate, as in realistic card-fraud work.
    fraud_idx = df.index[df["tx_fraud"] == 1].to_numpy()
    max_frauds = max(1, int(len(df) * cfg.target_max_fraud_rate))
    if len(fraud_idx) > max_frauds:
        keep = set(rng.choice(fraud_idx, size=max_frauds, replace=False).tolist())
        drop = [idx for idx in fraud_idx if idx not in keep]
        df.loc[drop, "tx_fraud"] = 0
        df.loc[drop, "fraud_scenario"] = "legit_after_calibration"

    df["distance_to_terminal"] = np.sqrt(
        (df["x_customer"] - df["x_terminal"]) ** 2 + (df["y_customer"] - df["y_terminal"]) ** 2
    )
    df["is_foreign_country"] = (df["country"] != df["home_country"]).astype(int)
    df = df.sort_values("tx_datetime").reset_index(drop=True)
    return df


def build_dataset(cfg: SimulationConfig = SimulationConfig()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()
    customers = generate_customer_profiles(cfg.n_customers, cfg.random_state)
    terminals = generate_terminal_profiles(cfg.n_terminals, cfg.random_state)
    transactions = generate_legitimate_transactions(customers, terminals, cfg)
    if transactions.empty:
        raise RuntimeError(
            "Simulation produced no legitimate transactions; increase customers, days, or transaction frequency"
        )
    transactions = add_fraud_scenarios(transactions, customers, terminals, cfg)
    customers.to_csv(CUSTOMERS_CSV, index=False)
    terminals.to_csv(TERMINALS_CSV, index=False)
    transactions.to_csv(TRANSACTIONS_CSV, index=False)
    return transactions, customers, terminals


if __name__ == "__main__":
    tx, cust, term = build_dataset()
    print(f"Generated {len(tx):,} transactions")
    print(f"Fraud rate: {tx.tx_fraud.mean() * 100:.3f}%")
    print(tx["fraud_scenario"].value_counts().head(10))
    print(f"Saved to {TRANSACTIONS_CSV}")
