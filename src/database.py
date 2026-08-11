"""SQLite persistence for transactions, review cases, and feedback."""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .config import DB_PATH, POLICY_VERSION

VALID_REVIEW_STATUSES = {
    "PENDING",
    "UNDER_REVIEW",
    "CONFIRMED_FRAUD",
    "FALSE_POSITIVE",
}
VALID_ACTIONS = {"APPROVE", "REVIEW", "BLOCK"}
ALLOWED_STATUS_TRANSITIONS = {
    "PENDING": {"UNDER_REVIEW", "CONFIRMED_FRAUD", "FALSE_POSITIVE"},
    "UNDER_REVIEW": {"CONFIRMED_FRAUD", "FALSE_POSITIVE"},
    "CONFIRMED_FRAUD": {"UNDER_REVIEW"},
    "FALSE_POSITIVE": {"UNDER_REVIEW"},
}


class DuplicateTransactionError(RuntimeError):
    """Raised when a source transaction is replayed."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(reset: bool = False) -> None:
    with _connect() as conn:
        cur = conn.cursor()
        if reset:
            cur.execute("DROP TABLE IF EXISTS review_events")
            cur.execute("DROP TABLE IF EXISTS transactions")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                tx_id TEXT PRIMARY KEY,
                transaction_id INTEGER,
                tx_datetime TEXT,
                customer_id INTEGER,
                terminal_id INTEGER,
                tx_amount REAL,
                country TEXT,
                device_id TEXT,
                channel TEXT,
                merchant_category TEXT,
                risk_score REAL CHECK(risk_score >= 0 AND risk_score <= 1),
                action TEXT CHECK(action IN ('APPROVE', 'REVIEW', 'BLOCK')),
                layer TEXT,
                latency_ms REAL CHECK(latency_ms >= 0),
                status TEXT,
                reasons TEXT,
                analyst_notes TEXT,
                ground_truth INTEGER,
                fraud_scenario TEXT,
                model_release TEXT,
                policy_version TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        existing_columns = {
            row[1] for row in cur.execute("PRAGMA table_info(transactions)").fetchall()
        }
        if "model_release" not in existing_columns:
            cur.execute("ALTER TABLE transactions ADD COLUMN model_release TEXT")
        if "policy_version" not in existing_columns:
            cur.execute("ALTER TABLE transactions ADD COLUMN policy_version TEXT")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS review_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_id TEXT,
                old_status TEXT,
                new_status TEXT,
                notes TEXT,
                event_time TEXT,
                FOREIGN KEY(tx_id) REFERENCES transactions(tx_id) ON DELETE CASCADE
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_transactions_review_queue "
            "ON transactions(action, status, risk_score DESC)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_transactions_datetime "
            "ON transactions(tx_datetime DESC)"
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_transactions_source_event "
            "ON transactions(transaction_id) WHERE transaction_id IS NOT NULL"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_review_events_tx_id "
            "ON review_events(tx_id, event_time DESC)"
        )
        conn.commit()


def _finite_float(
    record: dict[str, Any],
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = record.get(field)
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return result


def insert_transaction(record: dict[str, Any]) -> None:
    now = _utc_now()
    tx_id = str(record.get("tx_id", "")).strip()
    if not tx_id:
        raise ValueError("tx_id must be a non-empty string")
    action = str(record.get("action", ""))
    if action not in VALID_ACTIONS:
        raise ValueError(f"unknown transaction action: {action}")
    ground_truth = record.get("tx_fraud", 0)
    if isinstance(ground_truth, bool) or ground_truth not in {0, 1}:
        raise ValueError("tx_fraud must be 0 or 1")
    status = "PENDING" if action == "REVIEW" else "AUTO_RESOLVED"
    values = {
        "tx_id": tx_id,
        "transaction_id": (
            int(record["transaction_id"]) if record.get("transaction_id") is not None else None
        ),
        "tx_datetime": str(record.get("tx_datetime")),
        "customer_id": int(record.get("customer_id", -1)),
        "terminal_id": int(record.get("terminal_id", -1)),
        "tx_amount": _finite_float(record, "tx_amount", minimum=0),
        "country": str(record.get("country", "")),
        "device_id": str(record.get("device_id", "")),
        "channel": str(record.get("channel", "")),
        "merchant_category": str(record.get("merchant_category", "")),
        "risk_score": _finite_float(record, "risk_score", minimum=0, maximum=1),
        "action": action,
        "layer": str(record.get("layer", "")),
        "latency_ms": _finite_float(record, "latency_ms", minimum=0),
        "status": status,
        "reasons": str(record.get("reasons", "")),
        "analyst_notes": "",
        "ground_truth": int(ground_truth),
        "fraud_scenario": str(record.get("fraud_scenario", "unknown")),
        "model_release": str(record.get("model_release", "unknown")),
        "policy_version": str(record.get("policy_version", POLICY_VERSION)),
        "created_at": now,
        "updated_at": now,
    }
    with _connect() as conn:
        try:
            conn.execute(
                """
                INSERT INTO transactions (
                    tx_id, transaction_id, tx_datetime, customer_id, terminal_id,
                    tx_amount, country, device_id, channel, merchant_category,
                    risk_score, action, layer, latency_ms, status, reasons,
                    analyst_notes, ground_truth, fraud_scenario, model_release,
                    policy_version, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                tuple(values.values()),
            )
        except sqlite3.IntegrityError as exc:
            if "transactions.transaction_id" in str(exc):
                raise DuplicateTransactionError(
                    f"transaction_id already processed: {values['transaction_id']}"
                ) from exc
            raise
        conn.commit()


def transaction_exists(transaction_id: int) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM transactions WHERE transaction_id=? LIMIT 1",
            (transaction_id,),
        ).fetchone()
    return row is not None


def get_transactions(limit: int = 500) -> pd.DataFrame:
    limit = max(1, min(int(limit), 10_000))
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM transactions ORDER BY tx_datetime DESC LIMIT ?",
            conn,
            params=(limit,),
        )


def get_review_cases(status: str | None = None, limit: int = 500) -> pd.DataFrame:
    if status is not None and status not in VALID_REVIEW_STATUSES:
        raise ValueError(f"unknown review status: {status}")
    limit = max(1, min(int(limit), 10_000))
    query = "SELECT * FROM transactions WHERE action='REVIEW'"
    params: list[Any] = []
    if status:
        query += " AND status=?"
        params.append(status)
    query += " ORDER BY risk_score DESC, tx_datetime DESC LIMIT ?"
    params.append(limit)
    with _connect() as conn:
        return pd.read_sql_query(query, conn, params=params)


def update_transaction_status(tx_id: str, new_status: str, notes: str = "") -> None:
    if new_status not in VALID_REVIEW_STATUSES:
        raise ValueError(f"unknown review status: {new_status}")
    now = _utc_now()
    with _connect() as conn:
        old = conn.execute(
            "SELECT status, action FROM transactions WHERE tx_id=?", (tx_id,)
        ).fetchone()
        if old is None:
            raise KeyError(f"transaction not found: {tx_id}")
        old_status, action = old
        if action != "REVIEW":
            raise ValueError("only transactions routed to REVIEW can enter the analyst workflow")
        if old_status != new_status and new_status not in ALLOWED_STATUS_TRANSITIONS.get(old_status, set()):
            raise ValueError(f"invalid status transition: {old_status} -> {new_status}")
        conn.execute(
            "UPDATE transactions SET status=?, analyst_notes=?, updated_at=? WHERE tx_id=?",
            (new_status, notes, now, tx_id),
        )
        conn.execute(
            "INSERT INTO review_events (tx_id, old_status, new_status, notes, event_time) VALUES (?, ?, ?, ?, ?)",
            (tx_id, old_status, new_status, notes, now),
        )
        conn.commit()


def get_review_history(tx_id: str) -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT old_status, new_status, notes, event_time "
            "FROM review_events WHERE tx_id=? ORDER BY event_time DESC, id DESC",
            conn,
            params=(tx_id,),
        )


def analytics_snapshot(limit: int = 5000) -> dict[str, Any]:
    df = get_transactions(limit)
    if df.empty:
        return {}
    snapshot: dict[str, Any] = {
        "total": int(len(df)),
        "block_rate": float((df["action"] == "BLOCK").mean()),
        "review_rate": float((df["action"] == "REVIEW").mean()),
        "stage2_rate": float((df["layer"] == "Stage 2 (Advanced)").mean()),
        "avg_latency_ms": float(df["latency_ms"].mean()),
        "avg_risk_score": float(df["risk_score"].mean()),
        "model_releases": sorted(str(value) for value in df["model_release"].dropna().unique()),
        "policy_versions": sorted(str(value) for value in df["policy_version"].dropna().unique()),
    }
    resolved = df[df["status"].isin(["CONFIRMED_FRAUD", "FALSE_POSITIVE"])]
    if not resolved.empty:
        snapshot["analyst_confirmed_fraud_rate"] = float(
            (resolved["status"] == "CONFIRMED_FRAUD").mean()
        )
    return snapshot
