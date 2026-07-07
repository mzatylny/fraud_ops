"""SQLite persistence for transactions, review cases, and feedback."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DB_PATH


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(reset: bool = False) -> None:
    with _connect() as conn:
        cur = conn.cursor()
        if reset:
            cur.execute("DROP TABLE IF EXISTS transactions")
            cur.execute("DROP TABLE IF EXISTS review_events")
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
                risk_score REAL,
                action TEXT,
                layer TEXT,
                latency_ms REAL,
                status TEXT,
                reasons TEXT,
                analyst_notes TEXT,
                ground_truth INTEGER,
                fraud_scenario TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS review_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_id TEXT,
                old_status TEXT,
                new_status TEXT,
                notes TEXT,
                event_time TEXT,
                FOREIGN KEY(tx_id) REFERENCES transactions(tx_id)
            )
            """
        )
        conn.commit()


def insert_transaction(record: dict[str, Any]) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds")
    status = "PENDING" if record["action"] == "REVIEW" else "AUTO_RESOLVED"
    values = {
        "tx_id": record["tx_id"],
        "transaction_id": int(record.get("transaction_id", -1)),
        "tx_datetime": str(record.get("tx_datetime")),
        "customer_id": int(record.get("customer_id", -1)),
        "terminal_id": int(record.get("terminal_id", -1)),
        "tx_amount": float(record.get("tx_amount", 0.0)),
        "country": str(record.get("country", "")),
        "device_id": str(record.get("device_id", "")),
        "channel": str(record.get("channel", "")),
        "merchant_category": str(record.get("merchant_category", "")),
        "risk_score": float(record.get("risk_score", 0.0)),
        "action": str(record.get("action", "")),
        "layer": str(record.get("layer", "")),
        "latency_ms": float(record.get("latency_ms", 0.0)),
        "status": status,
        "reasons": str(record.get("reasons", "")),
        "analyst_notes": "",
        "ground_truth": int(record.get("tx_fraud", 0)),
        "fraud_scenario": str(record.get("fraud_scenario", "unknown")),
        "created_at": now,
        "updated_at": now,
    }
    cols = ",".join(values.keys())
    placeholders = ",".join("?" for _ in values)
    with _connect() as conn:
        conn.execute(f"INSERT OR REPLACE INTO transactions ({cols}) VALUES ({placeholders})", tuple(values.values()))
        conn.commit()


def get_transactions(limit: int = 500) -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM transactions ORDER BY tx_datetime DESC LIMIT ?",
            conn,
            params=(limit,),
        )


def get_review_cases(status: str | None = None, limit: int = 500) -> pd.DataFrame:
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
    now = datetime.utcnow().isoformat(timespec="seconds")
    with _connect() as conn:
        old = conn.execute("SELECT status FROM transactions WHERE tx_id=?", (tx_id,)).fetchone()
        old_status = old[0] if old else "UNKNOWN"
        conn.execute(
            "UPDATE transactions SET status=?, analyst_notes=?, updated_at=? WHERE tx_id=?",
            (new_status, notes, now, tx_id),
        )
        conn.execute(
            "INSERT INTO review_events (tx_id, old_status, new_status, notes, event_time) VALUES (?, ?, ?, ?, ?)",
            (tx_id, old_status, new_status, notes, now),
        )
        conn.commit()


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
    }
    resolved = df[df["status"].isin(["CONFIRMED_FRAUD", "FALSE_POSITIVE"])]
    if not resolved.empty:
        snapshot["analyst_precision"] = float((resolved["status"] == "CONFIRMED_FRAUD").mean())
    return snapshot
