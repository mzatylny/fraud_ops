"""Train and evaluate the Fraud Operations Platform models.

Run:
    python train_models.py --customers 900 --terminals 1800 --days 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import METRICS_CSV, MODELS_DIR, REPORTS_DIR, STAGE1_THRESHOLD, THRESHOLD_CSV
from src.dataset_generator import SimulationConfig, build_dataset
from src.features import build_batch_features
from src.model_wrappers import ProbabilityAveragingEnsemble


def build_models() -> dict[str, object]:
    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1500, random_state=42)),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=70,
        max_depth=11,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    extra = ExtraTreesClassifier(
        n_estimators=90,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    ensemble = ProbabilityAveragingEnsemble(
        estimators=[("rf", rf), ("extra", extra)],
        weights=[0.55, 0.45],
    )
    return {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "Extra Trees": extra,
        "Advanced Soft-Voting Ensemble": ensemble,
    }


def temporal_split(X: pd.DataFrame, y: pd.Series, enriched: pd.DataFrame, test_ratio: float = 0.25):
    enriched = enriched.sort_values("tx_datetime").reset_index(drop=True)
    split_idx = int(len(enriched) * (1 - test_ratio))
    train_idx = enriched.index[:split_idx]
    test_idx = enriched.index[split_idx:]
    return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx], enriched.loc[test_idx]


def threshold_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    pr = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc,
        "pr_auc": pr,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def precision_at_k(y_true: pd.Series, y_prob: np.ndarray, k_percent: float = 1.0) -> float:
    k = max(1, int(len(y_true) * k_percent / 100))
    idx = np.argsort(y_prob)[::-1][:k]
    return float(np.mean(np.asarray(y_true)[idx]))


def latency_ms(model: object, X_test: pd.DataFrame, sample_size: int = 1000) -> float:
    sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    start = time.perf_counter()
    _ = model.predict_proba(sample)
    return float((time.perf_counter() - start) / len(sample) * 1000)


def sweep_thresholds(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        row = threshold_metrics(y_true, y_prob, threshold=threshold)
        row["threshold"] = threshold
        rows.append(row)
    return pd.DataFrame(rows)


def simulate_two_stage(stage1: object, stage2: object, X_test: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    start = time.perf_counter()
    p1 = stage1.predict_proba(X_test)[:, 1]
    probs = p1.copy()
    suspicious = p1 >= STAGE1_THRESHOLD
    if suspicious.any():
        probs[suspicious] = stage2.predict_proba(X_test.loc[suspicious])[:, 1]
    ms = (time.perf_counter() - start) / len(X_test) * 1000
    return probs, float(suspicious.mean()), float(ms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=900)
    parser.add_argument("--terminals", type=int, default=1800)
    parser.add_argument("--days", type=int, default=55)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--max-transactions", type=int, default=None)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1) Building handbook-inspired dataset...")
    tx, customers, terminals = build_dataset(
        SimulationConfig(
            n_customers=args.customers,
            n_terminals=args.terminals,
            n_days=args.days,
            radius=args.radius,
            max_transactions=args.max_transactions,
        )
    )
    print(f"Transactions: {len(tx):,} | Fraud rate: {tx.tx_fraud.mean() * 100:.3f}%")
    print(tx["fraud_scenario"].value_counts().head(10).to_string())

    print("\n2) Building behavioural features...")
    X, y, enriched = build_batch_features(tx)
    X_train, X_test, y_train, y_test, test_rows = temporal_split(X, y, enriched)

    print("\n3) Training model candidates...")
    models = build_models()
    results = []
    fitted = {}
    for name, model in models.items():
        print(f"   - {name}")
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        row = threshold_metrics(y_test, probs, threshold=0.5)
        row["model"] = name
        row["precision_at_1_percent"] = precision_at_k(y_test, probs, 1.0)
        row["inference_ms_per_tx"] = latency_ms(model, X_test)
        results.append(row)
        fitted[name] = model

    print("\n4) Simulating two-stage architecture...")
    hybrid_probs, stage2_call_rate, hybrid_latency = simulate_two_stage(
        fitted["Logistic Regression"], fitted["Advanced Soft-Voting Ensemble"], X_test
    )
    hybrid = threshold_metrics(y_test, hybrid_probs, threshold=0.5)
    hybrid.update(
        {
            "model": "Two-Stage Hybrid",
            "precision_at_1_percent": precision_at_k(y_test, hybrid_probs, 1.0),
            "inference_ms_per_tx": hybrid_latency,
            "stage2_call_rate": stage2_call_rate,
        }
    )
    results.append(hybrid)

    results_df = pd.DataFrame(results).set_index("model")
    if "stage2_call_rate" not in results_df.columns:
        results_df["stage2_call_rate"] = np.nan
    results_df.to_csv(METRICS_CSV)

    threshold_df = sweep_thresholds(y_test, hybrid_probs)
    threshold_df.to_csv(THRESHOLD_CSV, index=False)

    joblib.dump(fitted["Logistic Regression"], MODELS_DIR / "stage1_model.joblib")
    joblib.dump(fitted["Advanced Soft-Voting Ensemble"], MODELS_DIR / "stage2_model.joblib")

    metadata = {
        "n_transactions": int(len(tx)),
        "fraud_rate": float(tx.tx_fraud.mean()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "stage1_threshold": STAGE1_THRESHOLD,
        "stage2_call_rate": float(stage2_call_rate),
        "features": list(X.columns),
    }
    with open(REPORTS_DIR / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n--- MODEL COMPARISON FOR REPORT ---")
    print(results_df.round(4).to_markdown())
    print(f"\nSaved metrics to {METRICS_CSV}")
    print(f"Saved threshold sweep to {THRESHOLD_CSV}")
    print("Saved models to models/stage1_model.joblib and models/stage2_model.joblib")


if __name__ == "__main__":
    main()
