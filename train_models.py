"""Train and evaluate the Fraud Operations Platform models.

Run:
    python train_models.py --customers 900 --terminals 1800 --days 60
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    BLOCK_THRESHOLD,
    DEFAULT_CUSTOMER_AVG_AMOUNT,
    EXPERIMENT_METADATA_JSON,
    METRICS_CSV,
    MODEL_MANIFEST_JSON,
    MODELS_DIR,
    MONITORING_BASELINE_JSON,
    POLICY_VERSION,
    REPORTS_DIR,
    REVIEW_THRESHOLD,
    STAGE1_THRESHOLD,
    THRESHOLD_CSV,
)
from src.dataset_generator import SimulationConfig, build_dataset
from src.features import build_batch_features
from src.model_registry import atomic_write_json, create_release_manifest
from src.model_wrappers import ProbabilityAveragingEnsemble
from src.monitoring import create_monitoring_baseline


def atomic_joblib_dump(model: Any, destination: Path) -> None:
    """Replace a model artifact only after its complete contents reach disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        joblib.dump(model, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


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


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    enriched: pd.DataFrame,
    test_ratio: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Split chronologically while preserving alignment across all three objects."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    if not (len(X) == len(y) == len(enriched)):
        raise ValueError("X, y, and enriched must contain the same number of rows")
    if len(enriched) < 8:
        raise ValueError("at least 8 transactions are required for a temporal split")

    order = enriched.assign(_position=np.arange(len(enriched))).sort_values(
        ["tx_datetime", "transaction_id"]
    )["_position"].to_numpy()
    split_idx = int(len(order) * (1 - test_ratio))
    train_positions = order[:split_idx]
    test_positions = order[split_idx:]
    return (
        X.iloc[train_positions].reset_index(drop=True),
        X.iloc[test_positions].reset_index(drop=True),
        y.iloc[train_positions].reset_index(drop=True),
        y.iloc[test_positions].reset_index(drop=True),
        enriched.iloc[test_positions].reset_index(drop=True),
    )


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
    if len(y_true) == 0:
        raise ValueError("y_true must not be empty")
    if not 0 < k_percent <= 100:
        raise ValueError("k_percent must be between 0 and 100")
    k = max(1, int(len(y_true) * k_percent / 100))
    idx = np.argsort(y_prob)[::-1][:k]
    return float(np.mean(np.asarray(y_true)[idx]))


def latency_ms(model: object, X_test: pd.DataFrame, sample_size: int = 1000) -> float:
    if X_test.empty:
        raise ValueError("X_test must not be empty")
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
    if X_test.empty:
        raise ValueError("X_test must not be empty")
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
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise RuntimeError(
            "The temporal train/test split does not contain both classes. "
            "Generate more transactions or increase the number of simulated days."
        )

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

    atomic_joblib_dump(fitted["Logistic Regression"], MODELS_DIR / "stage1_model.joblib")
    atomic_joblib_dump(
        fitted["Advanced Soft-Voting Ensemble"], MODELS_DIR / "stage2_model.joblib"
    )

    trained_at_utc = datetime.now(UTC).isoformat(timespec="seconds")
    thresholds = {
        "stage1": STAGE1_THRESHOLD,
        "review": REVIEW_THRESHOLD,
        "block": BLOCK_THRESHOLD,
    }
    manifest = create_release_manifest(
        MODELS_DIR,
        features=list(X.columns),
        policy_version=POLICY_VERSION,
        thresholds=thresholds,
        created_at_utc=trained_at_utc,
    )
    atomic_write_json(MODEL_MANIFEST_JSON, manifest)
    baseline = create_monitoring_baseline(
        test_rows,
        hybrid_probs,
        model_release=manifest["release_id"],
    )
    atomic_write_json(MONITORING_BASELINE_JSON, baseline)

    metadata = {
        "artifact_version": 3,
        "trained_at_utc": trained_at_utc,
        "model_release": manifest["release_id"],
        "policy_version": POLICY_VERSION,
        "n_transactions": int(len(tx)),
        "fraud_rate": float(tx.tx_fraud.mean()),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "thresholds": thresholds,
        "stage2_call_rate": float(stage2_call_rate),
        "features": list(X.columns),
        "feature_count": int(X.shape[1]),
        "cold_start_customer_avg_amount": DEFAULT_CUSTOMER_AVG_AMOUNT,
        "test_start": pd.Timestamp(test_rows["tx_datetime"].min()).isoformat(),
        "test_end": pd.Timestamp(test_rows["tx_datetime"].max()).isoformat(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "release_manifest": str(MODEL_MANIFEST_JSON.relative_to(MODEL_MANIFEST_JSON.parent.parent)),
        "monitoring_baseline": str(
            MONITORING_BASELINE_JSON.relative_to(MONITORING_BASELINE_JSON.parent.parent)
        ),
    }
    atomic_write_json(EXPERIMENT_METADATA_JSON, metadata)

    print("\n--- MODEL COMPARISON FOR REPORT ---")
    print(results_df.round(4).to_string())
    print(f"\nSaved metrics to {METRICS_CSV}")
    print(f"Saved threshold sweep to {THRESHOLD_CSV}")
    print("Saved models to models/stage1_model.joblib and models/stage2_model.joblib")
    print(f"Model release: {manifest['release_id']} | policy: {POLICY_VERSION}")
    print(f"Saved release manifest to {MODEL_MANIFEST_JSON}")
    print(f"Saved monitoring baseline to {MONITORING_BASELINE_JSON}")


if __name__ == "__main__":
    main()
