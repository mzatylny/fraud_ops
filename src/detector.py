"""Two-stage scoring service with latency measurement and explanations."""

from __future__ import annotations

import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import (
    BLOCK_THRESHOLD,
    FEATURES,
    MODELS_DIR,
    POLICY_VERSION,
    REVIEW_THRESHOLD,
    STAGE1_THRESHOLD,
)
from .model_registry import ModelManifestError, validate_release_manifest

_STAGE1 = None
_STAGE2 = None
_MODEL_SIGNATURE: tuple[tuple[int, int, int], ...] | None = None
_ACTIVE_RELEASE_ID: str | None = None


class ModelArtifactError(RuntimeError):
    """Raised when trained model artifacts are unavailable or invalid."""


def _validate_model(model: Any, name: str) -> None:
    if not callable(getattr(model, "predict_proba", None)):
        raise ModelArtifactError(f"{name} does not provide predict_proba")
    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is not None and int(feature_count) != len(FEATURES):
        raise ModelArtifactError(
            f"{name} expects {feature_count} features, but the application expects {len(FEATURES)}"
        )
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None and [str(value) for value in feature_names] != FEATURES:
        raise ModelArtifactError(f"{name} was trained with a different feature schema")


def load_models() -> tuple[Any, Any]:
    global _ACTIVE_RELEASE_ID, _MODEL_SIGNATURE, _STAGE1, _STAGE2
    stage1_path = MODELS_DIR / "stage1_model.joblib"
    stage2_path = MODELS_DIR / "stage2_model.joblib"
    manifest_path = MODELS_DIR / "model_manifest.json"
    paths = (stage1_path, stage2_path, manifest_path)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        _STAGE1 = None
        _STAGE2 = None
        _MODEL_SIGNATURE = None
        _ACTIVE_RELEASE_ID = None
        raise ModelArtifactError(
            "Model release artifacts are missing. Run `python train_models.py` first. Missing: "
            + ", ".join(missing)
        )
    signature = tuple(
        (path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_size) for path in paths
    )
    if _STAGE1 is not None and _STAGE2 is not None and signature == _MODEL_SIGNATURE:
        return _STAGE1, _STAGE2
    try:
        manifest = validate_release_manifest(
            manifest_path,
            MODELS_DIR,
            FEATURES,
            expected_policy_version=POLICY_VERSION,
            expected_thresholds={
                "stage1": STAGE1_THRESHOLD,
                "review": REVIEW_THRESHOLD,
                "block": BLOCK_THRESHOLD,
            },
        )
        stage1 = joblib.load(stage1_path)
        stage2 = joblib.load(stage2_path)
        _validate_model(stage1, "Stage 1 model")
        _validate_model(stage2, "Stage 2 model")
    except Exception as exc:
        _STAGE1 = None
        _STAGE2 = None
        _MODEL_SIGNATURE = None
        _ACTIVE_RELEASE_ID = None
        if isinstance(exc, ModelArtifactError):
            raise
        if isinstance(exc, ModelManifestError):
            raise ModelArtifactError(str(exc)) from exc
        raise ModelArtifactError(f"Could not load model artifacts: {exc}") from exc
    _STAGE1 = stage1
    _STAGE2 = stage2
    _MODEL_SIGNATURE = signature
    _ACTIVE_RELEASE_ID = str(manifest["release_id"])
    return _STAGE1, _STAGE2


def active_model_release() -> str:
    if _ACTIVE_RELEASE_ID is None:
        load_models()
    if _ACTIVE_RELEASE_ID is None:  # pragma: no cover - defensive invariant
        raise ModelArtifactError("model release is unavailable")
    return _ACTIVE_RELEASE_ID


def explain(features: dict[str, float], score: float, layer: str) -> str:
    reasons = []
    if features["amount_ratio_customer"] >= 4.0:
        reasons.append("amount much higher than customer history")
    if features["customer_tx_count_1h"] >= 3:
        reasons.append("rapid customer velocity within one hour")
    if features["terminal_tx_count_24h"] >= 25:
        reasons.append("unusually busy terminal in last 24h")
    if features["is_new_terminal_for_customer"] == 1:
        reasons.append("first time at this terminal")
    if features["is_new_device"] == 1:
        reasons.append("new customer device")
    if features["is_foreign_country"] == 1:
        reasons.append("transaction country differs from customer home country")
    if features["terminal_risk_score"] >= 0.12:
        reasons.append("high-risk merchant category / terminal")
    if features["time_since_last_tx_seconds"] < 90:
        reasons.append("very short time since previous transaction")
    if score >= 0.85:
        reasons.append("advanced ensemble returned severe risk")
    if not reasons:
        reasons.append("normal behavioural profile")
    return f"{layer}: " + "; ".join(reasons)


def score_transaction(features_df: pd.DataFrame, raw_features: dict[str, float]) -> tuple[float, str, str, float]:
    stage1, stage2 = load_models()
    start = time.perf_counter()
    p1 = float(stage1.predict_proba(features_df)[0, 1])
    if not np.isfinite(p1) or not 0 <= p1 <= 1:
        raise ModelArtifactError("Stage 1 returned an invalid probability")
    if p1 >= STAGE1_THRESHOLD:
        p2 = float(stage2.predict_proba(features_df)[0, 1])
        if not np.isfinite(p2) or not 0 <= p2 <= 1:
            raise ModelArtifactError("Stage 2 returned an invalid probability")
        layer = "Stage 2 (Advanced)"
        score = p2
    else:
        layer = "Stage 1 (Fast Screen)"
        score = p1
    latency = (time.perf_counter() - start) * 1000
    return score, layer, explain(raw_features, score, layer), latency
