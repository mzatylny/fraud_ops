import json

import pytest

from src.model_registry import (
    ModelManifestError,
    atomic_write_json,
    create_release_manifest,
    feature_schema_sha256,
    validate_release_manifest,
)


def test_release_manifest_is_reproducible_and_validated(tmp_path):
    (tmp_path / "stage1_model.joblib").write_bytes(b"stage-one")
    (tmp_path / "stage2_model.joblib").write_bytes(b"stage-two")
    features = ["amount", "velocity"]
    manifest = create_release_manifest(
        tmp_path,
        features,
        "policy-v1",
        {"stage1": 0.25, "review": 0.35, "block": 0.85},
        created_at_utc="2026-08-11T00:00:00+00:00",
    )
    path = tmp_path / "model_manifest.json"
    atomic_write_json(path, manifest)

    validated = validate_release_manifest(path, tmp_path, features)
    assert validated["release_id"] == manifest["release_id"]
    assert validated["feature_schema_sha256"] == feature_schema_sha256(features)
    assert json.loads(path.read_text())["policy_version"] == "policy-v1"


def test_manifest_detects_artifact_tampering(tmp_path):
    stage1 = tmp_path / "stage1_model.joblib"
    stage1.write_bytes(b"stage-one")
    (tmp_path / "stage2_model.joblib").write_bytes(b"stage-two")
    manifest = create_release_manifest(tmp_path, ["amount"], "v1", {})
    path = tmp_path / "model_manifest.json"
    atomic_write_json(path, manifest)
    stage1.write_bytes(b"tampered-model")

    with pytest.raises(ModelManifestError, match="mismatch"):
        validate_release_manifest(path, tmp_path, ["amount"])


def test_manifest_rejects_policy_mismatch(tmp_path):
    (tmp_path / "stage1_model.joblib").write_bytes(b"stage-one")
    (tmp_path / "stage2_model.joblib").write_bytes(b"stage-two")
    manifest = create_release_manifest(tmp_path, ["amount"], "policy-v1", {"review": 0.5})
    path = tmp_path / "model_manifest.json"
    atomic_write_json(path, manifest)

    with pytest.raises(ModelManifestError, match="policy version"):
        validate_release_manifest(
            path,
            tmp_path,
            ["amount"],
            expected_policy_version="policy-v2",
        )
