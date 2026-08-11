import joblib
import numpy as np
import pytest

from src import detector
from src.config import BLOCK_THRESHOLD, FEATURES, POLICY_VERSION, REVIEW_THRESHOLD, STAGE1_THRESHOLD
from src.model_registry import atomic_write_json, create_release_manifest


class FakeProbabilityModel:
    def __init__(self, marker: int, feature_count: int = len(FEATURES)):
        self.marker = marker
        self.n_features_in_ = feature_count
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)

    def predict_proba(self, values):
        probability = min(self.marker / 10, 1)
        return np.tile([1 - probability, probability], (len(values), 1))


def _reset_model_cache(monkeypatch, model_dir):
    monkeypatch.setattr(detector, "MODELS_DIR", model_dir)
    monkeypatch.setattr(detector, "_STAGE1", None)
    monkeypatch.setattr(detector, "_STAGE2", None)
    monkeypatch.setattr(detector, "_MODEL_SIGNATURE", None)
    monkeypatch.setattr(detector, "_ACTIVE_RELEASE_ID", None)


def _write_manifest(model_dir):
    manifest = create_release_manifest(
        model_dir,
        FEATURES,
        policy_version=POLICY_VERSION,
        thresholds={
            "stage1": STAGE1_THRESHOLD,
            "review": REVIEW_THRESHOLD,
            "block": BLOCK_THRESHOLD,
        },
    )
    atomic_write_json(model_dir / "model_manifest.json", manifest)
    return manifest


def test_load_models_reloads_changed_artifacts(tmp_path, monkeypatch):
    _reset_model_cache(monkeypatch, tmp_path)
    stage1_path = tmp_path / "stage1_model.joblib"
    stage2_path = tmp_path / "stage2_model.joblib"
    joblib.dump(FakeProbabilityModel(1), stage1_path)
    joblib.dump(FakeProbabilityModel(2), stage2_path)
    first_manifest = _write_manifest(tmp_path)

    first_stage1, first_stage2 = detector.load_models()
    assert (first_stage1.marker, first_stage2.marker) == (1, 2)
    assert detector.active_model_release() == first_manifest["release_id"]

    replacement_path = tmp_path / "replacement.joblib"
    joblib.dump(FakeProbabilityModel(3), replacement_path)
    replacement_path.replace(stage1_path)
    second_manifest = _write_manifest(tmp_path)

    reloaded_stage1, reloaded_stage2 = detector.load_models()
    assert reloaded_stage1.marker == 3
    assert reloaded_stage1 is not first_stage1
    assert reloaded_stage2.marker == 2
    assert detector.active_model_release() == second_manifest["release_id"]


def test_load_models_rejects_incompatible_feature_schema(tmp_path, monkeypatch):
    _reset_model_cache(monkeypatch, tmp_path)
    joblib.dump(FakeProbabilityModel(1, feature_count=1), tmp_path / "stage1_model.joblib")
    joblib.dump(FakeProbabilityModel(2), tmp_path / "stage2_model.joblib")
    _write_manifest(tmp_path)

    with pytest.raises(detector.ModelArtifactError, match="expects 1 features"):
        detector.load_models()


def test_load_models_rejects_checksum_mismatch(tmp_path, monkeypatch):
    _reset_model_cache(monkeypatch, tmp_path)
    stage1_path = tmp_path / "stage1_model.joblib"
    joblib.dump(FakeProbabilityModel(1), stage1_path)
    joblib.dump(FakeProbabilityModel(2), tmp_path / "stage2_model.joblib")
    _write_manifest(tmp_path)
    with stage1_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(detector.ModelArtifactError, match="mismatch"):
        detector.load_models()
