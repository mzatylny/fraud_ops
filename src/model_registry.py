"""Integrity-checked model release manifests."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

MANIFEST_SCHEMA_VERSION = 1
REQUIRED_ARTIFACTS = ("stage1_model.joblib", "stage2_model.joblib")


class ModelManifestError(RuntimeError):
    """Raised when a model release manifest or artifact is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_schema_sha256(features: list[str]) -> str:
    payload = json.dumps(features, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(destination: Path, payload: dict[str, Any]) -> None:
    """Durably replace a JSON artifact after complete serialization."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        json.dump(payload, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    try:
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def create_release_manifest(
    models_dir: Path,
    features: list[str],
    policy_version: str,
    thresholds: dict[str, float],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    for filename in REQUIRED_ARTIFACTS:
        path = models_dir / filename
        if not path.is_file():
            raise ModelManifestError(f"model artifact is missing: {path}")
        artifacts[filename] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    schema_digest = feature_schema_sha256(features)
    release_material = {
        "artifacts": artifacts,
        "feature_schema_sha256": schema_digest,
        "policy_version": policy_version,
        "thresholds": thresholds,
    }
    release_id = _release_id(release_material)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "release_id": release_id,
        "created_at_utc": created_at_utc or datetime.now(UTC).isoformat(timespec="seconds"),
        "feature_schema": features,
        "feature_schema_sha256": schema_digest,
        "policy_version": policy_version,
        "thresholds": thresholds,
        "artifacts": artifacts,
    }


def validate_release_manifest(
    manifest_path: Path,
    models_dir: Path,
    expected_features: list[str],
    expected_policy_version: str | None = None,
    expected_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelManifestError(f"could not read model manifest: {exc}") from exc
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ModelManifestError("unsupported model manifest schema version")
    if manifest.get("feature_schema") != expected_features:
        raise ModelManifestError("model manifest feature schema does not match the application")
    if manifest.get("feature_schema_sha256") != feature_schema_sha256(expected_features):
        raise ModelManifestError("model manifest feature schema digest is invalid")
    release_id = manifest.get("release_id")
    if not isinstance(release_id, str) or len(release_id) != 16:
        raise ModelManifestError("model manifest release_id is invalid")
    if expected_policy_version is not None and manifest.get("policy_version") != expected_policy_version:
        raise ModelManifestError("model manifest policy version does not match the application")
    if expected_thresholds is not None and manifest.get("thresholds") != expected_thresholds:
        raise ModelManifestError("model manifest thresholds do not match the application")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(REQUIRED_ARTIFACTS):
        raise ModelManifestError("model manifest artifact set is invalid")
    for filename in REQUIRED_ARTIFACTS:
        metadata = artifacts[filename]
        path = models_dir / filename
        if not path.is_file():
            raise ModelManifestError(f"model artifact is missing: {path}")
        if not isinstance(metadata, dict) or metadata.get("bytes") != path.stat().st_size:
            raise ModelManifestError(f"model artifact size mismatch: {filename}")
        if metadata.get("sha256") != sha256_file(path):
            raise ModelManifestError(f"model artifact checksum mismatch: {filename}")
    release_material = {
        "artifacts": artifacts,
        "feature_schema_sha256": manifest["feature_schema_sha256"],
        "policy_version": manifest.get("policy_version"),
        "thresholds": manifest.get("thresholds"),
    }
    if release_id != _release_id(release_material):
        raise ModelManifestError("model manifest release_id does not match its contents")
    return manifest


def _release_id(release_material: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(release_material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
