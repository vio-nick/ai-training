"""Immutable, integrity-checked model releases for offline inference."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import joblib
import sklearn

from .inference import FallerInferenceModel

RELEASE_FORMAT = "gstride_fall_model_release_v1"
MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "manifest.json"


class ReleaseIntegrityError(ValueError):
    """Raised when a model release is incomplete, incompatible, or altered."""


def sha256_file(path):
    """Return a file digest without loading its full content into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_manifest(manifest):
    required = {
        "release_format", "release_id", "model_filename", "model_sha256",
        "data_version", "model_name", "decision_threshold", "feature_columns",
        "training", "runtime",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ReleaseIntegrityError(f"Release manifest is missing required field(s): {missing}")
    if manifest["release_format"] != RELEASE_FORMAT:
        raise ReleaseIntegrityError(
            f"Unsupported release format {manifest['release_format']!r}; expected {RELEASE_FORMAT!r}."
        )
    if manifest["model_filename"] != MODEL_FILENAME:
        raise ReleaseIntegrityError("Release manifest names an unexpected model file.")
    thresh = manifest["decision_threshold"]
    if not isinstance(thresh, (int, float)) or not 0 <= float(thresh) <= 1:
        raise ReleaseIntegrityError("Release manifest contains an invalid decision threshold.")
    if not isinstance(manifest["feature_columns"], list) or not manifest["feature_columns"]:
        raise ReleaseIntegrityError("Release manifest must declare at least one feature column.")


def _release_manifest(model, *, release_id, training):
    return {
        "release_format": RELEASE_FORMAT,
        "release_id": release_id,
        "model_filename": MODEL_FILENAME,
        "model_sha256": "pending",
        "data_version": model.contract.data_version,
        "model_name": model.model_name,
        "decision_threshold": model.threshold,
        "feature_columns": list(model.contract.feature_columns),
        "id_column": model.contract.id_column,
        "training": dict(training),
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }


def save_model_release(model, release_dir, *, release_id, training, overwrite=False):
    """Persist one fully fitted model package and its integrity manifest."""
    if not isinstance(model, FallerInferenceModel):
        raise TypeError("Only FallerInferenceModel instances can be released.")
    dest = Path(release_dir)
    dest.mkdir(parents=True, exist_ok=True)
    model_path = dest / MODEL_FILENAME
    manifest_path = dest / MANIFEST_FILENAME
    if not overwrite and (model_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            f"Release destination already has a model or manifest: {dest}. Use overwrite=True to replace it."
        )

    tmp_model = dest / f".{MODEL_FILENAME}.tmp"
    try:
        joblib.dump(model, tmp_model)
        os.replace(tmp_model, model_path)
        manifest = _release_manifest(model, release_id=release_id, training=training)
        manifest["model_sha256"] = sha256_file(model_path)
        tmp_manifest = dest / f".{MANIFEST_FILENAME}.tmp"
        tmp_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(tmp_manifest, manifest_path)
    finally:
        if tmp_model.exists():
            tmp_model.unlink()

    return manifest


def load_model_release(release_dir):
    """Load a trusted model release after verifying its file digest and contract."""
    d = Path(release_dir)
    manifest_path = d / MANIFEST_FILENAME
    model_path = d / MODEL_FILENAME
    if not manifest_path.is_file() or not model_path.is_file():
        raise ReleaseIntegrityError("Release must contain both manifest.json and model.joblib.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseIntegrityError(f"Cannot read release manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ReleaseIntegrityError("Release manifest must be a JSON object.")
    _validate_manifest(manifest)
    actual_hash = sha256_file(model_path)
    if actual_hash != manifest["model_sha256"]:
        raise ReleaseIntegrityError("Model SHA-256 does not match the release manifest.")

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise ReleaseIntegrityError(f"Cannot load model artifact: {exc}") from exc
    if not isinstance(model, FallerInferenceModel):
        raise ReleaseIntegrityError("Serialized artifact is not a FallerInferenceModel.")
    if (
        model.contract.data_version != manifest["data_version"]
        or model.model_name != manifest["model_name"]
        or list(model.contract.feature_columns) != manifest["feature_columns"]
        or model.threshold != float(manifest["decision_threshold"])
    ):
        raise ReleaseIntegrityError("Loaded model contract does not match the release manifest.")
    return model, manifest
