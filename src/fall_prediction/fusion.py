"""Combine feature groups or model probabilities when they are available."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import ContractValidationError

MISSING_MODALITY_POLICIES = {"error", "available_only", "impute_zero"}


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    feature_columns: tuple[str, ...]
    required: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.feature_columns:
            raise ValueError("A modality requires a name and at least one feature column.")


def _policy(policy):
    if policy not in MISSING_MODALITY_POLICIES:
        raise ValueError(f"Unknown missing-modality policy {policy!r}; expected {sorted(MISSING_MODALITY_POLICIES)}")


def early_fuse_features(frame, modalities, *, missing_policy="error"):
    """Select the declared feature groups and record missing-data handling."""
    _policy(missing_policy)
    if not modalities:
        raise ValueError("At least one modality must be declared.")
    selected = []
    used = []
    dropped = []
    imputed = []
    working = frame.copy()
    for mod in modalities:
        absent = [c for c in mod.feature_columns if c not in working.columns]
        incomplete = bool(absent) or working.loc[:, [c for c in mod.feature_columns if c in working.columns]].isna().any().any()
        if incomplete and (mod.required or missing_policy == "error"):
            raise ContractValidationError(f"Required modality {mod.name!r} is unavailable: {absent or ['null values']}")
        if incomplete and missing_policy == "available_only":
            dropped.append(mod.name)
            continue
        if incomplete:
            for c in absent:
                working[c] = 0.0
            imputed.append(mod.name)
        selected.extend(mod.feature_columns)
        used.append(mod.name)
    if not selected:
        raise ContractValidationError("No usable modality remains after missing-modality handling.")
    output = working.loc[:, selected].apply(pd.to_numeric, errors="coerce")
    if missing_policy == "impute_zero":
        output = output.fillna(0.0)
    if output.isna().any().any() or not np.isfinite(output.to_numpy(dtype=float)).all():
        raise ContractValidationError("Early-fusion input contains missing or non-finite values.")
    return output, {"fusion_type": "early", "used_modalities": used, "dropped_modalities": dropped, "zero_imputed_modalities": imputed}


def late_fuse_probabilities(probabilities, *, weights=None, required_modalities=(), missing_policy="error"):
    """Return a weighted average of the supplied probability vectors."""
    _policy(missing_policy)
    if not probabilities:
        raise ValueError("At least one modality probability vector is required.")
    arrays = {}
    size = None
    for name, vals in probabilities.items():
        arr = np.asarray(vals, dtype=float)
        if arr.ndim != 1 or not len(arr) or not np.isfinite(arr).all() or ((arr < 0) | (arr > 1)).any():
            raise ValueError(f"Probability vector for {name!r} must be finite, non-empty and in [0, 1].")
        if size is not None and len(arr) != size:
            raise ValueError("All modality probability vectors must have equal length.")
        arrays[name] = arr
        size = len(arr)
    missing_required = set(required_modalities) - set(arrays)
    if missing_required:
        raise ContractValidationError(f"Missing required probability modalities: {sorted(missing_required)}")
    weights = {name: 1.0 for name in arrays} if weights is None else dict(weights)
    if set(weights) - set(arrays):
        raise ValueError("Weights supplied for unavailable modalities.")
    active = {name: float(weights.get(name, 1.0)) for name in arrays}
    if any(w < 0 for w in active.values()) or not any(active.values()):
        raise ValueError("Fusion weights must be non-negative and sum to a positive value.")
    fused = np.average(np.vstack(list(arrays.values())), axis=0, weights=np.asarray(list(active.values())))
    return fused, {"fusion_type": "late", "used_modalities": list(arrays), "missing_policy": missing_policy, "weights": active}


def hybrid_fuse_probabilities(early_probability, late_probabilities, *, early_weight=0.5, late_weights=None, missing_policy="error"):
    if not 0 <= early_weight <= 1:
        raise ValueError("early_weight must be in [0, 1].")
    early = np.asarray(early_probability, dtype=float)
    if early.ndim != 1 or not len(early) or not np.isfinite(early).all() or ((early < 0) | (early > 1)).any():
        raise ValueError("Early-fusion probabilities must be finite, non-empty and in [0, 1].")
    late, meta = late_fuse_probabilities(
        late_probabilities, weights=late_weights, missing_policy=missing_policy
    )
    if len(early) != len(late):
        raise ValueError("Early and late probability vectors must have equal length.")
    return early_weight * early + (1 - early_weight) * late, {
        "fusion_type": "hybrid",
        "early_weight": early_weight,
        "late": meta,
    }
