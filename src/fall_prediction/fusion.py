"""Explicit feature/probability fusion interfaces for future hospital modalities.

GSTRIDE currently provides only gait features.  These generic functions keep
future BMD or clinical fusion versioned and prevent silent missing-modality use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import ContractValidationError, ensure_columns_exist

MISSING_MODALITY_POLICIES = {"error", "available_only", "impute_zero"}


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    feature_columns: tuple[str, ...]
    required: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.feature_columns:
            raise ValueError("A modality requires a name and at least one feature column.")


def _policy(policy: str) -> None:
    if policy not in MISSING_MODALITY_POLICIES:
        raise ValueError(f"Unknown missing-modality policy {policy!r}; expected {sorted(MISSING_MODALITY_POLICIES)}")


def modality_availability(frame: pd.DataFrame, modalities: Sequence[ModalitySpec]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for modality in modalities:
        ensure_columns_exist(modality.feature_columns, frame, modality.name)
        result[modality.name] = not frame.loc[:, list(modality.feature_columns)].isna().any().any()
    return result


def early_fuse_features(frame: pd.DataFrame, modalities: Sequence[ModalitySpec], *, missing_policy: str = "error") -> tuple[pd.DataFrame, dict[str, object]]:
    """Concatenate declared modality features and record the degradation path."""
    _policy(missing_policy)
    if not modalities:
        raise ValueError("At least one modality must be declared.")
    selected: list[str] = []
    used: list[str] = []
    dropped: list[str] = []
    imputed: list[str] = []
    working = frame.copy()
    for modality in modalities:
        absent = [column for column in modality.feature_columns if column not in working.columns]
        incomplete = bool(absent) or working.loc[:, [c for c in modality.feature_columns if c in working.columns]].isna().any().any()
        if incomplete and (modality.required or missing_policy == "error"):
            raise ContractValidationError(f"Required modality {modality.name!r} is unavailable: {absent or ['null values']}")
        if incomplete and missing_policy == "available_only":
            dropped.append(modality.name)
            continue
        if incomplete:
            for column in absent:
                working[column] = 0.0
            imputed.append(modality.name)
        selected.extend(modality.feature_columns)
        used.append(modality.name)
    if not selected:
        raise ContractValidationError("No usable modality remains after missing-modality handling.")
    output = working.loc[:, selected].apply(pd.to_numeric, errors="coerce")
    if missing_policy == "impute_zero":
        output = output.fillna(0.0)
    if output.isna().any().any() or not np.isfinite(output.to_numpy(dtype=float)).all():
        raise ContractValidationError("Early-fusion input contains missing or non-finite values.")
    return output, {"fusion_type": "early", "used_modalities": used, "dropped_modalities": dropped, "zero_imputed_modalities": imputed}


def late_fuse_probabilities(probabilities: Mapping[str, Sequence[float] | np.ndarray], *, weights: Mapping[str, float] | None = None, required_modalities: Sequence[str] = (), missing_policy: str = "error") -> tuple[np.ndarray, dict[str, object]]:
    """Return a weighted probability average with modality provenance."""
    _policy(missing_policy)
    if not probabilities:
        raise ValueError("At least one modality probability vector is required.")
    arrays: dict[str, np.ndarray] = {}
    size: int | None = None
    for name, values in probabilities.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
            raise ValueError(f"Probability vector for {name!r} must be finite, non-empty and in [0, 1].")
        if size is not None and len(array) != size:
            raise ValueError("All modality probability vectors must have equal length.")
        arrays[name] = array
        size = len(array)
    missing_required = set(required_modalities) - set(arrays)
    if missing_required:
        raise ContractValidationError(f"Missing required probability modalities: {sorted(missing_required)}")
    weights = {name: 1.0 for name in arrays} if weights is None else dict(weights)
    if set(weights) - set(arrays):
        raise ValueError("Weights supplied for unavailable modalities.")
    active = {name: float(weights.get(name, 1.0)) for name in arrays}
    if any(weight < 0 for weight in active.values()) or not any(active.values()):
        raise ValueError("Fusion weights must be non-negative and sum to a positive value.")
    fused = np.average(np.vstack(list(arrays.values())), axis=0, weights=np.asarray(list(active.values())))
    return fused, {"fusion_type": "late", "used_modalities": list(arrays), "missing_policy": missing_policy, "weights": active}


def hybrid_fuse_probabilities(early_probability: Sequence[float] | np.ndarray, late_probabilities: Mapping[str, Sequence[float] | np.ndarray], *, early_weight: float = 0.5, late_weights: Mapping[str, float] | None = None, missing_policy: str = "error") -> tuple[np.ndarray, dict[str, object]]:
    if not 0 <= early_weight <= 1:
        raise ValueError("early_weight must be in [0, 1].")
    early = np.asarray(early_probability, dtype=float)
    if early.ndim != 1 or not len(early) or not np.isfinite(early).all() or ((early < 0) | (early > 1)).any():
        raise ValueError("Early-fusion probabilities must be finite, non-empty and in [0, 1].")
    late, provenance = late_fuse_probabilities(late_probabilities, weights=late_weights, missing_policy=missing_policy)
    if len(early) != len(late):
        raise ValueError("Early and late probability vectors must have equal length.")
    return early_weight * early + (1 - early_weight) * late, {"fusion_type": "hybrid", "early_weight": early_weight, "late": provenance}
