"""Versioned input contracts for the GSTRIDE faller prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .gstride import FEATURE_COLUMNS

GSTRIDE_FALL_DATA_VERSION = "gstride_fall_v1"
GSTRIDE_LABEL_COLUMN = "faller_last_year"
GSTRIDE_ID_COLUMN = "participant_id"


@dataclass(frozen=True)
class InputContract:
    data_version: str
    feature_columns: tuple[str, ...]
    label_column: str | None = None
    id_column: str | None = None

    @classmethod
    def gstride_fall_v1(cls, include_label: bool = False) -> "InputContract":
        return cls(
            GSTRIDE_FALL_DATA_VERSION,
            tuple(FEATURE_COLUMNS),
            GSTRIDE_LABEL_COLUMN if include_label else None,
            GSTRIDE_ID_COLUMN,
        )


class ContractValidationError(ValueError):
    """Raised when model input violates its versioned contract."""


def require_compatible_version(actual, expected):
    if actual != expected:
        raise ContractValidationError(f"Incompatible data version {actual!r}; model requires {expected!r}.")


def validate_feature_frame(
    frame,
    contract,
    *,
    data_version=None,
    allow_missing_values=True,
    require_label=False,
    require_unique_id=False,
):
    """Validate and return ordered numeric features without mutating ``frame``."""
    if not isinstance(frame, pd.DataFrame):
        raise ContractValidationError("Model input must be a pandas DataFrame.")
    if data_version is not None:
        require_compatible_version(data_version, contract.data_version)
    required = list(contract.feature_columns)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ContractValidationError(f"Missing required feature column(s): {missing}")
    if require_label:
        if not contract.label_column or contract.label_column not in frame.columns:
            raise ContractValidationError(f"Missing required label column: {contract.label_column}")
        label = pd.to_numeric(frame[contract.label_column], errors="coerce")
        if label.isna().any() or not label.isin([0, 1]).all():
            raise ContractValidationError(f"{contract.label_column} must contain only binary 0/1 labels.")
    if require_unique_id and contract.id_column:
        if contract.id_column not in frame.columns:
            raise ContractValidationError(f"Missing required ID column: {contract.id_column}")
        ids = frame[contract.id_column]
        if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.duplicated().any():
            raise ContractValidationError(f"{contract.id_column} must be non-empty and unique.")
    raw = frame.loc[:, required]
    values = raw.apply(pd.to_numeric, errors="coerce")
    invalid = values.isna() & ~raw.isna()
    if invalid.any().any():
        bad_cols = invalid.any(axis=0)[lambda s: s].index.tolist()
        raise ContractValidationError(f"Non-numeric feature value(s) in: {bad_cols}")
    numeric = values.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(numeric).any():
        raise ContractValidationError("Feature values must be finite when present.")
    if not allow_missing_values and values.isna().any().any():
        raise ContractValidationError("Missing feature values are not allowed for this input.")
    return values


def validate_feature_record(record, contract, *, data_version=None):
    if not isinstance(record, Mapping):
        raise ContractValidationError("Inference input must be a mapping.")
    return validate_feature_frame(pd.DataFrame([dict(record)]), contract, data_version=data_version)


def contract_from_config(config):
    try:
        return InputContract(
            data_version=str(config["data_version"]),
            feature_columns=tuple(str(v) for v in config["feature_columns"]),
            label_column=str(config["label_column"]),
            id_column=str(config["id_column"]),
        )
    except KeyError as exc:
        raise ContractValidationError(f"Training configuration is missing {exc.args[0]!r}.") from exc
