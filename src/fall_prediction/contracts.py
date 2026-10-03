"""Versioned input contracts for the GSTRIDE faller prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

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


def require_compatible_version(actual: str, expected: str) -> None:
    if actual != expected:
        raise ContractValidationError(f"Incompatible data version {actual!r}; model requires {expected!r}.")


def validate_feature_frame(
    frame: pd.DataFrame,
    contract: InputContract,
    *,
    data_version: str | None = None,
    allow_missing_values: bool = True,
    require_label: bool = False,
    require_unique_id: bool = False,
) -> pd.DataFrame:
    """Validate and return ordered numeric features without mutating ``frame``."""

    if not isinstance(frame, pd.DataFrame):
        raise ContractValidationError("Model input must be a pandas DataFrame.")
    if data_version is not None:
        require_compatible_version(data_version, contract.data_version)
    required = list(contract.feature_columns)
    missing = [column for column in required if column not in frame.columns]
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
        raise ContractValidationError(f"Non-numeric feature value(s) in: {invalid.any(axis=0)[lambda s: s].index.tolist()}")
    numeric = values.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(numeric).any():
        raise ContractValidationError("Feature values must be finite when present.")
    if not allow_missing_values and values.isna().any().any():
        raise ContractValidationError("Missing feature values are not allowed for this input.")
    return values


def validate_feature_record(record: Mapping[str, object], contract: InputContract, *, data_version: str | None = None) -> pd.DataFrame:
    if not isinstance(record, Mapping):
        raise ContractValidationError("Inference input must be a mapping.")
    return validate_feature_frame(pd.DataFrame([dict(record)]), contract, data_version=data_version)


def contract_from_config(config: Mapping[str, object]) -> InputContract:
    try:
        return InputContract(
            data_version=str(config["data_version"]),
            feature_columns=tuple(str(v) for v in config["feature_columns"]),
            label_column=str(config["label_column"]),
            id_column=str(config["id_column"]),
        )
    except KeyError as exc:
        raise ContractValidationError(f"Training configuration is missing {exc.args[0]!r}.") from exc


def ensure_columns_exist(columns: Sequence[str], frame: pd.DataFrame, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ContractValidationError(f"Modality {name!r} is missing declared column(s): {missing}")
