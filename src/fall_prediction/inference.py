"""Small inference adapter coupling a fitted model with its input contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import InputContract, validate_feature_frame, validate_feature_record


@dataclass
class FallerInferenceModel:
    """A sklearn-compatible probability model with a versioned interface."""

    estimator: object
    contract: InputContract
    threshold: float
    model_name: str = "model"

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError("Threshold must be in [0, 1].")
        if not hasattr(self.estimator, "predict_proba"):
            raise TypeError("Estimator must expose predict_proba.")

    def predict_frame(self, frame: pd.DataFrame, *, data_version: str | None = None) -> pd.DataFrame:
        """Score feature-only rows; labels are deliberately never read."""
        features = validate_feature_frame(frame, self.contract, data_version=data_version, allow_missing_values=True)
        probabilities = np.asarray(self.estimator.predict_proba(features), dtype=float)[:, 1]
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Estimator returned invalid probabilities.")
        output = pd.DataFrame({
            "predicted_probability": probabilities,
            "predicted_label": (probabilities >= self.threshold).astype(int),
            "decision_threshold": self.threshold,
            "data_version": self.contract.data_version,
            "model_name": self.model_name,
        })
        if self.contract.id_column and self.contract.id_column in frame.columns:
            output.insert(0, self.contract.id_column, frame[self.contract.id_column].to_numpy())
        return output

    def predict_record(self, record: Mapping[str, object], *, data_version: str | None = None) -> dict[str, object]:
        """Score one JSON-compatible feature mapping."""
        validate_feature_record(record, self.contract, data_version=data_version)
        return self.predict_frame(pd.DataFrame([dict(record)]), data_version=data_version).iloc[0].to_dict()


def score_records(model: FallerInferenceModel, records: Sequence[Mapping[str, object]], *, data_version: str | None = None) -> list[dict[str, object]]:
    """Batch convenience API which validates all requested records before scoring."""
    if not records:
        raise ValueError("At least one inference record is required.")
    return model.predict_frame(pd.DataFrame([dict(record) for record in records]), data_version=data_version).to_dict(orient="records")
