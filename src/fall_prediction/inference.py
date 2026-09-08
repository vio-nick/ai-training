"""Small inference adapter coupling a fitted model with its input contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import InputContract, validate_feature_frame, validate_feature_record

RISK_LEVEL_LOW_MAX_EXCLUSIVE = 0.3
RISK_LEVEL_HIGH_MIN_EXCLUSIVE = 0.7
RISK_LEVEL_DISPLAY_NAMES = {
    "low": "低风险",
    "medium": "中风险",
    "high": "高风险",
}


def classify_risk_level(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map model scores to the fixed display bands used by the prototype.

    The boundaries are intentionally inclusive for the middle band:
    ``low`` is below 0.3, ``medium`` is from 0.3 through 0.7, and ``high``
    is above 0.7. These are presentation bands for the retrospective model
    score, not clinically validated future-fall risk categories.
    """

    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    return np.select(
        [values < RISK_LEVEL_LOW_MAX_EXCLUSIVE, values <= RISK_LEVEL_HIGH_MIN_EXCLUSIVE],
        ["low", "medium"],
        default="high",
    )


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
        risk_levels = classify_risk_level(probabilities)
        output = pd.DataFrame({
            "predicted_probability": probabilities,
            "predicted_probability_percent": probabilities * 100,
            "risk_level": risk_levels,
            "risk_level_display": [RISK_LEVEL_DISPLAY_NAMES[level] for level in risk_levels],
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
