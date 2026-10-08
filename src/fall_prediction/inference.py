"""Inference wrapper for a released GSTRIDE estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import InputContract, validate_feature_frame, validate_feature_record
from .explainability import build_feature_contribution_records, exact_shapley_probability_contributions

RISK_LEVEL_LOW_MAX_EXCLUSIVE = 0.3
RISK_LEVEL_HIGH_MIN_EXCLUSIVE = 0.7
RISK_LEVEL_DISPLAY_NAMES = {
    "low": "低风险",
    "medium": "中风险",
    "high": "高风险",
}


def classify_risk_level(probabilities):
    """Map model scores to the fixed display bands used by the prototype."""
    vals = np.asarray(probabilities, dtype=float)
    if not np.isfinite(vals).all() or ((vals < 0) | (vals > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    return np.select(
        [vals < RISK_LEVEL_LOW_MAX_EXCLUSIVE, vals <= RISK_LEVEL_HIGH_MIN_EXCLUSIVE],
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

    def predict_frame(self, frame, *, data_version=None):
        """Score feature-only rows; labels are deliberately never read."""
        features = validate_feature_frame(frame, self.contract, data_version=data_version, allow_missing_values=True)
        probs = np.asarray(self.estimator.predict_proba(features), dtype=float)[:, 1]
        if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
            raise ValueError("Estimator returned invalid probabilities.")
        risk_levels = classify_risk_level(probs)
        out = pd.DataFrame({
            "predicted_probability": probs,
            "predicted_probability_percent": probs * 100,
            "risk_level": risk_levels,
            "risk_level_display": [RISK_LEVEL_DISPLAY_NAMES[lvl] for lvl in risk_levels],
            "predicted_label": (probs >= self.threshold).astype(int),
            "decision_threshold": self.threshold,
            "data_version": self.contract.data_version,
            "model_name": self.model_name,
        })
        contribs, baselines, recon = exact_shapley_probability_contributions(self.estimator, features)
        out["feature_contributions"] = build_feature_contribution_records(
            list(features.columns),
            features.to_numpy(dtype=float, na_value=np.nan),
            contribs,
        )
        out["feature_contribution_method"] = "exact_shapley_probability"
        out["feature_contribution_baseline_probability"] = baselines
        out["feature_contribution_reconstructed_probability"] = recon
        if self.contract.id_column and self.contract.id_column in frame.columns:
            out.insert(0, self.contract.id_column, frame[self.contract.id_column].to_numpy())
        return out

    def predict_record(self, record, *, data_version=None):
        """Score one JSON-compatible feature mapping."""
        validate_feature_record(record, self.contract, data_version=data_version)
        return self.predict_frame(pd.DataFrame([dict(record)]), data_version=data_version).iloc[0].to_dict()


def score_records(model, records, *, data_version=None):
    """Batch convenience API — validates all records before scoring."""
    if not records:
        raise ValueError("At least one inference record is required.")
    return model.predict_frame(
        pd.DataFrame([dict(r) for r in records]), data_version=data_version
    ).to_dict(orient="records")
