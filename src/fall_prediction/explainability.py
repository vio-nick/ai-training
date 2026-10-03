"""Local, model-agnostic explanations for the fixed six-feature contract."""

from __future__ import annotations

from math import factorial
from typing import Any

import numpy as np
import pandas as pd

from .gstride import FEATURE_DISPLAY_METADATA


def _positive_class_probability(estimator: object, features: pd.DataFrame) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("Estimator must expose predict_proba for feature explanations.")
    probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Estimator must return probabilities for both classes.")
    result = probabilities[:, 1]
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError("Estimator returned invalid probabilities for feature explanations.")
    return result


def exact_shapley_probability_contributions(
    estimator: object,
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose positive-class probability into exact Shapley contributions.

    The baseline is the prediction after replacing every feature with a
    missing value. The fitted pipeline's imputer then supplies the training
    baseline (the median in the released GSTRIDE models). With six features,
    the exact 2**6 coalition predictions are inexpensive and avoid an
    optional SHAP dependency.
    """

    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("Features must be a non-empty pandas DataFrame.")
    feature_names = list(features.columns)
    feature_count = len(feature_names)
    if feature_count == 0:
        raise ValueError("At least one feature is required for explanations.")
    values = features.to_numpy(dtype=float, na_value=np.nan)
    row_count = len(features)

    coalition_frames: list[pd.DataFrame] = []
    for mask in range(1 << feature_count):
        coalition = np.full((row_count, feature_count), np.nan, dtype=float)
        for feature_index in range(feature_count):
            if mask & (1 << feature_index):
                coalition[:, feature_index] = values[:, feature_index]
        coalition_frames.append(pd.DataFrame(coalition, columns=feature_names))

    # All coalitions share one model call. This keeps exact decomposition
    # practical for interactive single-record inference.
    stacked = pd.concat(coalition_frames, ignore_index=True)
    stacked_probabilities = _positive_class_probability(estimator, stacked)
    coalition_probabilities = {
        mask: stacked_probabilities[mask * row_count : (mask + 1) * row_count]
        for mask in range(1 << feature_count)
    }

    contributions = np.zeros((row_count, feature_count), dtype=float)
    denominator = factorial(feature_count)
    for feature_index in range(feature_count):
        without_feature_masks = [
            mask for mask in range(1 << feature_count)
            if not mask & (1 << feature_index)
        ]
        for mask in without_feature_masks:
            coalition_size = mask.bit_count()
            weight = (
                factorial(coalition_size)
                * factorial(feature_count - coalition_size - 1)
                / denominator
            )
            contributions[:, feature_index] += weight * (
                coalition_probabilities[mask | (1 << feature_index)]
                - coalition_probabilities[mask]
            )

    baseline = coalition_probabilities[0]
    reconstructed = baseline + contributions.sum(axis=1)
    return contributions, baseline, reconstructed


def build_feature_contribution_records(
    feature_names: list[str],
    feature_values: np.ndarray,
    contributions: np.ndarray,
) -> list[list[dict[str, Any]]]:
    """Build JSON-compatible, frontend-oriented records in contract order."""

    if contributions.shape != feature_values.shape:
        raise ValueError("Feature values and contributions must have the same shape.")
    records: list[list[dict[str, Any]]] = []
    for row_values, row_contributions in zip(feature_values, contributions):
        row: list[dict[str, Any]] = []
        for feature, value, contribution in zip(feature_names, row_values, row_contributions):
            metadata = FEATURE_DISPLAY_METADATA.get(feature, {"display_name": feature, "unit": ""})
            numeric_contribution = float(contribution)
            if numeric_contribution > 0:
                direction, direction_display = "positive", "正贡献"
            elif numeric_contribution < 0:
                direction, direction_display = "negative", "负贡献"
            else:
                direction, direction_display = "neutral", "无贡献"
            row.append(
                {
                    "feature": feature,
                    "display_name": metadata["display_name"],
                    "unit": metadata["unit"],
                    "value": None if np.isnan(value) else float(value),
                    "contribution_probability": numeric_contribution,
                    "contribution_probability_percent": numeric_contribution * 100,
                    "direction": direction,
                    "direction_display": direction_display,
                }
            )
        records.append(row)
    return records
