"""Evaluation utilities for retrospective binary faller-recognition models.

These helpers deliberately do not fit or recalibrate a model.  They consume
held-out probabilities so that selection and evaluation remain auditable.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def _arrays(y_true: Iterable[int], probabilities: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Labels and probabilities must be non-empty and have the same length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    return y, p


def calibration_summary(y_true: Iterable[int], probabilities: Iterable[float], n_bins: int = 5) -> dict[str, object]:
    """Calculate a fixed-bin reliability table and calibration diagnostics."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    y, p = _arrays(y_true, probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    bins: list[dict[str, float | int]] = []
    ece = 0.0
    mce = 0.0
    for index in range(n_bins):
        mask = indices == index
        count = int(mask.sum())
        if not count:
            continue
        observed = float(y[mask].mean())
        predicted = float(p[mask].mean())
        gap = abs(observed - predicted)
        ece += count / len(y) * gap
        mce = max(mce, gap)
        bins.append({
            "lower_bound": float(edges[index]),
            "upper_bound": float(edges[index + 1]),
            "n": count,
            "mean_predicted_probability": predicted,
            "observed_positive_rate": observed,
        })

    # Logistic recalibration summary. The clipping only makes the logit finite;
    # no probability is modified for the reliability table or Brier score.
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    if len(np.unique(y)) == 2 and len(np.unique(logit)) > 1:
        model = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, y)
        intercept = float(model.intercept_[0])
        slope = float(model.coef_[0, 0])
    else:
        intercept, slope = None, None
    return {
        "n": int(len(y)),
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "binning": "equal_width_probability_bins",
        "bins": bins,
    }


def threshold_impact(y_true: Iterable[int], probabilities: Iterable[float], thresholds: Iterable[float]) -> list[dict[str, float | int]]:
    """Describe confusion-matrix consequences for pre-specified thresholds."""
    y, p = _arrays(y_true, probabilities)
    rows: list[dict[str, float | int]] = []
    for threshold in sorted(set(float(item) for item in thresholds)):
        if not 0 <= threshold <= 1:
            raise ValueError("Thresholds must be in [0, 1]")
        predicted = p >= threshold
        tp = int(((y == 1) & predicted).sum())
        fp = int(((y == 0) & predicted).sum())
        tn = int(((y == 0) & ~predicted).sum())
        fn = int(((y == 1) & ~predicted).sum())
        rows.append({
            "threshold": threshold,
            "n": int(len(y)),
            "flagged_n": int(predicted.sum()),
            "flagged_rate": float(predicted.mean()),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "positive_predictive_value": float(tp / (tp + fp)) if tp + fp else None,
            "negative_predictive_value": float(tn / (tn + fn)) if tn + fn else None,
        })
    return rows


def error_cases(frame: pd.DataFrame, *, id_column: str, label_column: str, probability_column: str, threshold: float) -> pd.DataFrame:
    """Return held-out false positives/negatives with a deterministic review order."""
    required = [id_column, label_column, probability_column]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing error-analysis columns: {missing}")
    output = frame.loc[:, required].copy()
    output["predicted_label"] = (output[probability_column] >= threshold).astype(int)
    output["error_type"] = np.select(
        [(output[label_column] == 0) & (output["predicted_label"] == 1), (output[label_column] == 1) & (output["predicted_label"] == 0)],
        ["false_positive", "false_negative"], default="correct",
    )
    output["distance_from_threshold"] = (output[probability_column] - threshold).abs()
    return output.loc[output["error_type"] != "correct"].sort_values(
        ["error_type", "distance_from_threshold", id_column], ascending=[True, False, True]
    ).reset_index(drop=True)


def subgroup_template(frame: pd.DataFrame, *, label_column: str, probability_column: str, threshold: float, subgroup_columns: Iterable[str]) -> pd.DataFrame:
    """Compute a transparent subgroup table only for supplied, documented fields."""
    rows: list[dict[str, object]] = []
    for column in subgroup_columns:
        if column not in frame:
            rows.append({"subgroup_column": column, "status": "unavailable_in_evaluation_frame"})
            continue
        for value, group in frame.groupby(column, dropna=False, sort=True):
            y, p = _arrays(group[label_column], group[probability_column])
            predicted = p >= threshold
            rows.append({
                "subgroup_column": column, "subgroup_value": "missing" if pd.isna(value) else str(value), "status": "reported",
                "n": int(len(group)), "positive_rate": float(y.mean()), "brier_score": float(brier_score_loss(y, p)),
                "sensitivity": float(((y == 1) & predicted).sum() / (y == 1).sum()) if (y == 1).sum() else None,
                "specificity": float(((y == 0) & ~predicted).sum() / (y == 0).sum()) if (y == 0).sum() else None,
            })
    return pd.DataFrame(rows)


def logistic_feature_contributions(pipeline: object, features: pd.DataFrame) -> pd.DataFrame:
    """Give signed per-row log-odds contributions for a fitted logistic pipeline."""
    if not hasattr(pipeline, "named_steps") or "scaler" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError("Feature contributions require the fitted logistic imputer/scaler/model pipeline")
    imputed = pipeline.named_steps["imputer"].transform(features)
    standardized = pipeline.named_steps["scaler"].transform(imputed)
    coefficients = pipeline.named_steps["model"].coef_.ravel()
    return pd.DataFrame(standardized * coefficients, columns=list(features.columns), index=features.index)
