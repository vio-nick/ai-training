"""Evaluation utilities for retrospective binary faller-recognition models.

The functions in this module consume held-out probabilities.  They never fit,
recalibrate, or otherwise change the model, so validation/test boundaries
remain auditable.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score


def _arrays(y_true: Iterable[int], probabilities: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Labels and probabilities must be equally sized and non-empty.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    return y, p


def binary_metrics(y_true: Iterable[int], probabilities: Iterable[float], threshold: float) -> dict[str, float | int | None]:
    """Calculate discrimination, operating-point and Brier metrics."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1].")
    y, p = _arrays(y_true, probabilities)
    predicted = p >= threshold
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    tp = int(((y == 1) & predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    both = positives > 0 and negatives > 0
    return {
        "n": int(len(y)), "positive_rate": float(y.mean()), "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, p)) if both else None,
        "auprc": float(average_precision_score(y, p)) if both else None,
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)) if both else None,
        "sensitivity": float(tp / positives) if positives else None,
        "specificity": float(tn / negatives) if negatives else None,
        "brier_score": float(brier_score_loss(y, p)), "tp": tp,
        "fp": int(((y == 0) & predicted).sum()), "tn": tn,
        "fn": int(((y == 1) & ~predicted).sum()),
    }


def choose_balanced_accuracy_threshold(y_true: Iterable[int], probabilities: Iterable[float]) -> float:
    """Select the lowest validation-only threshold at maximum balanced accuracy."""
    y, p = _arrays(y_true, probabilities)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both label classes.")
    candidates = sorted({0.0, 0.5, 1.0, *[round(float(value), 6) for value in p]})
    scored = [(binary_metrics(y, p, threshold)["balanced_accuracy"], threshold) for threshold in candidates]
    best = max(score for score, _ in scored if score is not None)
    return min(threshold for score, threshold in scored if score == best)


def threshold_impact(y_true: Iterable[int], probabilities: Iterable[float], thresholds: Iterable[float]) -> list[dict[str, float | int | None]]:
    """Describe confusion-matrix consequences for pre-specified thresholds."""

    y, p = _arrays(y_true, probabilities)
    rows: list[dict[str, float | int | None]] = []
    for threshold in sorted(set(float(item) for item in thresholds)):
        metric = binary_metrics(y, p, threshold)
        predicted = p >= threshold
        tp, fp, tn, fn = metric["tp"], metric["fp"], metric["tn"], metric["fn"]
        rows.append({
            **metric,
            "flagged_n": int(predicted.sum()),
            "flagged_rate": float(predicted.mean()),
            "positive_predictive_value": float(tp / (tp + fp)) if tp + fp else None,
            "negative_predictive_value": float(tn / (tn + fn)) if tn + fn else None,
        })
    return rows


def calibration_table(y_true: Iterable[int], probabilities: Iterable[float], n_bins: int = 10) -> list[dict[str, float | int | None]]:
    """Return fixed-width calibration bins, retaining empty bins explicitly."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    y, p = _arrays(y_true, probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(p, edges[1:], right=True), n_bins - 1)
    rows = []
    for index in range(n_bins):
        mask = bins == index
        count = int(mask.sum())
        rows.append({"bin": index, "lower": float(edges[index]), "upper": float(edges[index + 1]), "n": count,
                     "mean_predicted_probability": float(p[mask].mean()) if count else None,
                     "observed_positive_rate": float(y[mask].mean()) if count else None})
    return rows


def expected_calibration_error(calibration: Iterable[Mapping[str, object]], total: int | None = None) -> float:
    rows = list(calibration)
    denominator = total if total is not None else sum(int(row["n"]) for row in rows)
    if denominator <= 0:
        raise ValueError("Calibration table must contain observations.")
    return float(sum(int(row["n"]) / denominator * abs(float(row["mean_predicted_probability"]) - float(row["observed_positive_rate"]))
                     for row in rows if row["mean_predicted_probability"] is not None and row["observed_positive_rate"] is not None))


def calibration_summary(y_true: Iterable[int], probabilities: Iterable[float], n_bins: int = 5) -> dict[str, object]:
    """Calculate fixed-bin reliability diagnostics without changing predictions."""

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    y, p = _arrays(y_true, probabilities)
    table = calibration_table(y, p, n_bins=n_bins)
    populated = [row for row in table if row["n"]]
    gaps = [
        abs(float(row["mean_predicted_probability"]) - float(row["observed_positive_rate"]))
        for row in populated
    ]

    # This is a descriptive calibration-regression diagnostic only.  The
    # fitted regression is never used to recalibrate the reported predictions.
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    if len(np.unique(y)) == 2 and len(np.unique(logit)) > 1:
        regression = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, y)
        intercept: float | None = float(regression.intercept_[0])
        slope: float | None = float(regression.coef_[0, 0])
    else:
        intercept, slope = None, None

    return {
        "n": int(len(y)),
        "brier_score": float(brier_score_loss(y, p)),
        "expected_calibration_error": expected_calibration_error(table, total=len(y)),
        "maximum_calibration_error": float(max(gaps, default=0.0)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "binning": "equal_width_probability_bins",
        "bins": [
            {
                "lower_bound": row["lower"],
                "upper_bound": row["upper"],
                "n": row["n"],
                "mean_predicted_probability": row["mean_predicted_probability"],
                "observed_positive_rate": row["observed_positive_rate"],
            }
            for row in populated
        ],
    }


def error_analysis(frame: pd.DataFrame, y_true: Iterable[int], probabilities: Iterable[float], threshold: float, *, id_column: str | None = None) -> pd.DataFrame:
    y, p = _arrays(y_true, probabilities)
    if len(frame) != len(y):
        raise ValueError("Frame and predictions must have equal row counts.")
    if id_column is not None and id_column not in frame.columns:
        raise ValueError(f"Missing identifier column: {id_column}")
    output = frame.copy().reset_index(drop=True)
    predicted = (p >= threshold).astype(int)
    output["observed_label"] = y
    output["predicted_probability"] = p
    output["predicted_label"] = predicted
    output["decision_threshold"] = float(threshold)
    output["error_type"] = np.select(
        [(y == 1) & (predicted == 1), (y == 0) & (predicted == 0), (y == 1), (y == 0)],
        ["true_positive", "true_negative", "false_negative", "false_positive"], default="unknown")
    return output


def error_cases(frame: pd.DataFrame, *, id_column: str, label_column: str, probability_column: str, threshold: float) -> pd.DataFrame:
    """Return false positives/negatives in a deterministic review order."""

    required = [id_column, label_column, probability_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing error-analysis columns: {missing}")
    output = error_analysis(
        frame.loc[:, required],
        frame[label_column],
        frame[probability_column],
        threshold,
        id_column=id_column,
    )
    output["distance_from_threshold"] = (output[probability_column] - threshold).abs()
    return output.loc[output["error_type"].isin(["false_positive", "false_negative"])].sort_values(
        ["error_type", "distance_from_threshold", id_column], ascending=[True, False, True]
    ).reset_index(drop=True)


def subgroup_metrics(frame: pd.DataFrame, subgroup_column: str, y_true: Iterable[int], probabilities: Iterable[float], threshold: float) -> list[dict[str, object]]:
    if subgroup_column not in frame.columns:
        raise ValueError(f"Missing subgroup column: {subgroup_column}")
    y, p = _arrays(y_true, probabilities)
    if len(frame) != len(y):
        raise ValueError("Frame and predictions must have equal row counts.")
    result = []
    for value, index in frame.groupby(subgroup_column, dropna=False, sort=True).groups.items():
        indices = np.asarray(list(index), dtype=int)
        result.append({"subgroup": str(value), **binary_metrics(y[indices], p[indices], threshold)})
    return result


def subgroup_template(frame: pd.DataFrame, *, label_column: str, probability_column: str, threshold: float, subgroup_columns: Iterable[str]) -> pd.DataFrame:
    """Report supplied subgroup fields and record unavailable fields transparently."""

    rows: list[dict[str, object]] = []
    for column in subgroup_columns:
        if column not in frame.columns:
            rows.append({"subgroup_column": column, "status": "unavailable_in_evaluation_frame"})
            continue
        for value, group in frame.groupby(column, dropna=False, sort=True):
            summary = binary_metrics(group[label_column], group[probability_column], threshold)
            rows.append({
                "subgroup_column": column,
                "subgroup_value": "missing" if pd.isna(value) else str(value),
                "status": "reported",
                **summary,
            })
    return pd.DataFrame(rows)


def logistic_feature_contributions(pipeline: object, features: pd.DataFrame) -> pd.DataFrame:
    """Return signed standardized feature contributions to fitted logistic log-odds."""

    if not hasattr(pipeline, "named_steps") or "imputer" not in pipeline.named_steps or "scaler" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError("Feature contributions require a fitted logistic imputer/scaler/model pipeline.")
    imputed = pipeline.named_steps["imputer"].transform(features)
    standardized = pipeline.named_steps["scaler"].transform(imputed)
    coefficients = pipeline.named_steps["model"].coef_.ravel()
    return pd.DataFrame(standardized * coefficients, columns=list(features.columns), index=features.index)
