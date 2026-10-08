"""Metrics and review tables for the retrospective faller model."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score


def _arrays(y_true, probabilities):
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Labels and probabilities must be equally sized and non-empty.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    return y, p


def binary_metrics(y_true, probabilities, threshold):
    """Calculate discrimination, operating-point and Brier metrics."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1].")
    y, p = _arrays(y_true, probabilities)
    pred = p >= threshold
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    tp = int(((y == 1) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    both = pos > 0 and neg > 0
    return {
        "n": int(len(y)), "positive_rate": float(y.mean()), "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, p)) if both else None,
        "auprc": float(average_precision_score(y, p)) if both else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else None,
        "sensitivity": float(tp / pos) if pos else None,
        "specificity": float(tn / neg) if neg else None,
        "brier_score": float(brier_score_loss(y, p)), "tp": tp,
        "fp": int(((y == 0) & pred).sum()), "tn": tn,
        "fn": int(((y == 1) & ~pred).sum()),
    }


def choose_balanced_accuracy_threshold(y_true, probabilities):
    """Pick the lowest threshold that gives max balanced accuracy."""
    y, p = _arrays(y_true, probabilities)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold selection requires both label classes.")
    candidates = sorted({0.0, 0.5, 1.0, *[round(float(v), 6) for v in p]})
    scored = [(binary_metrics(y, p, t)["balanced_accuracy"], t) for t in candidates]
    best = max(s for s, _ in scored if s is not None)
    return min(t for s, t in scored if s == best)


def threshold_impact(y_true, probabilities, thresholds):
    y, p = _arrays(y_true, probabilities)
    rows = []
    for t in sorted(set(float(x) for x in thresholds)):
        m = binary_metrics(y, p, t)
        pred = p >= t
        tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
        rows.append({
            **m,
            "flagged_n": int(pred.sum()),
            "flagged_rate": float(pred.mean()),
            "positive_predictive_value": float(tp / (tp + fp)) if tp + fp else None,
            "negative_predictive_value": float(tn / (tn + fn)) if tn + fn else None,
        })
    return rows


def calibration_table(y_true, probabilities, n_bins=10):
    """Return fixed-width calibration bins, including empty bins."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    y, p = _arrays(y_true, probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(p, edges[1:], right=True), n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = bins == i
        cnt = int(mask.sum())
        rows.append({"bin": i, "lower": float(edges[i]), "upper": float(edges[i + 1]), "n": cnt,
                     "mean_predicted_probability": float(p[mask].mean()) if cnt else None,
                     "observed_positive_rate": float(y[mask].mean()) if cnt else None})
    return rows


def expected_calibration_error(calibration, total=None):
    rows = list(calibration)
    denom = total if total is not None else sum(int(r["n"]) for r in rows)
    if denom <= 0:
        raise ValueError("Calibration table must contain observations.")
    return float(sum(
        int(r["n"]) / denom * abs(float(r["mean_predicted_probability"]) - float(r["observed_positive_rate"]))
        for r in rows if r["mean_predicted_probability"] is not None and r["observed_positive_rate"] is not None
    ))


def calibration_summary(y_true, probabilities, n_bins=5):
    """Return fixed-width calibration diagnostics."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    y, p = _arrays(y_true, probabilities)
    table = calibration_table(y, p, n_bins=n_bins)
    populated = [r for r in table if r["n"]]
    gaps = [
        abs(float(r["mean_predicted_probability"]) - float(r["observed_positive_rate"]))
        for r in populated
    ]

    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    if len(np.unique(y)) == 2 and len(np.unique(logit)) > 1:
        reg = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, y)
        intercept = float(reg.intercept_[0])
        slope = float(reg.coef_[0, 0])
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
                "lower_bound": r["lower"],
                "upper_bound": r["upper"],
                "n": r["n"],
                "mean_predicted_probability": r["mean_predicted_probability"],
                "observed_positive_rate": r["observed_positive_rate"],
            }
            for r in populated
        ],
    }


def error_analysis(frame, y_true, probabilities, threshold, *, id_column=None):
    y, p = _arrays(y_true, probabilities)
    if len(frame) != len(y):
        raise ValueError("Frame and predictions must have equal row counts.")
    if id_column is not None and id_column not in frame.columns:
        raise ValueError(f"Missing identifier column: {id_column}")
    out = frame.copy().reset_index(drop=True)
    pred = (p >= threshold).astype(int)
    out["observed_label"] = y
    out["predicted_probability"] = p
    out["predicted_label"] = pred
    out["decision_threshold"] = float(threshold)
    out["error_type"] = np.select(
        [(y == 1) & (pred == 1), (y == 0) & (pred == 0), (y == 1), (y == 0)],
        ["true_positive", "true_negative", "false_negative", "false_positive"], default="unknown")
    return out


def error_cases(frame, *, id_column, label_column, probability_column, threshold):
    """Return false positives and false negatives sorted for review."""
    required = [id_column, label_column, probability_column]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing error-analysis columns: {missing}")
    out = error_analysis(
        frame.loc[:, required],
        frame[label_column],
        frame[probability_column],
        threshold,
        id_column=id_column,
    )
    out["distance_from_threshold"] = (out[probability_column] - threshold).abs()
    return out.loc[out["error_type"].isin(["false_positive", "false_negative"])].sort_values(
        ["error_type", "distance_from_threshold", id_column], ascending=[True, False, True]
    ).reset_index(drop=True)


def subgroup_template(frame, *, label_column, probability_column, threshold, subgroup_columns):
    """Report requested subgroups and mark fields absent from the frame."""
    rows = []
    for col in subgroup_columns:
        if col not in frame.columns:
            rows.append({"subgroup_column": col, "status": "unavailable_in_evaluation_frame"})
            continue
        for val, group in frame.groupby(col, dropna=False, sort=True):
            summary = binary_metrics(group[label_column], group[probability_column], threshold)
            rows.append({
                "subgroup_column": col,
                "subgroup_value": "missing" if pd.isna(val) else str(val),
                "status": "reported",
                **summary,
            })
    return pd.DataFrame(rows)


def logistic_feature_contributions(pipeline, features):
    """Return signed standardized feature contributions to fitted logistic log-odds."""
    if not hasattr(pipeline, "named_steps") or "imputer" not in pipeline.named_steps or "scaler" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        raise ValueError("Feature contributions require a fitted logistic imputer/scaler/model pipeline.")
    imputed = pipeline.named_steps["imputer"].transform(features)
    scaled = pipeline.named_steps["scaler"].transform(imputed)
    coef = pipeline.named_steps["model"].coef_.ravel()
    return pd.DataFrame(scaled * coef, columns=list(features.columns), index=features.index)
