"""Config-driven single-modality training wrapper for Member B workflows."""

from __future__ import annotations

from typing import Mapping

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import contract_from_config, validate_feature_frame
from .evaluation import binary_metrics, choose_balanced_accuracy_threshold
from .inference import FallerInferenceModel


def build_estimator(model_name: str, config: Mapping[str, object]) -> Pipeline:
    """Build an unfitted deterministic pipeline from the shared JSON config."""
    models = config.get("models")
    if not isinstance(models, Mapping) or model_name not in models:
        raise ValueError(f"Unsupported or unconfigured model: {model_name}")
    options = dict(models[model_name])
    seed = int(config["random_seed"])
    if model_name == "logistic_regression":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                         ("model", LogisticRegression(random_state=seed, **options))])
    if model_name == "random_forest":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model", RandomForestClassifier(random_state=seed, **options))])
    raise ValueError(f"Unsupported model: {model_name}")


def fit_single_modality(train: pd.DataFrame, config: Mapping[str, object], model_name: str) -> FallerInferenceModel:
    """Fit imputation and model strictly on the supplied training partition."""
    contract = contract_from_config(config)
    features = validate_feature_frame(train, contract, data_version=contract.data_version, require_label=True)
    labels = train[contract.label_column].astype(int)
    estimator = build_estimator(model_name, config)
    estimator.fit(features, labels)
    return FallerInferenceModel(estimator, contract, 0.5, model_name)


def select_threshold_on_validation(model: FallerInferenceModel, validation: pd.DataFrame) -> FallerInferenceModel:
    """Apply a validation-selected operating point without refitting preprocessing."""
    label = model.contract.label_column
    if not label:
        raise ValueError("Model contract must contain a label to select a threshold.")
    features = validate_feature_frame(validation, model.contract, data_version=model.contract.data_version, require_label=True)
    model.threshold = choose_balanced_accuracy_threshold(validation[label].astype(int), model.estimator.predict_proba(features)[:, 1])
    return model


def evaluate_model(model: FallerInferenceModel, frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Evaluate labels without routing them through the public inference API."""
    label = model.contract.label_column
    if not label:
        raise ValueError("Model contract must contain a label to evaluate.")
    features = validate_feature_frame(frame, model.contract, data_version=model.contract.data_version, require_label=True)
    return binary_metrics(frame[label].astype(int), model.estimator.predict_proba(features)[:, 1], model.threshold)
