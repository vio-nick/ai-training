"""Run Member B's configuration-driven GSTRIDE training, evaluation and inference workflow.

The only formal results generated here use the six GSTRIDE gait features and
the retrospective faller_last_year label.  The fusion framework is deliberately
not trained because no compatible BMD or hospital longitudinal modality exists.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fall_prediction.contracts import contract_from_config, validate_feature_frame
from fall_prediction.evaluation import (
    calibration_summary,
    error_cases,
    logistic_feature_contributions,
    subgroup_template,
    threshold_impact,
)
from fall_prediction.training import evaluate_model, fit_single_modality, select_threshold_on_validation

WORKFLOW_CONFIG_PATH = ROOT / "configs" / "member_b_gstride_fall_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_partitioned_frame(base_config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(ROOT / str(base_config["feature_table_path"]))
    split = pd.read_csv(ROOT / str(base_config["split_path"]))
    identifier = str(base_config["id_column"])
    frame = table.merge(split, on=identifier, how="inner", validate="one_to_one")
    if len(frame) != len(table) or set(frame["split"]) != {"train", "validation", "test"}:
        raise ValueError("Feature table and participant split must form a complete train/validation/test contract.")
    contract = contract_from_config(base_config)
    validate_feature_frame(frame, contract, data_version=contract.data_version, require_label=True, require_unique_id=True)
    return tuple(frame.loc[frame["split"] == name].copy() for name in ("train", "validation", "test"))  # type: ignore[return-value]


def main() -> None:
    workflow = json.loads(WORKFLOW_CONFIG_PATH.read_text(encoding="utf-8"))
    base_path = ROOT / str(workflow["base_training_config_path"])
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if workflow["data_version"] != base["data_version"]:
        raise ValueError("Member B workflow and base training configuration use different data versions.")

    train, validation, test = load_partitioned_frame(base)
    identifier, label, features = str(base["id_column"]), str(base["label_column"]), list(base["feature_columns"])
    report_dir = ROOT / str(workflow["report_directory"])
    report_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "workflow_version": workflow["workflow_version"],
        "data_version": base["data_version"],
        "interface_contract": "gstride_fall_v1",
        "task_definition": "Retrospective recognition of self-reported fall during the year before the gait test.",
        "not_a_claim": "Not prospective fall prediction, fracture-risk prediction, osteoporosis diagnosis, or clinical screening.",
        "config_sha256": {"workflow": sha256(WORKFLOW_CONFIG_PATH), "base_training": sha256(base_path)},
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "formal_modality": "gait_sensor_only",
        "risk_display_policy": {
            "score_semantics": "Retrospective model score for self-reported fall during the year before the gait test; not prospective fall risk.",
            "probability_percent_formula": "predicted_probability * 100",
            "classification_input": "unrounded predicted_probability",
            "bands": [
                {"risk_level": "low", "risk_level_display": "低风险", "condition": "p < 0.3", "percentage_range": "< 30%"},
                {"risk_level": "medium", "risk_level_display": "中风险", "condition": "0.3 <= p <= 0.7", "percentage_range": "30%--70% (inclusive)"},
                {"risk_level": "high", "risk_level_display": "高风险", "condition": "p > 0.7", "percentage_range": "> 70%"},
            ],
            "decision_threshold_note": "The model-specific decision_threshold creates predicted_label and is independent of the three display bands.",
        },
        "fusion": {"status": workflow["fusion_result_status"], "reason": "No version-compatible BMD/hospital longitudinal modality is available."},
        "models": {},
    }

    for model_name in base["models"]:
        model = fit_single_modality(train, base, model_name)
        select_threshold_on_validation(model, validation)
        prediction = model.predict_frame(test, data_version=str(base["data_version"]))
        probability = prediction["predicted_probability"].to_numpy()
        prediction.insert(1, label, test[label].to_numpy())
        prediction.to_csv(report_dir / f"gstride_fall_v1_{model_name}_framework_test_predictions.csv", index=False, encoding="utf-8")

        errors = error_cases(
            prediction,
            id_column=identifier,
            label_column=label,
            probability_column="predicted_probability",
            threshold=model.threshold,
        )
        errors.to_csv(report_dir / f"gstride_fall_v1_{model_name}_framework_error_cases.csv", index=False, encoding="utf-8")
        subgroups = subgroup_template(
            prediction,
            label_column=label,
            probability_column="predicted_probability",
            threshold=model.threshold,
            subgroup_columns=list(workflow["subgroup_template_columns"]),
        )
        subgroups.to_csv(report_dir / f"gstride_fall_v1_{model_name}_framework_subgroups.csv", index=False, encoding="utf-8")

        explanation: dict[str, str]
        if model_name == "logistic_regression":
            contributions = logistic_feature_contributions(model.estimator, test[features])
            contributions.insert(0, identifier, test[identifier].to_numpy())
            contributions.insert(1, label, test[label].to_numpy())
            contributions.insert(2, "predicted_probability", probability)
            path = report_dir / "gstride_fall_v1_logistic_regression_framework_feature_contributions.csv"
            contributions.to_csv(path, index=False, encoding="utf-8")
            explanation = {"method": "Signed standardized contribution to fitted logistic-regression log-odds; association only.", "output": str(path.relative_to(ROOT)).replace("\\", "/")}
        else:
            importances = model.estimator.named_steps["model"].feature_importances_
            path = report_dir / "gstride_fall_v1_random_forest_framework_feature_importance.csv"
            pd.DataFrame({"feature": features, "global_impurity_importance": importances}).sort_values("global_impurity_importance", ascending=False).to_csv(path, index=False, encoding="utf-8")
            explanation = {"method": "Global random-forest impurity importance; not causal or per-participant explanation.", "output": str(path.relative_to(ROOT)).replace("\\", "/")}

        result["models"][model_name] = {
            "threshold_selected_on": "validation_max_balanced_accuracy",
            "threshold": model.threshold,
            "test_metrics": evaluate_model(model, test),
            "calibration_on_held_out_test": calibration_summary(test[label], probability, n_bins=int(workflow["calibration_bins"])),
            "threshold_impact_on_held_out_test": threshold_impact(test[label], probability, [*workflow["threshold_impact_values"], model.threshold]),
            "error_case_count": len(errors),
            "subgroup_status": "Template records unavailable fields because the six-feature contract contains no demographic/clinical fields.",
            "explainability": explanation,
        }

        if model_name == workflow["inference_example_model"]:
            example = test.loc[:, [identifier, *features]].iloc[0].to_dict()
            result["inference_example"] = {
                "input": example,
                "output": model.predict_record(example, data_version=str(base["data_version"])),
                "assurance": "The public inference method selects feature columns only and does not consume the label.",
            }

    output = report_dir / "gstride_fall_v1_member_b_workflow.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
