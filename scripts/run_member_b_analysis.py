"""Generate held-out evaluation analyses for the GSTRIDE sensor prototype."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fall_prediction.evaluation import calibration_summary, error_cases, logistic_feature_contributions, subgroup_template, threshold_impact
from run_gstride_fall_baseline import choose_threshold, make_pipeline


def main() -> None:
    config = json.loads((ROOT / "configs/gstride_fall_v1.json").read_text(encoding="utf-8"))
    table = pd.read_csv(ROOT / config["feature_table_path"])
    split = pd.read_csv(ROOT / config["split_path"])
    frame = table.merge(split, on=config["id_column"], validate="one_to_one")
    train = frame.loc[frame["split"] == "train"].copy()
    validation = frame.loc[frame["split"] == "validation"].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    features, label, identifier = config["feature_columns"], config["label_column"], config["id_column"]
    out_dir = ROOT / "reports" / "member_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "data_version": config["data_version"],
        "evaluation_scope": "held-out GSTRIDE test set; retrospective faller recognition only",
        "subgroup_caveat": "No demographic or clinical subgroup fields are present in the six-feature contract; unavailable fields are recorded as templates.",
        "models": {},
    }
    for name in config["models"]:
        pipeline = make_pipeline(name, config)
        pipeline.fit(train[features], train[label])
        validation_probability = pipeline.predict_proba(validation[features])[:, 1]
        threshold = choose_threshold(validation[label], validation_probability)
        probability = pipeline.predict_proba(test[features])[:, 1]
        predicted = (probability >= threshold).astype(int)
        analysis_frame = test[[identifier, label]].copy()
        analysis_frame["probability"] = probability
        analysis_frame["predicted_label"] = predicted
        errors = error_cases(analysis_frame, id_column=identifier, label_column=label, probability_column="probability", threshold=threshold)
        errors.to_csv(out_dir / f"gstride_fall_v1_{name}_error_cases.csv", index=False, encoding="utf-8")
        groups = subgroup_template(analysis_frame, label_column=label, probability_column="probability", threshold=threshold, subgroup_columns=["sex", "age_group", "device_type"])
        groups.to_csv(out_dir / f"gstride_fall_v1_{name}_subgroups.csv", index=False, encoding="utf-8")
        model_result: dict[str, object] = {
            "selected_on": "validation", "threshold": threshold,
            "calibration": calibration_summary(test[label], probability, n_bins=5),
            "threshold_impact": threshold_impact(test[label], probability, [0.25, 0.5, 0.75, threshold]),
            "error_case_count": int(len(errors)),
            "subgroup_output": str((out_dir / f"gstride_fall_v1_{name}_subgroups.csv").relative_to(ROOT)).replace("\\", "/"),
        }
        if name == "logistic_regression":
            contributions = logistic_feature_contributions(pipeline, test[features])
            contributions.insert(0, identifier, test[identifier].to_numpy())
            contributions.insert(1, label, test[label].to_numpy())
            contributions.insert(2, "probability", probability)
            contributions.to_csv(out_dir / "gstride_fall_v1_logistic_regression_feature_contributions.csv", index=False, encoding="utf-8")
            model_result["explainability"] = {
                "method": "signed standardized feature contribution to the fitted model log-odds",
                "output": "reports/member_b/gstride_fall_v1_logistic_regression_feature_contributions.csv",
            }
        else:
            importances = pipeline.named_steps["model"].feature_importances_
            pd.DataFrame({"feature": features, "global_impurity_importance": importances}).sort_values("global_impurity_importance", ascending=False).to_csv(
                out_dir / "gstride_fall_v1_random_forest_feature_importance.csv", index=False, encoding="utf-8"
            )
            model_result["explainability"] = {
                "method": "random-forest global impurity importance; not a causal or per-participant explanation",
                "output": "reports/member_b/gstride_fall_v1_random_forest_feature_importance.csv",
            }
        summary["models"][name] = model_result
    (out_dir / "gstride_fall_v1_evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
