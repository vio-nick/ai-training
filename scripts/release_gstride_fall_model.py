"""Train once from the approved historical GSTRIDE table and save a model release."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fall_prediction.release import save_model_release, sha256_file
from fall_prediction.training import fit_single_modality, select_threshold_on_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one fixed GSTRIDE model release for later inference."
    )
    parser.add_argument("--model-name", choices=("random_forest", "logistic_regression"), default="random_forest")
    parser.add_argument(
        "--fit-scope",
        choices=("full_feature_table", "train"),
        default="full_feature_table",
        help="Historical rows used to fit the released estimator. New inference records are never read here.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "model_releases" / "gstride_fall_v1_random_forest_full_historical",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace model.joblib and manifest.json in the output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = ROOT / "configs" / "gstride_fall_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    feature_path = ROOT / str(config["feature_table_path"])
    split_path = ROOT / str(config["split_path"])
    table = pd.read_csv(feature_path)
    split = pd.read_csv(split_path)
    identifier = str(config["id_column"])
    frame = table.merge(split, on=identifier, how="inner", validate="one_to_one")
    if len(frame) != len(table):
        raise ValueError("Feature table and split table must contain exactly the same participant IDs.")
    train = frame.loc[frame["split"] == "train"].copy()
    validation = frame.loc[frame["split"] == "validation"].copy()
    if train.empty or validation.empty:
        raise ValueError("Fixed train and validation partitions are required to create a release.")

    # Threshold selection remains validation-only.  This candidate is never
    # exposed for inference when the released estimator is refit on all
    # approved historical records.
    threshold_model = fit_single_modality(train, config, args.model_name)
    select_threshold_on_validation(threshold_model, validation)
    if args.fit_scope == "full_feature_table":
        released_model = fit_single_modality(table, config, args.model_name)
        released_rows = table
    else:
        released_model = fit_single_modality(train, config, args.model_name)
        released_rows = train
    released_model.threshold = threshold_model.threshold

    config_sha256 = sha256_file(config_path)
    training = {
        "fit_scope": args.fit_scope,
        "fit_row_count": int(len(released_rows)),
        "fit_participant_id_sha256": hashlib.sha256(
            "\n".join(sorted(released_rows[identifier].astype(str))).encode("utf-8")
        ).hexdigest(),
        "feature_table_path": str(feature_path.relative_to(ROOT)).replace("\\", "/"),
        "feature_table_sha256": sha256_file(feature_path),
        "split_path": str(split_path.relative_to(ROOT)).replace("\\", "/"),
        "split_sha256": sha256_file(split_path),
        "training_config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "training_config_sha256": config_sha256,
        "threshold_selection": "validation_max_balanced_accuracy",
        "threshold_selection_rows": int(len(validation)),
        "threshold_selection_fit_scope": "train",
        "new_inference_records_used_for_training": False,
    }
    release_id = (
        f"{config['data_version']}-{args.model_name}-{args.fit_scope}-{config_sha256[:12]}"
    )
    manifest = save_model_release(
        released_model,
        args.output_dir,
        release_id=release_id,
        training=training,
        overwrite=args.overwrite,
    )
    print(json.dumps({"release_directory": str(args.output_dir), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
