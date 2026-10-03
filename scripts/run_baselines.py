from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "baseline_synthetic_v1.json"


def read_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def config_sha256() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def load_dataset(config: dict) -> pd.DataFrame:
    features = pd.read_csv(ROOT / config["feature_table_path"])
    splits = pd.read_csv(ROOT / config["split_path"])
    if features[config["id_column"]].duplicated().any():
        raise ValueError("feature table contains duplicate participant IDs")
    if splits[config["id_column"]].duplicated().any():
        raise ValueError("split table contains duplicate participant IDs")
    if set(features[config["id_column"]]) != set(splits[config["id_column"]]):
        raise ValueError("feature and split ID sets do not match")
    frame = features.merge(splits, on=config["id_column"], how="left", validate="one_to_one")
    if frame["split"].isna().any():
        raise ValueError("some feature rows have no split")
    return frame


def make_preprocessor(config: dict) -> ColumnTransformer:
    features = config["feature_columns"]
    categorical = ["sex_at_birth"]
    numeric = [column for column in features if column not in categorical]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)],
        remainder="drop",
    )


def make_models(config: dict) -> dict[str, object]:
    settings = config["models"]
    return {
        "logistic_regression": LogisticRegression(random_state=config["random_seed"], **settings["logistic_regression"]),
        "random_forest": RandomForestClassifier(random_state=config["random_seed"], **settings["random_forest"]),
    }


def choose_threshold(y_true: pd.Series, probabilities) -> float:
    candidates = sorted({0.0, 0.5, 1.0, *[round(float(value), 6) for value in probabilities]})
    scored = [(f1_score(y_true, probabilities >= threshold, zero_division=0), threshold) for threshold in candidates]
    best_f1 = max(score for score, _ in scored)
    return min(threshold for score, threshold in scored if score == best_f1)


def metrics(y_true: pd.Series, probabilities, threshold: float) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    positives = int(y_true.sum())
    true_negatives = int(((y_true == 0) & (predictions == 0)).sum())
    false_positives = int(((y_true == 0) & (predictions == 1)).sum())
    true_positives = int(((y_true == 1) & (predictions == 1)).sum())
    negatives = int((y_true == 0).sum())
    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "sensitivity": float(true_positives / positives) if positives else 0.0,
        "specificity": float(true_negatives / negatives) if negatives else 0.0,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "tp": true_positives,
        "fp": false_positives,
        "tn": true_negatives,
        "fn": int(((y_true == 1) & (predictions == 0)).sum()),
    }


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["participant_id", "model", "split", "label", "probability", "threshold", "prediction"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path: Path, results: dict[str, object]) -> None:
    rows = [
        "# 基线实验记录：synthetic_v1",
        "",
        "本记录仅验证成员 A 的工程流程，不代表真实人群、临床性能或医学结论。",
        "",
        f"- 数据版本：`{results['data_version']}`",
        f"- 配置 SHA-256：`{results['config_sha256']}`",
        f"- Python：`{results['environment']['python']}`；scikit-learn：`{results['environment']['scikit_learn']}`",
        "- 复现命令：`<python-3.12> scripts\\run_baselines.py`（本次审查使用 Python 3.12.8）。",
        "- 预处理：仅在 `train` 拟合中位数插补、分类编码和数值标准化。",
        "- 阈值：在 `validation` 上选择 F1 最大阈值，固定后用于 `test`；测试集未参与选阈值。",
        "",
        "## 指标",
        "",
        "| 模型 | 集合 | 阈值 | AUROC | AUPRC | F1 | 灵敏度 | 特异度 | Brier |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name, model_result in results["models"].items():
        for split in ("validation", "test"):
            metrics_result = model_result[split]
            rows.append(
                f"| `{model_name}` | `{split}` | {metrics_result['threshold']:.6f} | {metrics_result['auroc']:.4f} | {metrics_result['auprc']:.4f} | {metrics_result['f1']:.4f} | {metrics_result['sensitivity']:.4f} | {metrics_result['specificity']:.4f} | {metrics_result['brier_score']:.4f} |"
            )
    rows.extend(
        [
            "",
            "## 审查结论",
            "",
            "- 两个模型使用同一份 `feature_table_v1.csv`、`split_v1.csv`、随机种子和预处理边界。",
            "- 验证集阈值分别为逻辑回归 0.579842、随机森林 0.409856；测试集只使用这些已冻结阈值。",
            "- 结果文件为合成数据产物；正式数据接入后必须重新记录数据版本、配置哈希、环境和结果，不能沿用本报告结论。",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    config = read_config()
    frame = load_dataset(config)
    feature_columns = config["feature_columns"]
    x_train = frame.loc[frame["split"] == "train", feature_columns]
    y_train = frame.loc[frame["split"] == "train", config["label_column"]]
    x_validation = frame.loc[frame["split"] == "validation", feature_columns]
    y_validation = frame.loc[frame["split"] == "validation", config["label_column"]]
    x_test = frame.loc[frame["split"] == "test", feature_columns]
    y_test = frame.loc[frame["split"] == "test", config["label_column"]]

    results: dict[str, object] = {
        "data_version": config["data_version"],
        "config_sha256": config_sha256(),
        "seed": config["random_seed"],
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "models": {},
    }
    prediction_rows: list[dict[str, object]] = []
    for name, estimator in make_models(config).items():
        pipeline = Pipeline([("preprocess", make_preprocessor(config)), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        validation_probabilities = pipeline.predict_proba(x_validation)[:, 1]
        threshold = choose_threshold(y_validation, validation_probabilities)
        test_probabilities = pipeline.predict_proba(x_test)[:, 1]
        results["models"][name] = {
            "validation": metrics(y_validation, validation_probabilities, threshold),
            "test": metrics(y_test, test_probabilities, threshold),
        }
        for row, label, probability in zip(frame.loc[frame["split"] == "test"].to_dict("records"), y_test, test_probabilities):
            prediction_rows.append(
                {
                    "participant_id": row[config["id_column"]],
                    "model": name,
                    "split": "test",
                    "label": int(label),
                    "probability": round(float(probability), 8),
                    "threshold": round(float(threshold), 8),
                    "prediction": int(probability >= threshold),
                }
            )

    output_dir = ROOT / "reports" / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "synthetic_v1_metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_predictions(output_dir / "synthetic_v1_test_predictions.csv", prediction_rows)
    write_markdown_report(ROOT / "docs" / "experiments" / "baseline_synthetic_v1.md", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
