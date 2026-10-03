"""Train and evaluate sensor-only faller baselines on GSTRIDE."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


CONFIG_PATH = ROOT / "configs" / "gstride_fall_v1.json"


def make_pipeline(model_name: str, config: dict[str, object]) -> Pipeline:
    models = config["models"]
    if model_name == "logistic_regression":
        estimator = LogisticRegression(random_state=config["random_seed"], **models[model_name])
        steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", estimator)]
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(random_state=config["random_seed"], **models[model_name])
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline(steps)


def metrics(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    predicted = probabilities >= threshold
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    tp = int(((y == 1) & predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, probabilities)),
        "auprc": float(average_precision_score(y, probabilities)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "sensitivity": float(tp / positives) if positives else 0.0,
        "specificity": float(tn / negatives) if negatives else 0.0,
        "brier_score": float(brier_score_loss(y, probabilities)),
        "tp": tp,
        "fp": int(((y == 0) & predicted).sum()),
        "tn": tn,
        "fn": int(((y == 1) & ~predicted).sum()),
    }


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    candidates = sorted({0.0, 0.5, 1.0, *[round(float(value), 6) for value in probabilities]})
    scored = [(metrics(y_true, probabilities, threshold)["balanced_accuracy"], threshold) for threshold in candidates]
    best_score = max(score for score, _ in scored)
    return min(threshold for score, threshold in scored if score == best_score)


def training_only_repeated_cv(train: pd.DataFrame, feature_columns: list[str], label_column: str, config: dict[str, object], model_name: str) -> dict[str, object]:
    settings = config["cross_validation"]
    split = RepeatedStratifiedKFold(
        n_splits=settings["n_splits"], n_repeats=settings["n_repeats"], random_state=config["random_seed"]
    )
    y = train[label_column].to_numpy(dtype=int)
    probabilities: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    for fit_index, evaluate_index in split.split(train[feature_columns], y):
        pipeline = make_pipeline(model_name, config)
        fitted = train.iloc[fit_index]
        held_out = train.iloc[evaluate_index]
        pipeline.fit(fitted[feature_columns], fitted[label_column])
        probabilities.append(pipeline.predict_proba(held_out[feature_columns])[:, 1])
        observed.append(held_out[label_column].to_numpy(dtype=int))
    probability = np.concatenate(probabilities)
    truth = np.concatenate(observed)
    return {
        "scope": settings["scope"],
        "n_predictions": int(len(probability)),
        "n_splits": settings["n_splits"],
        "n_repeats": settings["n_repeats"],
        "auroc": float(roc_auc_score(truth, probability)),
        "auprc": float(average_precision_score(truth, probability)),
        "brier_score": float(brier_score_loss(truth, probability)),
    }


def model_feature_effects(pipeline: Pipeline, feature_columns: list[str], model_name: str) -> dict[str, float]:
    estimator = pipeline.named_steps["model"]
    if model_name == "logistic_regression":
        values = estimator.coef_.ravel()
    else:
        values = estimator.feature_importances_
    return {column: float(value) for column, value in zip(feature_columns, values, strict=True)}


def markdown_report(result: dict[str, object]) -> str:
    lines = [
        "# GSTRIDE 跌倒者识别基线：v1",
        "",
        "本实验只使用足部 IMU 汇总出的六项 GSTRIDE 步态参数，目标为区分“步态测试前一年内报告过跌倒”与“未报告跌倒”的受试者。它是回顾性跌倒者识别原型，不是对未来跌倒的前瞻性预测，更不是骨折风险或临床诊断结论。",
        "",
        f"- 数据版本：`{result['data_version']}`；标签：`{result['label']}`",
        f"- 样本切分（训练/验证/测试）：`{result['split_sizes']['train']}/{result['split_sizes']['validation']}/{result['split_sizes']['test']}`；随机种子：`{result['seed']}`",
        f"- 配置 SHA-256：`{result['config_sha256']}`",
        "- 插补器和标准化器仅在训练集拟合；决策阈值仅在验证集上以平衡准确率选择；测试集只用于最终一次评估。",
        "",
        "| 模型 | 训练集重复交叉验证 AUROC | 验证 AUROC | 测试 AUROC | 测试 AUPRC | 测试阈值 | 测试敏感度 | 测试特异度 | 测试 Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in result["models"].items():
        validation = values["validation"]
        test = values["test"]
        cv = values["training_only_repeated_cv"]
        lines.append(
            f"| `{name}` | {cv['auroc']:.4f} | {validation['auroc']:.4f} | {test['auroc']:.4f} | "
            f"{test['auprc']:.4f} | {test['threshold']:.6f} | {test['sensitivity']:.4f} | "
            f"{test['specificity']:.4f} | {test['brier_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "样本仅 163 人，标签来自受试者对测试前一年的回顾性报告，且采集地点、功能状态和辅助设备均可能混杂。固定测试集样本较小，性能区间宽；这些结果只能用于验证传感器特征管线和原型可行性。后续须用按时间划分的医院队列验证前瞻性跌倒终点，并在获得合规骨密度和骨折结局后，以新的数据版本评估融合模型。",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_report_clean(result: dict[str, object]) -> str:
    """Render the experiment report with explicit UTF-8 Chinese text."""
    lines = [
        "# GSTRIDE 跌倒者识别基线：v1",
        "",
        "本实验只使用足部 IMU 汇总出的六项 GSTRIDE 步态参数，目标是区分“步态测试前一年内报告过跌倒”与“未报告跌倒”的受试者。它是回顾性跌倒者识别原型，不是对未来跌倒的前瞻性预测，更不是骨折风险或临床诊断结论。",
        "",
        f"- 数据版本：`{result['data_version']}`；标签：`{result['label']}`",
        f"- 样本切分（训练/验证/测试）：`{result['split_sizes']['train']}/{result['split_sizes']['validation']}/{result['split_sizes']['test']}`；随机种子：`{result['seed']}`",
        f"- 配置 SHA-256：`{result['config_sha256']}`",
        "- 插补器和标准化器仅在训练集拟合；决策阈值仅在验证集上以平衡准确率选择；测试集只用于最终一次评估。",
        "",
        "| 模型 | 训练集重复交叉验证 AUROC | 验证 AUROC | 测试 AUROC | 测试 AUPRC | 测试阈值 | 测试敏感度 | 测试特异度 | 测试 Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in result["models"].items():
        validation = values["validation"]
        test = values["test"]
        cv = values["training_only_repeated_cv"]
        lines.append(
            f"| `{name}` | {cv['auroc']:.4f} | {validation['auroc']:.4f} | {test['auroc']:.4f} | "
            f"{test['auprc']:.4f} | {test['threshold']:.6f} | {test['sensitivity']:.4f} | "
            f"{test['specificity']:.4f} | {test['brier_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "样本仅 163 人，标签来自受试者对测试前一年的回顾性报告，且采集地点、功能状态和辅助设备均可能造成混杂。固定测试集样本较小，性能区间宽；这些结果只能用于验证传感器特征管线和原型可行性。后续须以按时间划分的医院队列验证前瞻性跌倒终点，并在获得合规骨密度和骨折结局后，以新的数据版本评估融合模型。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    features = pd.read_csv(ROOT / config["feature_table_path"])
    split = pd.read_csv(ROOT / config["split_path"])
    frame = features.merge(split, on=config["id_column"], how="inner", validate="one_to_one")
    if len(frame) != len(features):
        raise ValueError("Feature and split tables must contain exactly the same participant IDs")
    feature_columns = config["feature_columns"]
    label_column = config["label_column"]
    train = frame.loc[frame["split"] == "train"].copy()
    validation = frame.loc[frame["split"] == "validation"].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    if min(train[label_column].value_counts()) < config["cross_validation"]["n_splits"]:
        raise ValueError("Training data do not contain enough observations per class for configured cross-validation")

    result: dict[str, object] = {
        "data_version": config["data_version"],
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "seed": config["random_seed"],
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "label": "faller_last_year (self-reported fall during the year before the gait test; retrospective)",
        "feature_columns": feature_columns,
        "split_sizes": {name: int(len(data)) for name, data in {"train": train, "validation": validation, "test": test}.items()},
        "models": {},
    }
    predictions = test[[config["id_column"], label_column]].copy()
    for model_name in config["models"]:
        pipeline = make_pipeline(model_name, config)
        pipeline.fit(train[feature_columns], train[label_column])
        validation_probabilities = pipeline.predict_proba(validation[feature_columns])[:, 1]
        threshold = choose_threshold(validation[label_column], validation_probabilities)
        test_probabilities = pipeline.predict_proba(test[feature_columns])[:, 1]
        result["models"][model_name] = {
            "validation": metrics(validation[label_column], validation_probabilities, threshold),
            "test": metrics(test[label_column], test_probabilities, threshold),
            "training_only_repeated_cv": training_only_repeated_cv(train, feature_columns, label_column, config, model_name),
            "feature_effects": model_feature_effects(pipeline, feature_columns, model_name),
        }
        predictions[f"{model_name}_probability"] = test_probabilities
        predictions[f"{model_name}_predicted_label"] = (test_probabilities >= threshold).astype(int)

    report_dir = ROOT / "reports" / "baselines"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "gstride_fall_v1_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    predictions.to_csv(report_dir / "gstride_fall_v1_test_predictions.csv", index=False, encoding="utf-8")
    (ROOT / "docs" / "experiments" / "gstride_fall_v1.md").write_text(markdown_report_clean(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
