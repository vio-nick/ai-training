from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "nhanes_exploratory_v2.json"

BASE_FEATURES = [
    "age_years", "sex_at_birth", "bmi_kg_m2", "lumbar_spine_bmd_g_cm2",
    "pelvis_bmd_g_cm2", "head_bmd_g_cm2",
]


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [column for column in features if column in {"sex_at_birth", "survey_cycle"}]
    numeric = [column for column in features if column not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def metric_values(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    weights: pd.Series | None = None,
) -> dict[str, float | int]:
    y = y_true.to_numpy(dtype=int)
    predicted = probabilities >= threshold
    w = np.ones(len(y), dtype=float) if weights is None else weights.to_numpy(dtype=float)
    positives = float(w[y == 1].sum())
    negatives = float(w[y == 0].sum())
    tp = float(w[(y == 1) & predicted].sum())
    tn = float(w[(y == 0) & ~predicted].sum())
    return {
        "n": int(len(y)),
        "positive_rate": float(np.average(y, weights=w)),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, probabilities, sample_weight=w)),
        "auprc": float(average_precision_score(y, probabilities, sample_weight=w)),
        "f1": float(f1_score(y, predicted, sample_weight=w, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted, sample_weight=w)),
        "sensitivity": float(tp / positives) if positives else 0.0,
        "specificity": float(tn / negatives) if negatives else 0.0,
        "brier_score": float(brier_score_loss(y, probabilities, sample_weight=w)),
    }


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray, weights: pd.Series) -> float:
    candidates = sorted({0.0, 0.5, 1.0, *[round(float(value), 6) for value in probabilities]})
    constrained: list[tuple[float, float]] = []
    for threshold in candidates:
        values = metric_values(y_true, probabilities, threshold, weights)
        if values["specificity"] >= 0.5:
            constrained.append((float(values["sensitivity"]), threshold))
    if constrained:
        best_sensitivity = max(score for score, _ in constrained)
        return min(threshold for score, threshold in constrained if score == best_sensitivity)
    scores = [
        (balanced_accuracy_score(y_true, probabilities >= threshold, sample_weight=weights), threshold)
        for threshold in candidates
    ]
    best = max(score for score, _ in scores)
    return min(threshold for score, threshold in scores if score == best)


def make_pipeline(name: str, features: list[str], config: dict) -> Pipeline:
    if name == "logistic_regression":
        model = LogisticRegression(random_state=config["random_seed"], **config["models"][name])
    elif name == "random_forest":
        model = RandomForestClassifier(random_state=config["random_seed"], **config["models"][name])
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preprocess", make_preprocessor(features)), ("model", model)])


def fit_with_weights(pipeline: Pipeline, frame: pd.DataFrame, features: list[str], config: dict) -> Pipeline:
    weights = frame[config["weight_column"]].to_numpy(dtype=float)
    weights = weights / np.mean(weights)
    pipeline.fit(frame[features], frame[config["label_column"]], model__sample_weight=weights)
    return pipeline


def repeated_cv(
    frame: pd.DataFrame,
    features: list[str],
    model_name: str,
    config: dict,
) -> dict[str, object]:
    splitter = RepeatedStratifiedKFold(
        n_splits=config["cross_validation"]["n_splits"],
        n_repeats=config["cross_validation"]["n_repeats"],
        random_state=config["random_seed"],
    )
    predictions: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    y = frame[config["label_column"]].to_numpy(dtype=int)
    for train_idx, validation_idx in splitter.split(frame, y):
        pipeline = make_pipeline(model_name, features, config)
        train = frame.iloc[train_idx]
        validation = frame.iloc[validation_idx]
        fit_with_weights(pipeline, train, features, config)
        predictions.append(pipeline.predict_proba(validation[features])[:, 1])
        observed.append(validation[config["label_column"]].to_numpy(dtype=int))
        weights.append(validation[config["weight_column"]].to_numpy(dtype=float))
    probability = np.concatenate(predictions)
    observed_array = np.concatenate(observed)
    weight_array = np.concatenate(weights)
    return {
        "n_predictions": int(len(probability)),
        "n_repeats": config["cross_validation"]["n_repeats"],
        "auroc": float(roc_auc_score(observed_array, probability, sample_weight=weight_array)),
        "auprc": float(average_precision_score(observed_array, probability, sample_weight=weight_array)),
        "brier_score": float(brier_score_loss(observed_array, probability, sample_weight=weight_array)),
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    feature_path = ROOT / config["feature_table_path"]
    split_path = ROOT / config["split_path"]
    frame = pd.read_csv(feature_path).merge(
        pd.read_csv(split_path), on=config["id_column"], how="inner", validate="one_to_one"
    )
    full_features = config["feature_columns"]
    feature_sets = {"base": BASE_FEATURES, "expanded": full_features}
    models = {name: make_pipeline(name, full_features, config) for name in config["models"]}
    result: dict[str, object] = {
        "data_version": config["data_version"],
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "seed": config["random_seed"],
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "label": "fracture_history (OSQ080; cross-sectional history, not 24-month endpoint)",
        "weighting": "WTMEC2YR, normalized within each training split; pooled-cycle weight scale is not a population estimate",
        "models": {},
        "ablation": {},
    }
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    test = frame[frame["split"] == "test"]
    test_predictions = test[[config["id_column"], config["label_column"]]].copy()

    for feature_set_name, features in feature_sets.items():
        for model_name in config["models"]:
            pipeline = make_pipeline(model_name, features, config)
            fit_with_weights(pipeline, train, features, config)
            validation_prob = pipeline.predict_proba(validation[features])[:, 1]
            test_prob = pipeline.predict_proba(test[features])[:, 1]
            validation_weights = validation[config["weight_column"]]
            test_weights = test[config["weight_column"]]
            threshold = choose_threshold(validation[config["label_column"]], validation_prob, validation_weights)
            key = f"{feature_set_name}_{model_name}"
            result["models"][key] = {
                "feature_set": feature_set_name,
                "validation": {
                    "fixed_0_5": metric_values(validation[config["label_column"]], validation_prob, 0.5, validation_weights),
                    "selected": metric_values(validation[config["label_column"]], validation_prob, threshold, validation_weights),
                },
                "test": {
                    "fixed_0_5": metric_values(test[config["label_column"]], test_prob, 0.5, test_weights),
                    "selected": metric_values(test[config["label_column"]], test_prob, threshold, test_weights),
                },
                "repeated_cv": repeated_cv(frame, features, model_name, config),
            }
            if feature_set_name == "expanded":
                test_predictions[f"{model_name}_probability"] = test_prob

    for feature_set_name, features in feature_sets.items():
        result["ablation"][feature_set_name] = result["models"][f"{feature_set_name}_logistic_regression"]["repeated_cv"]

    output = ROOT / "reports" / "baselines" / "nhanes_exploratory_v2_metrics.json"
    predictions_output = ROOT / "reports" / "baselines" / "nhanes_exploratory_v2_test_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    test_predictions.to_csv(predictions_output, index=False)

    lines = [
        "# NHANES 2013–2018 探索性基线：v2",
        "",
        "本实验合并 CDC/NCHS NHANES 2013–2014 与 2017–2018 两个兼容周期，主标签仍为 `OSQ080` 既往骨折史。它是横断面历史标签，不是索引时点后的24个月骨折终点，结果不得作为正式临床性能或筛查结论。",
        "",
        f"- 数据版本：`{result['data_version']}`；样本数：`{len(frame)}`；切分：`{len(train)}/{len(validation)}/{len(test)}`（训练/验证/测试）",
        f"- 配置 SHA-256：`{result['config_sha256']}`；权重：`WTMEC2YR`，仅作相对加权，不作总体估计",
        "- 预处理只在训练集拟合；BMD及数值特征保留缺失指示器；分类缺失单独编码；测试集不参与阈值或特征选择。",
        "",
        "## 消融与重复交叉验证",
        "",
        "| 特征集 | 重复折数 | 加权 AUROC | 加权 AUPRC | 加权 Brier |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("base", "expanded"):
        m = result["ablation"][name]
        lines.append(f"| `{name}` | {m['n_repeats']}×5 | {m['auroc']:.4f} | {m['auprc']:.4f} | {m['brier_score']:.4f} |")
    lines.extend(["", "## 固定切分结果", "", "| 模型 | 验证 AUROC | 测试 AUROC | 测试 AUPRC | 测试固定阈值特异度 | 测试验证阈值特异度 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for key, values in result["models"].items():
        lines.append(
            f"| `{key}` | {values['validation']['fixed_0_5']['auroc']:.4f} | {values['test']['fixed_0_5']['auroc']:.4f} | "
            f"{values['test']['fixed_0_5']['auprc']:.4f} | {values['test']['fixed_0_5']['specificity']:.4f} | "
            f"{values['test']['selected']['specificity']:.4f} |"
        )
    lines.extend([
        "",
        "## 解释与限制",
        "",
        "扩展特征集用于检验索引时点临床风险信息是否能改善横断面历史标签的可分性；改善不能证明对24个月未来骨折有预测效度。两个周期的问卷仍主要覆盖50–59岁，BMD和部分问卷字段存在选择性缺失，调查权重也未用于复杂抽样方差估计。正式任务仍需经批准的纵向队列和明确事件日期。",
        "",
        "复现命令：`D:\\anaconda3\\python.exe scripts/build_nhanes_multi_cycle.py`，然后运行 `D:\\anaconda3\\python.exe scripts/run_nhanes_baseline_v2.py`。",
    ])
    (ROOT / "docs" / "experiments" / "nhanes_exploratory_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
