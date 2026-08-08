from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "nhanes_exploratory_v1.json"


def choose_threshold(y_true: pd.Series, probabilities) -> float:
    candidates = sorted({0.0, 0.5, 1.0, *[round(float(value), 6) for value in probabilities]})
    scores = [(f1_score(y_true, probabilities >= threshold, zero_division=0), threshold) for threshold in candidates]
    best = max(score for score, _ in scores)
    return min(threshold for score, threshold in scores if score == best)


def metric_values(y_true: pd.Series, probabilities, threshold: float) -> dict[str, float | int]:
    predicted = probabilities >= threshold
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    tp = int(((y_true == 1) & predicted).sum())
    tn = int(((y_true == 0) & ~predicted).sum())
    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "sensitivity": float(tp / positives) if positives else 0.0,
        "specificity": float(tn / negatives) if negatives else 0.0,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "tp": tp,
        "fp": int(((y_true == 0) & predicted).sum()),
        "tn": tn,
        "fn": int(((y_true == 1) & ~predicted).sum()),
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    feature_path = ROOT / config["feature_table_path"]
    split_path = ROOT / config["split_path"]
    frame = pd.read_csv(feature_path).merge(pd.read_csv(split_path), on=config["id_column"], how="inner", validate="one_to_one")
    features = config["feature_columns"]
    categorical = ["sex_at_birth"]
    numeric = [column for column in features if column not in categorical]
    preprocessor = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        remainder="drop",
    )
    models = {
        "logistic_regression": LogisticRegression(random_state=config["random_seed"], **config["models"]["logistic_regression"]),
        "random_forest": RandomForestClassifier(random_state=config["random_seed"], **config["models"]["random_forest"]),
    }
    result: dict[str, object] = {
        "data_version": config["data_version"],
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "seed": config["random_seed"],
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "label": "fracture_history (OSQ080; cross-sectional history, not 24-month endpoint)",
        "models": {},
    }
    for name, model in models.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        train = frame[frame["split"] == "train"]
        validation = frame[frame["split"] == "validation"]
        test = frame[frame["split"] == "test"]
        pipeline.fit(train[features], train[config["label_column"]])
        validation_prob = pipeline.predict_proba(validation[features])[:, 1]
        threshold = choose_threshold(validation[config["label_column"]], validation_prob)
        test_prob = pipeline.predict_proba(test[features])[:, 1]
        result["models"][name] = {
            "validation": metric_values(validation[config["label_column"]], validation_prob, threshold),
            "test": metric_values(test[config["label_column"]], test_prob, threshold),
        }
    output = ROOT / "reports" / "baselines" / "nhanes_exploratory_v1_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# NHANES 2017–2018 探索性基线：v1",
        "",
        "本结果使用 CDC/NCHS NHANES 横断面公开数据，标签为 `OSQ080` 的既往骨折史，不是 24 个月骨折终点。不得据此作临床性能或筛查结论。",
        "",
        f"- 数据版本：`{result['data_version']}`",
        f"- 配置 SHA-256：`{result['config_sha256']}`",
        f"- Python：`{result['environment']['python']}`；scikit-learn：`{result['environment']['scikit_learn']}`",
        "- 预处理只在训练集合拟合，阈值只在验证集合按 F1 选择，测试集合仅作最终评估。",
        "",
        "| 模型 | 集合 | 阈值 | AUROC | AUPRC | F1 | 灵敏度 | 特异度 | Brier |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in result["models"].items():
        for split in ("validation", "test"):
            m = values[split]
            lines.append(f"| `{name}` | `{split}` | {m['threshold']:.6f} | {m['auroc']:.4f} | {m['auprc']:.4f} | {m['f1']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} | {m['brier_score']:.4f} |")
    (ROOT / "docs" / "experiments" / "nhanes_exploratory_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
