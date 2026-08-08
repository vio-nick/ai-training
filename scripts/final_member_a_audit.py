from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    required = [
        "docs/data_dictionary.md",
        "docs/data_versions/synthetic_v1.md",
        "docs/data_quality/synthetic_v1.md",
        "docs/interfaces/feature_table_v1.md",
        "docs/interfaces/data_split_v1.md",
        "docs/interfaces/baseline_results_v1.md",
        "docs/interfaces/integration_handoff_v1.md",
        "docs/experiments/baseline_synthetic_v1.md",
        "configs/baseline_synthetic_v1.json",
        "data/synthetic/synthetic_v1_raw.csv",
        "data/processed/feature_table_v1.csv",
        "data/splits/split_v1.csv",
        "reports/baselines/synthetic_v1_metrics.json",
        "reports/baselines/synthetic_v1_test_predictions.csv",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required handoff files: {missing}")

    feature_rows = read_csv(ROOT / "data/processed/feature_table_v1.csv")
    split_rows = read_csv(ROOT / "data/splits/split_v1.csv")
    feature_ids = {row["participant_id"] for row in feature_rows}
    split_ids = [row["participant_id"] for row in split_rows]
    if len(feature_rows) != 240 or len(feature_ids) != 240:
        raise ValueError("Feature table must contain 240 unique rows")
    if len(split_rows) != 240 or len(set(split_ids)) != 240 or set(split_ids) != feature_ids:
        raise ValueError("Split and feature IDs are not a one-to-one match")

    metrics_path = ROOT / "reports/baselines/synthetic_v1_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if set(metrics["models"]) != {"logistic_regression", "random_forest"}:
        raise ValueError("Expected exactly two baseline models")
    for result in metrics["models"].values():
        if result["validation"]["n"] != 48 or result["test"]["n"] != 48:
            raise ValueError("Validation/test sample counts are not 48")
        if result["validation"]["threshold"] != result["test"]["threshold"]:
            raise ValueError("Test threshold differs from validation threshold")

    prediction_rows = read_csv(ROOT / "reports/baselines/synthetic_v1_test_predictions.csv")
    if len(prediction_rows) != 96:
        raise ValueError("Expected 96 test predictions (2 models x 48 rows)")
    for model in ("logistic_regression", "random_forest"):
        model_ids = [row["participant_id"] for row in prediction_rows if row["model"] == model]
        if len(model_ids) != 48 or len(set(model_ids)) != 48:
            raise ValueError(f"Prediction IDs invalid for {model}")

    checks = [
        "契约、数据版本、质量报告、接口和交接文档齐全",
        "合成特征表 240 行且 participant_id 唯一",
        "切分清单与特征表 ID 一一对应，无跨集合重复",
        "基线验证/测试样本数均为 48，测试阈值沿用验证阈值",
        "两个模型共生成 96 条唯一测试预测",
        "原始数据、特征表和切分文件校验值已记录",
    ]
    lines = [
        "# 成员 A 最终交付审计：synthetic_v1",
        "",
        "以下审计由 `scripts/final_member_a_audit.py` 自动执行，结论只适用于合成开发数据。",
        "",
        "## 通过项",
        "",
    ]
    lines.extend(f"- [x] {check}" for check in checks)
    lines.extend(
        [
            "",
            "## 外部前置条件",
            "",
            "- [ ] 成员 B 确认 `v1` 字段、切分、结果格式和适配器接口。",
            "- [ ] 使用获批的真实/公开数据完成正式数据准入、质量审查和基线重跑。",
            "- [ ] 跨成员集成 PR 完成双人审阅并合并。",
            "",
            "在上述条件完成前，不得把合成数据指标写成临床性能或对外结论。",
        ]
    )
    report_path = ROOT / "docs/experiments/member_a_final_audit.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)
    print("raw_sha256=" + sha256(ROOT / "data/synthetic/synthetic_v1_raw.csv"))
    print("feature_sha256=" + sha256(ROOT / "data/processed/feature_table_v1.csv"))
    print("split_sha256=" + sha256(ROOT / "data/splits/split_v1.csv"))


if __name__ == "__main__":
    main()
