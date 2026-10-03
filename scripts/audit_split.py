from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from member_a.data_pipeline import read_csv, read_split_csv, validate_split


def main() -> None:
    feature_path = ROOT / "data" / "processed" / "feature_table_v1.csv"
    split_path = ROOT / "data" / "splits" / "split_v1.csv"
    rows = read_csv(feature_path)
    split_rows = read_split_csv(split_path)
    validate_split(rows, split_rows)
    labels = {row["participant_id"]: row["fracture_within_24m"] for row in rows}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in split_rows:
        counts[row["split"]][labels[row["participant_id"]]] += 1
    expected_sizes = {"train": 144, "validation": 48, "test": 48}
    for split, expected_size in expected_sizes.items():
        actual = sum(counts[split].values())
        if actual != expected_size:
            raise ValueError(f"{split} expected {expected_size} rows, got {actual}")
        if set(counts[split]) != {"0", "1"}:
            raise ValueError(f"{split} must contain both label classes")
    lines = [
        "# 切分审查：synthetic_v1",
        "",
        "## 审查范围",
        "",
        "本审查验证合成开发数据的集合完整性、样本隔离和标签分层；不代表真实数据不存在时间、中心或重复测量泄漏。",
        "",
        "## 结果",
        "",
        "| 集合 | 标签 0 | 标签 1 | 总计 | 阳性比例 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "validation", "test"):
        negative = counts[split]["0"]
        positive = counts[split]["1"]
        total = negative + positive
        lines.append(f"| `{split}` | {negative} | {positive} | {total} | {positive / total:.3f} |")
    lines.extend(
        [
            "",
            "## 审查结论",
            "",
            "- 特征表和切分清单的 ID 集合完全一致，且每个 ID 只出现一次。",
            "- 训练、验证和测试集合大小分别为 144、48 和 48，符合 60/20/20 目标。",
            "- 每个集合均包含标签 0 和标签 1，满足分层训练与评估的最小条件。",
            "- 基线配置要求插补、编码和标准化只能在 `train` 集合拟合；`validation` 仅用于模型选择，`test` 仅用于最终评估。",
        ]
    )
    report_path = ROOT / "docs" / "experiments" / "split_audit_synthetic_v1.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
