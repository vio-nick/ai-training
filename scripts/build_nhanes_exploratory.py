from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "public" / "nhanes_2017_2018" / "raw"
OUT = ROOT / "data" / "public" / "nhanes_2017_2018" / "processed"
SEED = 20260808

FEATURE_COLUMNS = [
    "participant_id",
    "fracture_history",
    "osteoporosis_diagnosis",
    "age_years",
    "sex_at_birth",
    "bmi_kg_m2",
    "lumbar_spine_bmd_g_cm2",
    "pelvis_bmd_g_cm2",
    "head_bmd_g_cm2",
    "survey_exam_weight",
    "survey_stratum",
    "survey_psu",
]


def read_xpt(name: str) -> pd.DataFrame:
    return pd.read_sas(RAW / f"{name}.xpt", format="xport", encoding="latin1")


def yes_no(series: pd.Series) -> pd.Series:
    return series.map({1.0: 1, 2.0: 0}).astype("Int64")


def split_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(SEED)
    output: list[dict[str, str]] = []
    for label, group in frame.groupby("fracture_history", sort=True):
        ids = list(group["participant_id"])
        rng.shuffle(ids)
        train_n = round(len(ids) * 0.60)
        validation_n = round(len(ids) * 0.20)
        for index, participant_id in enumerate(ids):
            split = "train" if index < train_n else "validation" if index < train_n + validation_n else "test"
            output.append({"participant_id": participant_id, "split": split})
    return pd.DataFrame(output).sort_values("participant_id").reset_index(drop=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    demo = read_xpt("DEMO_J")
    bmx = read_xpt("BMX_J")
    osq = read_xpt("OSQ_J")
    dxx = read_xpt("DXX_J")
    for frame in (demo, bmx, osq, dxx):
        frame["SEQN"] = pd.to_numeric(frame["SEQN"], errors="raise").astype(int)

    demo = demo.loc[pd.to_numeric(demo["RIDAGEYR"], errors="coerce") >= 50, ["SEQN", "RIDAGEYR", "RIAGENDR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"]]
    bmx = bmx[["SEQN", "BMXBMI"]]
    osq = osq[["SEQN", "OSQ060", "OSQ080"]]
    dxx = dxx[["SEQN", "DXXLSBMD", "DXXPEBMD", "DXXHEBMD"]]
    merged = demo.merge(bmx, on="SEQN", how="inner", validate="one_to_one")
    merged = merged.merge(osq, on="SEQN", how="inner", validate="one_to_one")
    merged = merged.merge(dxx, on="SEQN", how="inner", validate="one_to_one")

    result = pd.DataFrame(
        {
            "participant_id": merged["SEQN"].map(lambda value: f"NHANES_J_{int(value):05d}"),
            "fracture_history": yes_no(merged["OSQ080"]),
            "osteoporosis_diagnosis": yes_no(merged["OSQ060"]),
            "age_years": pd.to_numeric(merged["RIDAGEYR"], errors="coerce").round().astype("Int64"),
            "sex_at_birth": merged["RIAGENDR"].map({1.0: "male", 2.0: "female"}),
            "bmi_kg_m2": pd.to_numeric(merged["BMXBMI"], errors="coerce"),
            "lumbar_spine_bmd_g_cm2": pd.to_numeric(merged["DXXLSBMD"], errors="coerce"),
            "pelvis_bmd_g_cm2": pd.to_numeric(merged["DXXPEBMD"], errors="coerce"),
            "head_bmd_g_cm2": pd.to_numeric(merged["DXXHEBMD"], errors="coerce"),
            "survey_exam_weight": pd.to_numeric(merged["WTMEC2YR"], errors="coerce"),
            "survey_stratum": merged["SDMVSTRA"],
            "survey_psu": merged["SDMVPSU"],
        }
    )
    result = result.dropna(subset=["participant_id", "fracture_history", "osteoporosis_diagnosis", "age_years", "sex_at_birth", "bmi_kg_m2"])
    result["fracture_history"] = result["fracture_history"].astype(int)
    result["osteoporosis_diagnosis"] = result["osteoporosis_diagnosis"].astype(int)
    result = result[FEATURE_COLUMNS].sort_values("participant_id").reset_index(drop=True)
    if result["participant_id"].duplicated().any():
        raise ValueError("NHANES participant IDs are not unique after merging")

    splits = split_rows(result)
    if set(splits["participant_id"]) != set(result["participant_id"]):
        raise ValueError("NHANES feature and split IDs do not match")
    OUT.mkdir(parents=True, exist_ok=True)
    feature_path = OUT / "feature_table_nhanes_exploratory_v1.csv"
    split_path = OUT / "split_nhanes_exploratory_v1.csv"
    quality_path = OUT / "nhanes_exploratory_v1_quality.json"
    result.to_csv(feature_path, index=False, na_rep="")
    splits.to_csv(split_path, index=False)
    quality = {
        "n_rows": int(len(result)),
        "label_counts": {str(k): int(v) for k, v in result["fracture_history"].value_counts().sort_index().items()},
        "osteoporosis_diagnosis_counts": {str(k): int(v) for k, v in result["osteoporosis_diagnosis"].value_counts().sort_index().items()},
        "missing_counts": {column: int(result[column].isna().sum()) for column in FEATURE_COLUMNS},
        "split_counts": {str(k): int(v) for k, v in splits["split"].value_counts().sort_index().items()},
        "source_sha256": {name: sha256(RAW / f"{name}.xpt") for name in ("DEMO_J", "BMX_J", "OSQ_J", "DXX_J")},
        "feature_table_sha256": sha256(feature_path),
        "split_sha256": sha256(split_path),
        "seed": SEED,
    }
    quality_path.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    report_lines = [
        "# 数据质量报告：nhanes_2017_2018_public_v1",
        "",
        "本报告描述 CDC/NCHS NHANES 2017–2018 横断面公开数据的工程质量，不代表 24 个月骨折风险性能。",
        "",
        f"- 合并后样本数：{len(result)}",
        f"- `fracture_history` 标签计数：{ {int(k): int(v) for k, v in result['fracture_history'].value_counts().sort_index().items()} }",
        f"- `osteoporosis_diagnosis` 计数：{ {int(k): int(v) for k, v in result['osteoporosis_diagnosis'].value_counts().sort_index().items()} }",
        f"- 切分计数：{ {str(k): int(v) for k, v in splits['split'].value_counts().sort_index().items()} }",
        "- 纳入：年龄至少 50 岁、两个探索性标签有效、年龄/性别/BMI 有效，并在四份官方文件中完成 SEQN 合并。",
        "- 缺失 BMD 保留为空，由训练集合拟合的插补器处理。",
        "- NHANES 复杂抽样权重仅保留为元数据，本次探索性模型未使用调查加权。",
        "",
        "## 字段缺失",
        "",
        "| 字段 | 缺失行数 |",
        "| --- | ---: |",
    ]
    report_lines.extend(f"| `{column}` | {int(result[column].isna().sum())} |" for column in FEATURE_COLUMNS)
    report_lines.extend(
        [
            "",
            "## 结论",
            "",
            "字段、ID 唯一性、标签编码和切分已通过程序校验。标签没有事件日期和 24 个月观察窗；模型结果只能作为公开横断面探索，不得写成临床验证结论。",
        ]
    )
    (ROOT / "docs" / "data_quality").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "data_quality" / "nhanes_2017_2018_public_v1.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
