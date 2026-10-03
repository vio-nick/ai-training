"""Standard-library data pipeline for the member A synthetic contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable


SEED = 20260808
EXPECTED_COLUMNS = [
    "participant_id",
    "fracture_within_24m",
    "age_years",
    "sex_at_birth",
    "bmi_kg_m2",
    "prior_fracture",
    "smoker",
    "alcohol_use",
    "falls_last_year",
    "family_history",
    "lumbar_spine_t_score",
    "femoral_neck_t_score",
    "gait_speed_m_s",
    "stride_length_m",
    "plantar_peak_pressure_kpa",
]
SPLIT_COLUMNS = ["participant_id", "split"]

REQUIRED_COLUMNS = {
    "participant_id",
    "fracture_within_24m",
    "age_years",
    "sex_at_birth",
    "bmi_kg_m2",
    "prior_fracture",
    "smoker",
    "alcohol_use",
    "falls_last_year",
    "family_history",
}

RANGES = {
    "age_years": (50, 95),
    "bmi_kg_m2": (15.0, 45.0),
    "falls_last_year": (0, 10),
    "lumbar_spine_t_score": (-5.0, 2.0),
    "femoral_neck_t_score": (-5.0, 2.0),
    "gait_speed_m_s": (0.2, 2.0),
    "stride_length_m": (0.2, 2.0),
    "plantar_peak_pressure_kpa": (50.0, 900.0),
}


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def generate_synthetic_rows(n: int = 240, seed: int = SEED) -> list[dict[str, str]]:
    """Generate deterministic, non-clinical fixture rows."""

    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for index in range(1, n + 1):
        age = rng.randint(50, 90)
        female = rng.random() < 0.72
        prior = int(rng.random() < _sigmoid(-1.6 + 0.045 * (age - 60)))
        smoker = int(rng.random() < 0.24)
        alcohol = int(rng.random() < 0.18)
        falls = min(8, int(rng.expovariate(0.55)))
        family = int(rng.random() < 0.28)
        bmi = _bounded(rng.gauss(24.8, 3.6), 17.0, 39.0)
        lumbar = _bounded(rng.gauss(-1.15 - 0.025 * (age - 60) - 0.35 * prior, 0.62), -4.5, 1.2)
        femoral = _bounded(rng.gauss(-1.35 - 0.027 * (age - 60) - 0.30 * prior, 0.65), -4.7, 1.0)
        gait = _bounded(rng.gauss(1.05 - 0.009 * (age - 60) - 0.07 * falls, 0.16), 0.35, 1.65)
        stride = _bounded(rng.gauss(1.18 - 0.008 * (age - 60) - 0.04 * falls, 0.15), 0.45, 1.75)
        pressure = _bounded(rng.gauss(410 + 2.0 * (age - 60) + 18 * falls, 75), 100, 750)
        risk = (
            -3.65
            + 0.045 * (age - 60)
            + 0.95 * prior
            + 0.16 * falls
            + 0.55 * family
            - 0.62 * lumbar
            - 0.42 * femoral
            - 0.55 * (gait - 1.0)
            + 0.18 * smoker
            + 0.08 * alcohol
        )
        label = int(rng.random() < _sigmoid(risk))

        row = {
            "participant_id": f"SYN-{index:04d}",
            "fracture_within_24m": str(label),
            "age_years": str(age),
            "sex_at_birth": "female" if female else "male",
            "bmi_kg_m2": f"{bmi:.3f}",
            "prior_fracture": str(prior),
            "smoker": str(smoker),
            "alcohol_use": str(alcohol),
            "falls_last_year": str(falls),
            "family_history": str(family),
            "lumbar_spine_t_score": "" if rng.random() < 0.06 else f"{lumbar:.3f}",
            "femoral_neck_t_score": "" if rng.random() < 0.06 else f"{femoral:.3f}",
            "gait_speed_m_s": "" if rng.random() < 0.10 else f"{gait:.3f}",
            "stride_length_m": "" if rng.random() < 0.10 else f"{stride:.3f}",
            "plantar_peak_pressure_kpa": "" if rng.random() < 0.12 else f"{pressure:.3f}",
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, str]], columns: list[str] = EXPECTED_COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def read_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SPLIT_COLUMNS:
            raise ValueError(f"Unexpected split columns in {path}: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No split rows found in {path}")
    return rows


def _parse_number(row: dict[str, str], column: str, integer: bool = False) -> float | None:
    raw = row[column].strip()
    if raw == "":
        return None
    try:
        value = int(raw) if integer else float(raw)
    except ValueError as exc:
        raise ValueError(f"{column} must be numeric, got {raw!r}") from exc
    return float(value)


def validate_rows(rows: list[dict[str, str]]) -> None:
    """Validate schema, uniqueness, requiredness, categorical values, and ranges."""

    ids = [row["participant_id"].strip() for row in rows]
    if any(not value for value in ids):
        raise ValueError("participant_id cannot be empty")
    if len(ids) != len(set(ids)):
        raise ValueError("participant_id must be unique")
    for row in rows:
        if set(row) != set(EXPECTED_COLUMNS):
            raise ValueError("Row columns do not match the v1 schema")
        if row["sex_at_birth"] not in {"female", "male"}:
            raise ValueError("sex_at_birth must be female or male")
        for column in REQUIRED_COLUMNS - {"participant_id", "sex_at_birth"}:
            if row[column].strip() == "":
                raise ValueError(f"Required column {column} cannot be empty")
        for column in ("fracture_within_24m", "prior_fracture", "smoker", "alcohol_use", "family_history"):
            value = _parse_number(row, column, integer=True)
            if value not in {0.0, 1.0}:
                raise ValueError(f"{column} must be 0 or 1")
        for column, (low, high) in RANGES.items():
            value = _parse_number(row, column, integer=column in {"age_years", "falls_last_year"})
            if value is not None and not low <= value <= high:
                raise ValueError(f"{column} outside [{low}, {high}]: {value}")


def quality_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    validate_rows(rows)
    missing = {column: sum(row[column].strip() == "" for row in rows) for column in EXPECTED_COLUMNS}
    labels = Counter(row["fracture_within_24m"] for row in rows)
    modality_columns = {
        "clinical": ["age_years", "sex_at_birth", "bmi_kg_m2", "prior_fracture", "smoker", "alcohol_use", "falls_last_year", "family_history"],
        "bmd": ["lumbar_spine_t_score", "femoral_neck_t_score"],
        "gait": ["gait_speed_m_s", "stride_length_m"],
        "plantar_pressure": ["plantar_peak_pressure_kpa"],
    }
    availability = {
        modality: sum(all(row[column].strip() for column in columns) for row in rows)
        for modality, columns in modality_columns.items()
    }
    return {
        "n_rows": len(rows),
        "label_counts": dict(sorted(labels.items())),
        "missing_counts": missing,
        "modality_complete_rows": availability,
    }


def stratified_split(rows: list[dict[str, str]], seed: int = SEED) -> list[dict[str, str]]:
    """Create a deterministic 60/20/20 participant-level stratified split."""

    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = {"0": [], "1": []}
    for row in rows:
        by_label[row["fracture_within_24m"]].append(row)
    output: list[dict[str, str]] = []
    for label, group in by_label.items():
        rng.shuffle(group)
        train_count = round(len(group) * 0.60)
        validation_count = round(len(group) * 0.20)
        for index, row in enumerate(group):
            split = "train" if index < train_count else "validation" if index < train_count + validation_count else "test"
            output.append({"participant_id": row["participant_id"], "split": split})
    return sorted(output, key=lambda row: row["participant_id"])


def validate_split(rows: list[dict[str, str]], split_rows: list[dict[str, str]]) -> None:
    expected_ids = {row["participant_id"] for row in rows}
    actual_ids = [row["participant_id"] for row in split_rows]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Split contains duplicate participant_id values")
    if set(actual_ids) != expected_ids:
        raise ValueError("Split IDs must exactly match feature table IDs")
    if any(row["split"] not in {"train", "validation", "test"} for row in split_rows):
        raise ValueError("Split contains an unknown split name")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_quality_report(path: Path, summary: dict[str, object]) -> None:
    labels = summary["label_counts"]
    missing = summary["missing_counts"]
    modalities = summary["modality_complete_rows"]
    split_counts = summary["split_counts"]
    missing_lines = "\n".join(f"| `{column}` | {count} |" for column, count in missing.items())
    modality_lines = "\n".join(f"| `{modality}` | {count} |" for modality, count in modalities.items())
    content = f"""# 数据质量报告：synthetic_v1

## 结论范围

本报告只验证合成开发数据的工程完整性，不代表真实人群分布、临床性能或医学结论。

## 样本与标签

- 样本数：{summary['n_rows']}
- 标签计数：`0`={labels['0']}，`1`={labels['1']}
- 切分计数：train={split_counts['train']}，validation={split_counts['validation']}，test={split_counts['test']}
- 随机种子：`20260808`

## 缺失计数

| 字段 | 缺失行数 |
| --- | ---: |
{missing_lines}

## 模态完整行数

| 模态 | 完整行数 |
| --- | ---: |
{modality_lines}

## 完整性审查

- 必填字段非空、字段顺序和允许范围已通过程序校验。
- `participant_id` 唯一，特征表与切分清单 ID 集合完全一致。
- 切分按标签分层，并按参与者 ID 隔离；预处理器拟合边界记录在切分接口中。
- 数据版本：`synthetic_v1`；正式数据准入、伦理和许可状态仍需另行核验。

## 文件校验值

| 文件 | SHA-256 |
| --- | --- |
| `data/synthetic/synthetic_v1_raw.csv` | `{summary['raw_sha256']}` |
| `data/processed/feature_table_v1.csv` | `{summary['feature_table_sha256']}` |
| `data/splits/split_v1.csv` | `{summary['split_sha256']}` |
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
