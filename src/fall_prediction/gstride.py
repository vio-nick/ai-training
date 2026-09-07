"""Create the six-feature sensor-only GSTRIDE faller data set."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path

import pandas as pd


RANDOM_SEED = 20260907
RAW_ENCODING = "cp1252"
FEATURE_COLUMNS = [
    "step_speed_m_s",
    "cadence_strides_per_min",
    "stride_length_m",
    "double_support_pct",
    "swing_to_stance_ratio",
    "stride_time_cv_pct",
]
OUTPUT_COLUMNS = ["participant_id", "faller_last_year", *FEATURE_COLUMNS]

# Database_register.csv has three documentation/header rows. These indices refer
# to the source's second row and are retained explicitly to make the mapping auditable.
SOURCE_COLUMNS = {
    "participant_id": 1,
    "faller_last_year": 4,
    "stride_time_avg_s": 34,
    "stride_time_std_s": 35,
    "load_pct": 36,
    "foot_flat_pct": 38,
    "push_pct": 40,
    "swing_pct": 42,
    "cadence_strides_per_min": 48,
    "step_speed_m_s": 50,
    "stride_length_m": 52,
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric(source: pd.Series) -> pd.Series:
    cleaned = source.astype(str).str.strip().replace({"": None, "-": None, "Incapable": None})
    return pd.to_numeric(cleaned.str.replace(",", ".", regex=False), errors="coerce")


def load_feature_table(raw_path: Path) -> pd.DataFrame:
    """Load the public register and derive the fixed sensor-only feature set."""

    register = pd.read_csv(
        raw_path,
        sep=";",
        encoding=RAW_ENCODING,
        header=None,
        skiprows=3,
        dtype=str,
        keep_default_na=False,
    )
    source = {name: register.iloc[:, index] for name, index in SOURCE_COLUMNS.items()}
    label_text = source["faller_last_year"].str.strip().str.upper()
    labels = label_text.map({"NO": 0, "YES": 1})
    if labels.isna().any():
        invalid = sorted(label_text[labels.isna()].unique())
        raise ValueError(f"Unexpected falls label(s): {invalid}")

    step_speed = _numeric(source["step_speed_m_s"])
    cadence = _numeric(source["cadence_strides_per_min"])
    stride_length = _numeric(source["stride_length_m"])
    load = _numeric(source["load_pct"])
    foot_flat = _numeric(source["foot_flat_pct"])
    push = _numeric(source["push_pct"])
    swing = _numeric(source["swing_pct"])
    stride_time_avg = _numeric(source["stride_time_avg_s"])
    stride_time_std = _numeric(source["stride_time_std_s"])

    result = pd.DataFrame(
        {
            "participant_id": source["participant_id"].str.strip(),
            "faller_last_year": labels.astype("Int64"),
            "step_speed_m_s": step_speed,
            "cadence_strides_per_min": cadence,
            "stride_length_m": stride_length,
            # Load and push are the initial and terminal double-support phases.
            "double_support_pct": load + push,
            "swing_to_stance_ratio": swing / (load + foot_flat + push),
            "stride_time_cv_pct": 100 * stride_time_std / stride_time_avg,
        }
    )
    if result["participant_id"].eq("").any() or result["participant_id"].duplicated().any():
        raise ValueError("GSTRIDE participant IDs must be present and unique")
    if not result["faller_last_year"].isin([0, 1]).all():
        raise ValueError("Faller labels must be binary")
    return result.sort_values("participant_id").reset_index(drop=True)


def make_stratified_split(frame: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create a deterministic 60/20/20 participant-level split."""

    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for _, group in frame.groupby("faller_last_year", sort=True):
        participant_ids = sorted(group["participant_id"].tolist())
        rng.shuffle(participant_ids)
        train_count = round(len(participant_ids) * 0.60)
        validation_count = round(len(participant_ids) * 0.20)
        for index, participant_id in enumerate(participant_ids):
            split = "train" if index < train_count else "validation" if index < train_count + validation_count else "test"
            rows.append({"participant_id": participant_id, "split": split})
    result = pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)
    if set(result["participant_id"]) != set(frame["participant_id"]):
        raise ValueError("Split IDs do not exactly match GSTRIDE feature IDs")
    return result


def quality_summary(frame: pd.DataFrame, split: pd.DataFrame, raw_path: Path, seed: int = RANDOM_SEED) -> dict[str, object]:
    """Return source, completeness, and split metadata for the processed data."""

    return {
        "data_version": "gstride_fall_v1",
        "n_rows": int(len(frame)),
        "label_column": "faller_last_year",
        "label_definition": "1=reported one or more falls during the year before the gait test; 0=no reported fall",
        "label_counts": {str(key): int(value) for key, value in sorted(Counter(frame["faller_last_year"]).items())},
        "feature_columns": FEATURE_COLUMNS,
        "missing_counts": {column: int(frame[column].isna().sum()) for column in OUTPUT_COLUMNS},
        "value_ranges": {
            column: {"min": float(frame[column].min()), "max": float(frame[column].max())}
            for column in FEATURE_COLUMNS
        },
        "split_counts": {str(key): int(value) for key, value in sorted(Counter(split["split"]).items())},
        "source": {
            "record": "Zenodo 8003441 (version v1.0)",
            "doi": "10.5281/zenodo.8003441",
            "license": "CC-BY-4.0",
            "file": "GSTRIDE_database/Database_register.csv",
            "encoding": RAW_ENCODING,
            "sha256": sha256(raw_path),
        },
        "random_seed": seed,
    }
