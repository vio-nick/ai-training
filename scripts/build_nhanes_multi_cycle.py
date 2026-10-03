from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260808
CYCLES = {
    "2013_2014": "H",
    "2017_2018": "J",
}
RAW_ROOT = ROOT / "data" / "public"
AUX_ROOT = ROOT / "data" / "public" / "nhanes_aux" / "raw"
OUT = ROOT / "data" / "public" / "nhanes_2013_2018" / "processed"

LABEL = "fracture_history"
FEATURE_COLUMNS = [
    "participant_id",
    LABEL,
    "osteoporosis_diagnosis",
    "age_years",
    "sex_at_birth",
    "bmi_kg_m2",
    "lumbar_spine_bmd_g_cm2",
    "pelvis_bmd_g_cm2",
    "head_bmd_g_cm2",
    "smoking_ever",
    "smoking_current",
    "alcohol_ever",
    "vigorous_activity",
    "moderate_activity",
    "high_blood_pressure",
    "diabetes",
    "coronary_heart_disease",
    "heart_attack",
    "stroke",
    "survey_cycle",
    "survey_exam_weight",
    "survey_stratum",
    "survey_psu",
]


def read_xpt(path: Path) -> pd.DataFrame:
    frame = pd.read_sas(path, format="xport", encoding="latin1")
    frame["SEQN"] = pd.to_numeric(frame["SEQN"], errors="raise").astype(int)
    return frame


def binary(series: pd.Series) -> pd.Series:
    return series.map({1.0: 1, 2.0: 0}).astype("Float64")


def merge_cycle(cycle: str, suffix: str) -> pd.DataFrame:
    raw = RAW_ROOT / f"nhanes_{cycle}" / "raw"
    demo = read_xpt(raw / f"DEMO_{suffix}.xpt")
    bmx = read_xpt(raw / f"BMX_{suffix}.xpt")
    osq = read_xpt(raw / f"OSQ_{suffix}.xpt")
    dxx = read_xpt(raw / f"DXX_{suffix}.xpt")

    demo = demo.loc[
        pd.to_numeric(demo["RIDAGEYR"], errors="coerce") >= 50,
        ["SEQN", "RIDAGEYR", "RIAGENDR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"],
    ]
    merged = demo.merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="inner", validate="one_to_one")
    merged = merged.merge(osq[["SEQN", "OSQ060", "OSQ080"]], on="SEQN", how="inner", validate="one_to_one")
    merged = merged.merge(
        dxx[["SEQN", "DXXLSBMD", "DXXPEBMD", "DXXHEBMD"]],
        on="SEQN",
        how="inner",
        validate="one_to_one",
    )

    auxiliary = {}
    for name in ("MCQ", "SMQ", "ALQ", "PAQ", "BPQ", "DIQ"):
        frame = read_xpt(AUX_ROOT / f"{name}_{suffix}.xpt")
        wanted = [
            column
            for column in (
                "SMQ020", "SMQ040", "ALQ151", "PAQ605", "PAQ665", "BPQ020", "DIQ010",
                "MCQ160A", "MCQ160B", "MCQ160C",
            )
            if column in frame.columns
        ]
        auxiliary[name] = frame[["SEQN", *wanted]]
    for frame in auxiliary.values():
        merged = merged.merge(frame, on="SEQN", how="left", validate="one_to_one")

    # OSQ080 is kept only as a valid binary exploratory label. Other OSQ event
    # details are intentionally excluded because they can leak the label.
    merged = merged.loc[merged["OSQ080"].isin([1.0, 2.0])].copy()
    result = pd.DataFrame(
        {
            "participant_id": merged["SEQN"].map(lambda value: f"NHANES_{suffix}_{int(value):05d}"),
            LABEL: binary(merged["OSQ080"]),
            "osteoporosis_diagnosis": binary(merged["OSQ060"]),
            "age_years": pd.to_numeric(merged["RIDAGEYR"], errors="coerce"),
            "sex_at_birth": merged["RIAGENDR"].map({1.0: "male", 2.0: "female"}),
            "bmi_kg_m2": pd.to_numeric(merged["BMXBMI"], errors="coerce"),
            "lumbar_spine_bmd_g_cm2": pd.to_numeric(merged["DXXLSBMD"], errors="coerce"),
            "pelvis_bmd_g_cm2": pd.to_numeric(merged["DXXPEBMD"], errors="coerce"),
            "head_bmd_g_cm2": pd.to_numeric(merged["DXXHEBMD"], errors="coerce"),
            "smoking_ever": binary(merged["SMQ020"]),
            "smoking_current": binary(merged["SMQ040"]),
            "alcohol_ever": binary(merged["ALQ151"]),
            "vigorous_activity": binary(merged["PAQ605"]),
            "moderate_activity": binary(merged["PAQ665"]),
            "high_blood_pressure": binary(merged["BPQ020"]),
            "diabetes": binary(merged["DIQ010"]),
            "coronary_heart_disease": binary(merged["MCQ160A"]),
            "heart_attack": binary(merged["MCQ160B"]),
            "stroke": binary(merged["MCQ160C"]),
            "survey_cycle": cycle,
            "survey_exam_weight": pd.to_numeric(merged["WTMEC2YR"], errors="coerce"),
            "survey_stratum": merged["SDMVSTRA"],
            "survey_psu": merged["SDMVPSU"],
        }
    )
    required = [LABEL, "age_years", "sex_at_birth", "bmi_kg_m2"]
    result = result.dropna(subset=required)
    result[LABEL] = result[LABEL].astype(int)
    return result[FEATURE_COLUMNS]


def split_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(SEED)
    output: list[dict[str, str]] = []
    for label, group in frame.groupby(LABEL, sort=True):
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
    frames = [merge_cycle(cycle, suffix) for cycle, suffix in CYCLES.items()]
    result = pd.concat(frames, ignore_index=True).sort_values("participant_id").reset_index(drop=True)
    if result["participant_id"].duplicated().any():
        raise ValueError("NHANES participant IDs are not unique")
    splits = split_rows(result)
    if set(result["participant_id"]) != set(splits["participant_id"]):
        raise ValueError("Feature and split IDs do not match")

    OUT.mkdir(parents=True, exist_ok=True)
    feature_path = OUT / "feature_table_nhanes_exploratory_v2.csv"
    split_path = OUT / "split_nhanes_exploratory_v2.csv"
    quality_path = OUT / "nhanes_exploratory_v2_quality.json"
    result.to_csv(feature_path, index=False, na_rep="")
    splits.to_csv(split_path, index=False)
    source_paths = []
    for cycle, suffix in CYCLES.items():
        source_paths.extend((RAW_ROOT / f"nhanes_{cycle}" / "raw" / f"{name}_{suffix}.xpt") for name in ("DEMO", "BMX", "OSQ", "DXX"))
        source_paths.extend(AUX_ROOT / f"{name}_{suffix}.xpt" for name in ("MCQ", "SMQ", "ALQ", "PAQ", "BPQ", "DIQ"))
    quality = {
        "n_rows": int(len(result)),
        "label_counts": {str(k): int(v) for k, v in result[LABEL].value_counts().sort_index().items()},
        "cycle_counts": {str(k): int(v) for k, v in result["survey_cycle"].value_counts().sort_index().items()},
        "split_counts": {str(k): int(v) for k, v in splits["split"].value_counts().sort_index().items()},
        "missing_counts": {column: int(result[column].isna().sum()) for column in FEATURE_COLUMNS},
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        "feature_table_sha256": sha256(feature_path),
        "split_sha256": sha256(split_path),
        "seed": SEED,
        "label_definition": "OSQ080 historical fracture, cross-sectional exploratory endpoint; not a 24-month outcome",
    }
    quality_path.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
