from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from member_a.data_pipeline import (
    EXPECTED_COLUMNS,
    quality_summary,
    read_csv,
    read_split_csv,
    sha256,
    stratified_split,
    validate_split,
    write_csv,
    write_json,
    write_quality_report,
)


def main() -> None:
    raw_path = ROOT / "data" / "synthetic" / "synthetic_v1_raw.csv"
    feature_path = ROOT / "data" / "processed" / "feature_table_v1.csv"
    split_path = ROOT / "data" / "splits" / "split_v1.csv"
    rows = read_csv(raw_path)
    summary = quality_summary(rows)
    write_csv(feature_path, rows, EXPECTED_COLUMNS)
    feature_rows = read_csv(feature_path)
    quality_summary(feature_rows)
    split_rows = stratified_split(rows)
    write_csv(split_path, split_rows, ["participant_id", "split"])
    validate_split(feature_rows, read_split_csv(split_path))
    summary["raw_sha256"] = sha256(raw_path)
    summary["feature_table_sha256"] = sha256(feature_path)
    summary["split_sha256"] = sha256(split_path)
    summary["split_counts"] = {name: sum(row["split"] == name for row in split_rows) for name in ("train", "validation", "test")}
    write_json(ROOT / "data" / "processed" / "synthetic_v1_quality.json", summary)
    write_quality_report(ROOT / "docs" / "data_quality" / "synthetic_v1.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
