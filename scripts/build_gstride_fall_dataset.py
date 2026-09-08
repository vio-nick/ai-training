"""Build the GSTRIDE sensor-only faller feature table and split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fall_prediction.gstride import RANDOM_SEED, load_feature_table, make_stratified_split, quality_summary, sha256


RAW_PATH = ROOT / "data" / "public" / "gstride_v1" / "raw" / "Database_register.csv"
OUT_DIR = ROOT / "data" / "public" / "gstride_v1" / "processed"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing public GSTRIDE register: {RAW_PATH}")
    features = load_feature_table(RAW_PATH)
    split = make_stratified_split(features, RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = OUT_DIR / "feature_table_gstride_fall_v1.csv"
    split_path = OUT_DIR / "split_gstride_fall_v1.csv"
    quality_path = OUT_DIR / "gstride_fall_v1_quality.json"
    features.to_csv(feature_path, index=False, encoding="utf-8")
    split.to_csv(split_path, index=False, encoding="utf-8")
    quality = quality_summary(features, split, RAW_PATH, RANDOM_SEED)
    quality["feature_table_sha256"] = sha256(feature_path)
    quality["split_sha256"] = sha256(split_path)
    quality_path.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
