"""Score new feature records with a fixed, previously released GSTRIDE model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fall_prediction.inference import score_records
from fall_prediction.release import load_model_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score new GSTRIDE records without fitting, recalibrating, or updating a model."
    )
    parser.add_argument("--release-dir", type=Path, required=True, help="Directory containing manifest.json and model.joblib.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-json", type=Path, help="One JSON object or a JSON array of input objects.")
    source.add_argument("--input-csv", type=Path, help="CSV with one input record per row.")
    parser.add_argument("--output", type=Path, help="Optional .json or .csv destination. Results are always printed as JSON.")
    return parser.parse_args()


def _json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    raise ValueError("Input JSON must be one object or an array of objects.")


def _write_output(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        pd.DataFrame(results).to_csv(path, index=False, encoding="utf-8")
    elif path.suffix.lower() == ".json":
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        raise ValueError("Output path must use a .json or .csv extension.")


def main() -> None:
    args = parse_args()
    model, manifest = load_model_release(args.release_dir)
    if args.input_json:
        results = score_records(model, _json_records(args.input_json), data_version=manifest["data_version"])
    else:
        frame = pd.read_csv(args.input_csv)
        results = model.predict_frame(frame, data_version=manifest["data_version"]).to_dict(orient="records")
    if args.output:
        _write_output(args.output, results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
