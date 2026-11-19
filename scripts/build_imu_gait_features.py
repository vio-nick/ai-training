"""Convert calibrated GSTRIDE CSV files into model feature records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_DIR = ROOT / "src" / "GSTRIDE_IMU_to_Gait_Algorithm"
sys.path.insert(0, str(ALGORITHM_DIR))

from csv_adapter import (  # noqa: E402
    InputQualityError,
    load_calibrated_csv,
    run_gait_algorithm,
    summarize_gait,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Calibrated CSV files")
    parser.add_argument("--output", type=Path, required=True, help="Detailed JSON output")
    parser.add_argument("--model-input", type=Path, required=True, help="Six-feature JSON input for the release scorer")
    parser.add_argument("--allow-quality-fail", action="store_true", help="Emit a technical fallback record when cycle QC fails")
    parser.add_argument("--use-valid-cycles", action="store_true", help="Aggregate only cycles that pass cycle QC; requires at least one valid cycle")
    parser.add_argument("--minimum-valid-cycles", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    details = []
    model_records = []
    failures = []
    for path in args.inputs:
        try:
            imu = load_calibrated_csv(path)
            run = run_gait_algorithm(imu, path)
            result = summarize_gait(
                run,
                allow_quality_fail=args.allow_quality_fail,
                minimum_valid_cycles=args.minimum_valid_cycles,
                use_valid_cycles=args.use_valid_cycles,
            )
            participant_id = path.stem.replace("calibrated_", "")
            result["participant_id"] = participant_id
            details.append(result)
            model_records.append({"participant_id": participant_id, **result["features"]})
        except (InputQualityError, OSError, ValueError) as exc:
            failures.append({"input_path": str(path), "error": str(exc)})
    if failures:
        raise SystemExit(json.dumps({"failures": failures}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model_input.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.model_input.write_text(json.dumps(model_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(args.model_input)
    print(json.dumps(details, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
