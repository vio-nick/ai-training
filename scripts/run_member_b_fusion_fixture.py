"""Execute a clearly non-clinical smoke test for the pluggable fusion interfaces."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fall_prediction.fusion import ModalitySpec, early_fuse_features, hybrid_fuse_probabilities, late_fuse_probabilities


def main() -> None:
    config = json.loads((ROOT / "configs/member_b_fusion_fixture_v1.json").read_text(encoding="utf-8"))
    if not config["fixture_only"]:
        raise ValueError("This smoke-test runner is restricted to an explicitly marked fixture.")
    fixture = pd.DataFrame({"gait_fixture_score": [0.2, 0.8], "bmd_fixture_score": [0.4, np.nan]})
    modalities = [
        ModalitySpec("gait_fixture", ("gait_fixture_score",), required=True),
        ModalitySpec("bmd_fixture", ("bmd_fixture_score",), required=False),
    ]
    early_available, early_available_meta = early_fuse_features(fixture, modalities, missing_policy="available_only")
    early_zero, early_zero_meta = early_fuse_features(fixture, modalities, missing_policy="impute_zero")
    late, late_meta = late_fuse_probabilities(
        {"gait_fixture": [0.2, 0.8], "bmd_fixture": [0.4, 0.6]},
        weights=config["late_fusion_weights"],
        missing_policy="error",
    )
    hybrid, hybrid_meta = hybrid_fuse_probabilities([0.3, 0.7], {"late_fixture": late}, early_weight=float(config["hybrid_early_weight"]))
    result = {
        "fixture_version": config["fixture_version"],
        "fixture_only": True,
        "clinical_performance_reported": False,
        "labels_present": False,
        "early_available_only": {"columns": list(early_available.columns), "metadata": early_available_meta},
        "early_impute_zero": {"values": early_zero.to_dict(orient="list"), "metadata": early_zero_meta},
        "late": {"probabilities": late.tolist(), "metadata": late_meta},
        "hybrid": {"probabilities": hybrid.tolist(), "metadata": hybrid_meta},
    }
    out = ROOT / "reports/member_b/member_b_fusion_fixture_v1_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
