import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fall_prediction.evaluation import calibration_summary, error_cases, subgroup_template, threshold_impact


class MemberBEvaluationTests(unittest.TestCase):
    def test_calibration_contains_reliability_bins(self) -> None:
        result = calibration_summary([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9], n_bins=2)
        self.assertEqual(result["n"], 4)
        self.assertEqual(len(result["bins"]), 2)
        self.assertGreaterEqual(result["brier_score"], 0.0)

    def test_threshold_impact_and_error_cases(self) -> None:
        impact = threshold_impact([0, 1, 1], [0.9, 0.4, 0.8], [0.5])
        self.assertEqual(impact[0]["fp"], 1)
        self.assertEqual(impact[0]["fn"], 1)
        frame = pd.DataFrame({"id": ["a", "b", "c"], "label": [0, 1, 1], "p": [0.9, 0.4, 0.8]})
        errors = error_cases(frame, id_column="id", label_column="label", probability_column="p", threshold=0.5)
        self.assertEqual(set(errors["error_type"]), {"false_positive", "false_negative"})

    def test_subgroup_template_records_unavailable_contract_field(self) -> None:
        frame = pd.DataFrame({"label": [0, 1], "p": [0.1, 0.9]})
        result = subgroup_template(frame, label_column="label", probability_column="p", threshold=0.5, subgroup_columns=["sex"])
        self.assertEqual(result.iloc[0]["status"], "unavailable_in_evaluation_frame")


if __name__ == "__main__":
    unittest.main()
