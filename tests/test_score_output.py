import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from score_gstride_fall_release import _write_output


class ScoreOutputTests(unittest.TestCase):
    def test_csv_keeps_feature_contributions_as_valid_json(self) -> None:
        results = [
            {
                "participant_id": "user-001",
                "predicted_probability": 0.6,
                "feature_contributions": [
                    {"feature": "step_speed_m_s", "direction": "positive", "contribution_probability": 0.1}
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "score.csv"
            _write_output(path, results)
            with path.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
        self.assertNotIn("feature_contributions", row)
        self.assertEqual(json.loads(row["feature_contributions_json"])[0]["feature"], "step_speed_m_s")


if __name__ == "__main__":
    unittest.main()
