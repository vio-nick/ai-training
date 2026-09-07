import json
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fall_prediction.gstride import FEATURE_COLUMNS, load_feature_table, make_stratified_split


class GStrideFallTests(unittest.TestCase):
    def test_source_maps_to_the_fixed_sensor_feature_contract(self) -> None:
        raw_path = ROOT / "data/public/gstride_v1/raw/Database_register.csv"
        frame = load_feature_table(raw_path)
        self.assertEqual(list(frame.columns), ["participant_id", "faller_last_year", *FEATURE_COLUMNS])
        self.assertEqual(len(frame), 163)
        self.assertEqual(frame["participant_id"].nunique(), 163)
        self.assertEqual(frame["faller_last_year"].value_counts().to_dict(), {1: 86, 0: 77})
        self.assertFalse(frame[FEATURE_COLUMNS].isna().any().any())
        self.assertTrue((frame["double_support_pct"] > 0).all())
        self.assertTrue((frame["swing_to_stance_ratio"] > 0).all())

    def test_split_is_reproducible_and_participant_level(self) -> None:
        frame = load_feature_table(ROOT / "data/public/gstride_v1/raw/Database_register.csv")
        first = make_stratified_split(frame)
        self.assertTrue(first.equals(make_stratified_split(frame)))
        self.assertEqual(set(first["participant_id"]), set(frame["participant_id"]))
        self.assertEqual(set(first["split"]), {"train", "validation", "test"})

    def test_build_and_train_outputs_have_expected_shape(self) -> None:
        subprocess.run([sys.executable, "scripts/build_gstride_fall_dataset.py"], cwd=ROOT, check=True, capture_output=True, text=True)
        subprocess.run([sys.executable, "scripts/run_gstride_fall_baseline.py"], cwd=ROOT, check=True, capture_output=True, text=True)
        metrics = json.loads((ROOT / "reports/baselines/gstride_fall_v1_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(set(metrics["models"]), {"logistic_regression", "random_forest"})
        for result in metrics["models"].values():
            self.assertEqual(result["test"]["n"], 33)
            self.assertGreaterEqual(result["test"]["auroc"], 0.0)
            self.assertLessEqual(result["test"]["auroc"], 1.0)
        predictions = pd.read_csv(ROOT / "reports/baselines/gstride_fall_v1_test_predictions.csv")
        self.assertEqual(len(predictions), 33)


if __name__ == "__main__":
    unittest.main()
