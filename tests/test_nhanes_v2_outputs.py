import csv
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "data/public/nhanes_2013_2018/processed/feature_table_nhanes_exploratory_v2.csv"
SPLIT_PATH = ROOT / "data/public/nhanes_2013_2018/processed/split_nhanes_exploratory_v2.csv"
METRICS_PATH = ROOT / "reports/baselines/nhanes_exploratory_v2_metrics.json"


class NhanesV2OutputTests(unittest.TestCase):
    def test_feature_and_split_contract(self) -> None:
        with FEATURE_PATH.open(encoding="utf-8", newline="") as handle:
            features = list(csv.DictReader(handle))
        with SPLIT_PATH.open(encoding="utf-8", newline="") as handle:
            splits = list(csv.DictReader(handle))
        self.assertEqual(len(features), 1771)
        self.assertEqual(len(splits), 1771)
        self.assertEqual(len({row["participant_id"] for row in features}), 1771)
        self.assertEqual({row["participant_id"] for row in features}, {row["participant_id"] for row in splits})
        self.assertEqual({row["fracture_history"] for row in features}, {"0", "1"})
        self.assertEqual({row["split"] for row in splits}, {"train", "validation", "test"})

    def test_expanded_features_improve_repeated_cv_over_base(self) -> None:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        base = metrics["ablation"]["base"]
        expanded = metrics["ablation"]["expanded"]
        self.assertEqual(expanded["n_repeats"], 3)
        self.assertGreater(expanded["auroc"], base["auroc"])
        self.assertGreater(expanded["auroc"], 0.60)
        self.assertLess(expanded["brier_score"], base["brier_score"])


if __name__ == "__main__":
    unittest.main()
