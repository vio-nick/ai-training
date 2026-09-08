import csv
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "data/public/nhanes_2017_2018/processed/feature_table_nhanes_exploratory_v1.csv"
SPLIT_PATH = ROOT / "data/public/nhanes_2017_2018/processed/split_nhanes_exploratory_v1.csv"


class NhanesPublicDataTests(unittest.TestCase):
    def test_public_feature_table_and_split_are_reproducible_shape(self) -> None:
        with FEATURE_PATH.open(encoding="utf-8", newline="") as handle:
            features = list(csv.DictReader(handle))
        with SPLIT_PATH.open(encoding="utf-8", newline="") as handle:
            splits = list(csv.DictReader(handle))
        self.assertEqual(len(features), 865)
        self.assertEqual(len(splits), 865)
        feature_ids = [row["participant_id"] for row in features]
        split_ids = [row["participant_id"] for row in splits]
        self.assertEqual(len(feature_ids), len(set(feature_ids)))
        self.assertEqual(set(feature_ids), set(split_ids))
        self.assertEqual({row["fracture_history"] for row in features}, {"0", "1"})
        self.assertEqual({row["split"] for row in splits}, {"train", "validation", "test"})

    def test_public_quality_metadata_matches_expected_source(self) -> None:
        quality = json.loads((ROOT / "data/public/nhanes_2017_2018/processed/nhanes_exploratory_v1_quality.json").read_text(encoding="utf-8"))
        self.assertEqual(quality["n_rows"], 865)
        self.assertEqual(quality["label_counts"], {"0": 681, "1": 184})
        self.assertEqual(quality["split_counts"], {"test": 173, "train": 519, "validation": 173})
        self.assertEqual(len(quality["source_sha256"]), 4)


if __name__ == "__main__":
    unittest.main()
