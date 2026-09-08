import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from member_a.data_pipeline import EXPECTED_COLUMNS, generate_synthetic_rows, quality_summary, stratified_split, validate_rows, validate_split


class DataPipelineTests(unittest.TestCase):
    def test_generation_is_deterministic_and_valid(self) -> None:
        first = generate_synthetic_rows()
        second = generate_synthetic_rows()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 240)
        self.assertEqual(set(first[0]), set(EXPECTED_COLUMNS))
        summary = quality_summary(first)
        self.assertEqual(summary["n_rows"], 240)
        self.assertEqual(sum(summary["label_counts"].values()), 240)

    def test_split_is_participant_level_and_reproducible(self) -> None:
        rows = generate_synthetic_rows()
        first = stratified_split(rows)
        second = stratified_split(rows)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(rows))
        self.assertEqual(len({row["participant_id"] for row in first}), len(rows))
        self.assertEqual({row["split"] for row in first}, {"train", "validation", "test"})
        validate_split(rows, first)

    def test_duplicate_participant_id_is_rejected(self) -> None:
        rows = generate_synthetic_rows()
        rows[1]["participant_id"] = rows[0]["participant_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_rows(rows)

    def test_incomplete_split_is_rejected(self) -> None:
        rows = generate_synthetic_rows()
        split_rows = stratified_split(rows)[:-1]
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_split(rows, split_rows)


if __name__ == "__main__":
    unittest.main()
