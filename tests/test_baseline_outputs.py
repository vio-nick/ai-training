import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BaselineOutputTests(unittest.TestCase):
    def test_config_keeps_preprocessing_and_evaluation_boundaries(self) -> None:
        config = json.loads((ROOT / "configs" / "baseline_synthetic_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(config["preprocessing"]["fit_split"], "train")
        self.assertEqual(config["model_selection_split"], "validation")
        self.assertEqual(config["final_evaluation_split"], "test")
        self.assertEqual(config["threshold_selection"], "validation_max_f1")

    def test_baseline_outputs_have_two_models_and_test_predictions(self) -> None:
        metrics = json.loads((ROOT / "reports" / "baselines" / "synthetic_v1_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(set(metrics["models"]), {"logistic_regression", "random_forest"})
        for model in metrics["models"].values():
            self.assertEqual(model["validation"]["n"], 48)
            self.assertEqual(model["test"]["n"], 48)
            self.assertGreaterEqual(model["test"]["auroc"], 0.0)
            self.assertLessEqual(model["test"]["auroc"], 1.0)
        prediction_lines = (ROOT / "reports" / "baselines" / "synthetic_v1_test_predictions.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(prediction_lines), 97)


if __name__ == "__main__":
    unittest.main()
