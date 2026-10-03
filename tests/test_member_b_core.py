import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fall_prediction.contracts import ContractValidationError, InputContract, validate_feature_frame
from fall_prediction.explainability import exact_shapley_probability_contributions
from fall_prediction.fusion import ModalitySpec, early_fuse_features, hybrid_fuse_probabilities, late_fuse_probabilities
from fall_prediction.inference import classify_risk_level
from fall_prediction.training import fit_single_modality, select_threshold_on_validation


class MemberBCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "configs/gstride_fall_v1.json").read_text(encoding="utf-8"))
        feature = pd.read_csv(ROOT / cls.config["feature_table_path"])
        split = pd.read_csv(ROOT / cls.config["split_path"])
        frame = feature.merge(split, on=cls.config["id_column"], validate="one_to_one")
        cls.train = frame.loc[frame["split"] == "train"].copy()
        cls.validation = frame.loc[frame["split"] == "validation"].copy()
        cls.test = frame.loc[frame["split"] == "test"].copy()

    def test_contract_rejects_version_column_and_non_numeric_errors(self) -> None:
        contract = InputContract.gstride_fall_v1(include_label=True)
        ordered = validate_feature_frame(self.train, contract, data_version="gstride_fall_v1", require_label=True, require_unique_id=True)
        self.assertEqual(list(ordered.columns), self.config["feature_columns"])
        with self.assertRaises(ContractValidationError):
            validate_feature_frame(self.train, contract, data_version="gstride_fall_v2")
        malformed = self.test.drop(columns=[self.config["feature_columns"][0]])
        with self.assertRaises(ContractValidationError):
            validate_feature_frame(malformed, contract)
        malformed = self.test.copy()
        malformed[self.config["feature_columns"][0]] = malformed[self.config["feature_columns"][0]].astype(object)
        malformed.loc[malformed.index[0], self.config["feature_columns"][0]] = "not-a-number"
        with self.assertRaises(ContractValidationError):
            validate_feature_frame(malformed, contract)

    def test_training_inference_uses_only_features_and_never_refits(self) -> None:
        model = fit_single_modality(self.train, self.config, "logistic_regression")
        select_threshold_on_validation(model, self.validation)
        before = model.estimator.named_steps["imputer"].statistics_.copy()
        row = self.test.iloc[[0]].copy()
        first = model.predict_frame(row, data_version="gstride_fall_v1")
        changed_label = row.copy()
        changed_label["faller_last_year"] = 1 - changed_label["faller_last_year"]
        second = model.predict_frame(changed_label, data_version="gstride_fall_v1")
        np.testing.assert_allclose(first["predicted_probability"], second["predicted_probability"])
        np.testing.assert_allclose(before, model.estimator.named_steps["imputer"].statistics_)
        record = row.loc[:, ["participant_id", *self.config["feature_columns"]]].iloc[0].to_dict()
        response = model.predict_record(record, data_version="gstride_fall_v1")
        self.assertEqual(response["participant_id"], row.iloc[0]["participant_id"])
        self.assertNotIn("faller_last_year", response)
        self.assertAlmostEqual(response["predicted_probability_percent"], response["predicted_probability"] * 100)
        self.assertIn(response["risk_level"], {"low", "medium", "high"})
        self.assertIn(response["risk_level_display"], {"低风险", "中风险", "高风险"})
        self.assertEqual(response["feature_contribution_method"], "exact_shapley_probability")
        self.assertEqual(len(response["feature_contributions"]), 6)
        self.assertEqual(
            [item["feature"] for item in response["feature_contributions"]],
            self.config["feature_columns"],
        )
        self.assertTrue(
            all(item["direction"] in {"positive", "negative", "neutral"} for item in response["feature_contributions"])
        )
        self.assertAlmostEqual(
            response["feature_contribution_reconstructed_probability"],
            response["predicted_probability"],
            places=12,
        )
        self.assertAlmostEqual(
            response["feature_contribution_baseline_probability"]
            + sum(item["contribution_probability"] for item in response["feature_contributions"]),
            response["predicted_probability"],
            places=12,
        )
        with self.assertRaises(ContractValidationError):
            model.predict_record(record, data_version="gstride_fall_v2")

    def test_feature_contributions_support_missing_values_without_refitting(self) -> None:
        model = fit_single_modality(self.train, self.config, "random_forest")
        row = self.test.loc[:, ["participant_id", *self.config["feature_columns"]]].iloc[[0]].copy()
        row.loc[row.index[0], self.config["feature_columns"][0]] = np.nan
        before = model.estimator.named_steps["imputer"].statistics_.copy()
        result = model.predict_frame(row, data_version="gstride_fall_v1").iloc[0].to_dict()
        np.testing.assert_allclose(before, model.estimator.named_steps["imputer"].statistics_)
        self.assertIsNone(result["feature_contributions"][0]["value"])
        self.assertAlmostEqual(
            result["feature_contribution_reconstructed_probability"],
            result["predicted_probability"],
            places=12,
        )

    def test_exact_shapley_decomposition_is_batch_consistent(self) -> None:
        model = fit_single_modality(self.train, self.config, "random_forest")
        rows = self.test.loc[:, self.config["feature_columns"]].iloc[:2].copy()
        contributions, baseline, reconstructed = exact_shapley_probability_contributions(model.estimator, rows)
        self.assertEqual(contributions.shape, (2, 6))
        np.testing.assert_allclose(
            reconstructed,
            model.estimator.predict_proba(rows)[:, 1],
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(reconstructed, baseline + contributions.sum(axis=1), rtol=0, atol=1e-12)

    def test_risk_levels_use_the_requested_inclusive_middle_boundaries(self) -> None:
        levels = classify_risk_level([0.0, 0.299999, 0.3, 0.5, 0.7, 0.700001, 1.0])
        self.assertEqual(levels.tolist(), ["low", "low", "medium", "medium", "medium", "high", "high"])
        with self.assertRaises(ValueError):
            classify_risk_level([-0.01])
        with self.assertRaises(ValueError):
            classify_risk_level([1.01])

    def test_fusion_interfaces_have_explicit_missing_modality_behaviour(self) -> None:
        frame = pd.DataFrame({"gait": [0.2, 0.8], "bmd": [0.4, np.nan]})
        modalities = [ModalitySpec("gait", ("gait",), required=True), ModalitySpec("bmd", ("bmd",))]
        available, metadata = early_fuse_features(frame, modalities, missing_policy="available_only")
        self.assertEqual(list(available.columns), ["gait"])
        self.assertEqual(metadata["dropped_modalities"], ["bmd"])
        zero, _ = early_fuse_features(frame, modalities, missing_policy="impute_zero")
        self.assertEqual(float(zero.loc[1, "bmd"]), 0.0)
        with self.assertRaises(ContractValidationError):
            early_fuse_features(frame, modalities, missing_policy="error")
        late, late_meta = late_fuse_probabilities({"gait": [0.2, 0.8], "bmd": [0.4, 0.6]}, weights={"gait": 0.75, "bmd": 0.25})
        np.testing.assert_allclose(late, [0.25, 0.75])
        self.assertEqual(late_meta["fusion_type"], "late")
        hybrid, hybrid_meta = hybrid_fuse_probabilities([0.3, 0.7], {"late": late}, early_weight=0.5)
        np.testing.assert_allclose(hybrid, [0.275, 0.725])
        self.assertEqual(hybrid_meta["fusion_type"], "hybrid")


if __name__ == "__main__":
    unittest.main()
