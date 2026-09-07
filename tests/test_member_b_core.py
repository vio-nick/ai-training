import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fall_prediction.contracts import ContractValidationError, InputContract, validate_feature_frame
from fall_prediction.fusion import ModalitySpec, early_fuse_features, hybrid_fuse_probabilities, late_fuse_probabilities
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
        with self.assertRaises(ContractValidationError):
            model.predict_record(record, data_version="gstride_fall_v2")

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
