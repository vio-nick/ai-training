import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fall_prediction.release import ReleaseIntegrityError, load_model_release, save_model_release
from fall_prediction.training import fit_single_modality, select_threshold_on_validation


class ModelReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "configs" / "gstride_fall_v1.json").read_text(encoding="utf-8"))
        table = pd.read_csv(ROOT / cls.config["feature_table_path"])
        split = pd.read_csv(ROOT / cls.config["split_path"])
        cls.frame = table.merge(split, on=cls.config["id_column"], validate="one_to_one")
        cls.train = cls.frame.loc[cls.frame["split"] == "train"].copy()
        cls.validation = cls.frame.loc[cls.frame["split"] == "validation"].copy()

    def _fitted_model(self):
        model = fit_single_modality(self.train, self.config, "random_forest")
        return select_threshold_on_validation(model, self.validation)

    def test_release_loads_identical_model_without_refitting(self) -> None:
        model = self._fitted_model()
        record = self.frame.loc[0, ["participant_id", *self.config["feature_columns"]]].to_dict()
        expected = model.predict_record(record, data_version="gstride_fall_v1")
        with tempfile.TemporaryDirectory() as directory:
            manifest = save_model_release(
                model, directory, release_id="test-release",
                training={"fit_scope": "train", "new_inference_records_used_for_training": False},
            )
            loaded, loaded_manifest = load_model_release(directory)
            actual = loaded.predict_record(record, data_version="gstride_fall_v1")
        self.assertEqual(manifest["model_sha256"], loaded_manifest["model_sha256"])
        self.assertEqual(expected, actual)
        self.assertFalse(loaded_manifest["training"]["new_inference_records_used_for_training"])

    def test_release_rejects_a_modified_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save_model_release(
                self._fitted_model(), directory, release_id="test-release",
                training={"fit_scope": "train", "new_inference_records_used_for_training": False},
            )
            model_path = Path(directory) / "model.joblib"
            model_path.write_bytes(model_path.read_bytes() + b"altered")
            with self.assertRaises(ReleaseIntegrityError):
                load_model_release(directory)


if __name__ == "__main__":
    unittest.main()
