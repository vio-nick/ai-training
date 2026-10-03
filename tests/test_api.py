"""HTTP integration against a temporary real trained model release."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

api = importlib.import_module("api.main")
ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = dict(
    step_speed_m_s=0.82,
    cadence_strides_per_min=52,
    stride_length_m=0.95,
    double_support_pct=24,
    swing_to_stance_ratio=0.31,
    stride_time_cv_pct=8.5,
)


class APITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/release_gstride_fall_model.py"),
                "--output-dir",
                cls.temp.name,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            cwd=ROOT,
        )
        cls.release_patch = patch.object(api, "RELEASE_DIR", Path(cls.temp.name))
        cls.release_patch.start()
        cls.client_context = TestClient(api.app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        cls.release_patch.stop()
        cls.temp.cleanup()

    def test_real_prediction_and_contribution_sum(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        response = self.client.post("/predict", json=PAYLOAD)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(len(result["feature_contributions"]), 6)
        total = result["feature_contribution_baseline_probability"] + sum(
            c["contribution_probability"] for c in result["feature_contributions"]
        )
        self.assertAlmostEqual(total, result["predicted_probability"])
        self.assertIsNone(result["participant_id"])
        batch = self.client.post("/predict/batch", json=[PAYLOAD, PAYLOAD])
        self.assertEqual(batch.status_code, 200)
        self.assertEqual(batch.json()[0], result)

    def test_invalid_input_is_json_422(self):
        for payload in [
            {},
            dict(PAYLOAD, step_speed_m_s=-1),
            dict(PAYLOAD, double_support_pct=101),
            dict(PAYLOAD, step_speed_m_s="Infinity"),
            dict(PAYLOAD, unexpected=1),
        ]:
            response = self.client.post("/predict", json=payload)
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn("detail", response.json())
        for payload in [[], [PAYLOAD] * 101]:
            self.assertEqual(
                self.client.post("/predict/batch", json=payload).status_code, 422
            )

    def test_missing_model(self):
        with patch.object(api.app.state, "model", None):
            self.assertEqual(self.client.get("/health").status_code, 503)
            self.assertEqual(
                self.client.post("/predict", json=PAYLOAD).status_code, 503
            )

    def test_response_contract(self):
        schema = self.client.get("/openapi.json").json()
        self.assertEqual(
            schema["paths"]["/predict"]["post"]["responses"]["200"]["content"][
                "application/json"
            ]["schema"]["$ref"],
            "#/components/schemas/Prediction",
        )
