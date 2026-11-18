from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/GSTRIDE_IMU_to_Gait_Algorithm"))
import csv_adapter
from csv_adapter import CalibratedIMU, InputQualityError, _aggregate, run_gait_algorithm
from Gait_Parameters_Estimation import Gait_Parameters_Estimation


def test_training_summary_trims_each_measure_independently_and_uses_register_precision():
    gpe = Gait_Parameters_Estimation(5)
    gpe.tcycle = np.array([1.00, 1.01, 1.02, 1.03, 2.00])
    gpe.cadence = np.array([60.0, 59.4, 58.8, 58.3, 30.0])
    gpe.speed = np.array([0.45, 1.010, 1.011, 1.006, 0.99])
    gpe.StrideLengths = np.array([1.00, 1.02, 1.01, 0.98, 0.99])
    gpe.PTO = np.array([12.0, 12.2, 12.4, 12.1, 12.3])
    gpe.Pswing = np.array([30.0, 30.5, 31.0, 29.5, 30.2])
    gpe.PHS = np.array([18.0, 18.2, 18.4, 18.1, 18.3])
    gpe.PFF = np.array([40.0, 39.0, 38.0, 41.0, 40.5])
    features, counts, indices = _aggregate(gpe, np.ones(5, dtype=bool), minimum_feature_cycles=3)

    assert indices["tcycle"] == [0, 1, 2, 3]
    assert indices["speed"] == [0, 1, 2, 3, 4]
    assert counts["speed"] == 5
    assert counts["tcycle"] == 4
    assert features["step_speed_m_s"] == 0.893
    assert features["stride_time_cv_pct"] == pytest.approx(1.083743842364532)
    assert features["double_support_pct"] == 30.4
    assert features["swing_to_stance_ratio"] == pytest.approx(30.24 / (12.2 + 18.2 + 39.7))


def test_canonical_gait_run_passes_midfoot_flat_as_zupt_anchor(monkeypatch):
    n_samples = 80
    phases = np.array([[10, 20, 30, 40, 50]])
    mid_ff = np.array([45])
    captured = {}

    class FakeAHRS:
        def __init__(self, fs, count):
            self.angles = np.zeros((count, 3))

        def UpdateIMU(self, gyro, acc):
            pass

    def fake_detector(*args):
        empty = np.array([], dtype=int)
        return (
            np.zeros(n_samples), empty, np.array([1]), np.array([2]), empty, empty,
            mid_ff, empty, empty, empty, empty, empty, phases, np.zeros(1),
        )

    class FakeZUPT:
        def __init__(self, fs, count):
            assert count == len(mid_ff)
            self.Speed_G_corr = np.zeros((n_samples, 3))
            self.deltasXYZ = np.zeros((n_samples, 3))
            self.XYZ = np.zeros((n_samples, 3))

        def UPDATE_ZUPT(self, acc, gyr, anchors, still):
            captured["anchors"] = np.array(anchors)

    class FakeGPE:
        def __init__(self, count):
            assert count == len(mid_ff)

        def estimate(self, *args):
            captured["estimate_called"] = True

    monkeypatch.setattr(csv_adapter, "MahonyAHRS", FakeAHRS)
    monkeypatch.setattr(csv_adapter, "StepDetection_MultiCorrector_Pitch", fake_detector)
    monkeypatch.setattr(csv_adapter, "INS_ZUPT", FakeZUPT)
    monkeypatch.setattr(csv_adapter, "Gait_Parameters_Estimation", FakeGPE)

    imu = CalibratedIMU(
        timestamp_s=np.arange(n_samples) / 100,
        acc=np.zeros((n_samples, 3)),
        gyr=np.zeros((n_samples, 3)),
        fs=100.0,
        interpolated_samples=0,
        max_gap_s=0.01,
    )
    result = run_gait_algorithm(imu, "test.csv")

    np.testing.assert_array_equal(captured["anchors"], mid_ff)
    np.testing.assert_array_equal(result.zupt_anchors, mid_ff)
    assert captured["estimate_called"]


def test_gait_run_rejects_anchor_count_mismatch(monkeypatch):
    n_samples = 80
    phases = np.array([[10, 20, 30, 40, 50]])
    empty = np.array([], dtype=int)

    class FakeAHRS:
        def __init__(self, fs, count):
            self.angles = np.zeros((count, 3))

        def UpdateIMU(self, gyro, acc):
            pass

    def fake_detector(*args):
        return (np.zeros(n_samples), empty, np.array([1]), np.array([2]), empty, empty,
                empty, empty, empty, empty, empty, empty, phases, np.zeros(1))

    monkeypatch.setattr(csv_adapter, "MahonyAHRS", FakeAHRS)
    monkeypatch.setattr(csv_adapter, "StepDetection_MultiCorrector_Pitch", fake_detector)

    imu = CalibratedIMU(
        timestamp_s=np.arange(n_samples) / 100,
        acc=np.zeros((n_samples, 3)),
        gyr=np.zeros((n_samples, 3)),
        fs=100.0,
        interpolated_samples=0,
        max_gap_s=0.01,
    )
    with pytest.raises(InputQualityError, match="mid-foot-flat anchors"):
        run_gait_algorithm(imu, "mismatch.csv")
