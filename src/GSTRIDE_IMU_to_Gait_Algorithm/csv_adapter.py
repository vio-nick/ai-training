"""Adapter for calibrated GSTRIDE CSV files.

The original pipeline reads raw space-separated TXT files and applies a
beacon-specific MAT calibration. This module handles the already-calibrated
CSV contract used by data/test and then reuses the existing gait algorithm.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from Gait_Parameters_Estimation import Gait_Parameters_Estimation
from INS_ZUPT import INS_ZUPT
from MahonyAHRS import MahonyAHRS
from StepDetection_MultiCorrector_Pitch import StepDetection_MultiCorrector_Pitch


CSV_COLUMNS = (
    "timestamp_s",
    "accX[m/s2]",
    "accY[m/s2]",
    "accZ[m/s2]",
    "gyrX[rad/s]",
    "gyrY[rad/s]",
    "gyrZ[rad/s]",
)


class InputQualityError(ValueError):
    """Raised when a CSV cannot safely enter the gait pipeline."""


@dataclass
class CalibratedIMU:
    timestamp_s: np.ndarray
    acc: np.ndarray
    gyr: np.ndarray
    fs: float
    interpolated_samples: int
    max_gap_s: float


@dataclass
class GaitRun:
    input_path: str
    calibrated_imu: CalibratedIMU
    still: np.ndarray
    new_still: np.ndarray
    end_still: np.ndarray
    phases: np.ndarray
    gpe: Gait_Parameters_Estimation


def _read_numeric_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != CSV_COLUMNS:
            raise InputQualityError(
                f"{path}: expected columns {CSV_COLUMNS}, got {reader.fieldnames}"
            )
        rows: list[list[float]] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                values = [float(row[column]) for column in CSV_COLUMNS]
            except (TypeError, ValueError) as exc:
                raise InputQualityError(
                    f"{path}: non-numeric value at CSV line {line_number}"
                ) from exc
            rows.append(values)
    if not rows:
        raise InputQualityError(f"{path}: CSV has no data rows")
    matrix = np.asarray(rows, dtype=float)
    if not np.isfinite(matrix).all():
        raise InputQualityError(f"{path}: CSV contains NaN or infinite values")
    return matrix[:, 0], matrix[:, 1:4], matrix[:, 4:7]


def load_calibrated_csv(
    path: str | Path,
    *,
    target_fs: float = 100.0,
    max_interpolatable_gap_s: float = 0.03,
) -> CalibratedIMU:
    """Load an already-calibrated CSV and resample short gaps to target_fs."""

    source_path = Path(path)
    timestamp_s, acc, gyr = _read_numeric_csv(source_path)
    differences = np.diff(timestamp_s)
    if len(differences) == 0 or np.any(differences <= 0):
        raise InputQualityError(f"{source_path}: timestamps must be strictly increasing")
    nominal_dt = float(np.median(differences))
    if not np.isfinite(nominal_dt) or nominal_dt <= 0:
        raise InputQualityError(f"{source_path}: invalid timestamp spacing")
    max_gap = float(np.max(differences))
    if max_gap > max_interpolatable_gap_s:
        raise InputQualityError(
            f"{source_path}: gap {max_gap:.6f}s exceeds the allowed "
            f"interpolation limit {max_interpolatable_gap_s:.6f}s"
        )
    if target_fs <= 0:
        raise ValueError("target_fs must be positive")
    target_dt = 1.0 / target_fs
    uniform_t = np.arange(timestamp_s[0], timestamp_s[-1] + target_dt * 0.5, target_dt)
    uniform_acc = np.column_stack(
        [np.interp(uniform_t, timestamp_s, acc[:, axis]) for axis in range(3)]
    )
    uniform_gyr = np.column_stack(
        [np.interp(uniform_t, timestamp_s, gyr[:, axis]) for axis in range(3)]
    )
    return CalibratedIMU(
        timestamp_s=uniform_t,
        acc=uniform_acc,
        gyr=uniform_gyr,
        fs=float(target_fs),
        # A slightly shorter uniform grid can occur when timestamps drift by
        # a fraction of one sample; that is not a negative interpolation.
        interpolated_samples=max(0, int(len(uniform_t) - len(timestamp_s))),
        max_gap_s=max_gap,
    )


def run_gait_algorithm(imu: CalibratedIMU, path: str | Path) -> GaitRun:
    """Run the existing AHRS, event, ZUPT and parameter algorithms."""

    ahrs = MahonyAHRS(imu.fs, len(imu.acc))
    for index in range(len(imu.acc)):
        ahrs.UpdateIMU(imu.gyr[index], imu.acc[index])
    detected = StepDetection_MultiCorrector_Pitch(ahrs.angles, imu.gyr, 0, imu.fs)
    still, _, new_still, end_still, _, _, _, _, _, _, _, _, phases, pitch0 = detected
    if len(new_still) == 0 or len(end_still) == 0:
        raise InputQualityError(f"{path}: no qualifying initial static segment was detected")
    if len(phases) == 0:
        raise InputQualityError(f"{path}: no gait phases were detected")
    zupt = INS_ZUPT(imu.fs, len(phases))
    zupt.UPDATE_ZUPT(imu.acc, imu.gyr, phases[:, 0], still)
    gpe = Gait_Parameters_Estimation(len(phases))
    gpe.estimate(
        phases,
        zupt.Speed_G_corr,
        zupt.deltasXYZ,
        zupt.XYZ,
        ahrs.angles[:, 0],
        pitch0,
        imu.timestamp_s,
    )
    return GaitRun(
        input_path=str(path),
        calibrated_imu=imu,
        still=still,
        new_still=new_still,
        end_still=end_still,
        phases=phases,
        gpe=gpe,
    )


def _cycle_quality(run: GaitRun) -> tuple[np.ndarray, list[dict[str, Any]]]:
    gpe = run.gpe
    phases = np.asarray(run.phases, dtype=float)
    mask = np.ones(len(phases), dtype=bool)
    rejected: list[dict[str, Any]] = []
    arrays = {
        "tcycle": np.asarray(gpe.tcycle, dtype=float),
        "speed": np.asarray(gpe.speed, dtype=float),
        "stride_length": np.asarray(gpe.StrideLengths, dtype=float),
    }
    for index in range(len(phases)):
        reasons: list[str] = []
        row = phases[index]
        if not np.isfinite(row).all() or np.any(np.diff(row) <= 0):
            reasons.append("phase_order_invalid")
        if not 0.5 <= arrays["tcycle"][index] <= 3.0:
            reasons.append("cycle_time_out_of_range")
        if not 0.05 <= arrays["stride_length"][index] <= 2.0:
            reasons.append("stride_length_out_of_range")
        if not 0.0 <= arrays["speed"][index] <= 3.0:
            reasons.append("speed_out_of_range")
        if reasons:
            mask[index] = False
            rejected.append({"cycle_index": index, "reasons": reasons})
    return mask, rejected


def _aggregate(gpe: Gait_Parameters_Estimation, mask: np.ndarray) -> dict[str, float]:
    def mean(name: str) -> float:
        values = np.asarray(getattr(gpe, name), dtype=float)[mask]
        return float(np.mean(values))

    tcycle = np.asarray(gpe.tcycle, dtype=float)[mask]
    if len(tcycle) == 0:
        raise InputQualityError("no usable gait cycles remain after quality filtering")
    mean_swing = mean("Pswing")
    mean_cycle = float(np.mean(tcycle))
    return {
        "step_speed_m_s": mean("speed"),
        "cadence_strides_per_min": mean("cadence"),
        "stride_length_m": mean("StrideLengths"),
        "double_support_pct": mean("PTO") + mean("PHS"),
        "swing_to_stance_ratio": mean_swing / (100.0 - mean_swing),
        "stride_time_cv_pct": 100.0 * float(np.std(tcycle)) / mean_cycle,
    }


def summarize_gait(
    run: GaitRun,
    *,
    allow_quality_fail: bool = False,
    minimum_valid_cycles: int = 3,
    use_valid_cycles: bool = False,
) -> dict[str, Any]:
    """Return model features plus an auditable cycle quality result."""

    mask, rejected = _cycle_quality(run)
    valid_count = int(mask.sum())
    quality_ok = valid_count >= minimum_valid_cycles
    if not quality_ok and not allow_quality_fail:
        raise InputQualityError(
            f"{run.input_path}: only {valid_count} valid cycles; "
            f"at least {minimum_valid_cycles} are required"
        )
    if use_valid_cycles:
        if valid_count == 0:
            raise InputQualityError(f"{run.input_path}: no valid cycles remain")
        used_mask = mask
    else:
        used_mask = mask if quality_ok else np.ones(len(run.phases), dtype=bool)
    features = _aggregate(run.gpe, used_mask)
    return {
        "input_path": run.input_path,
        "rows_resampled": int(len(run.calibrated_imu.timestamp_s)),
        "interpolated_samples": run.calibrated_imu.interpolated_samples,
        "max_gap_s": run.calibrated_imu.max_gap_s,
        "detected_cycle_count": int(len(run.phases)),
        "valid_cycle_count": valid_count,
        "minimum_valid_cycles": minimum_valid_cycles,
        "quality_status": "pass" if quality_ok else "fail_allow_quality_override",
        "aggregation_mode": (
            "valid_cycles_only"
            if (use_valid_cycles or quality_ok)
            else "all_cycles_on_quality_override"
        ),
        "quality_rejected_cycles": rejected,
        "used_cycle_indices": np.flatnonzero(used_mask).astype(int).tolist(),
        "features": features,
    }
