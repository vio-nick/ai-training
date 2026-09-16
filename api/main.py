"""HTTP API wrapper for the released GSTRIDE faller inference model.

This module exposes a stable REST API so that frontend or other consumers can
score records without importing the project internals.  The model release is
loaded once at startup and is never modified by incoming requests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fall_prediction.inference import score_records  # noqa: E402
from fall_prediction.release import (  # noqa: E402
    ReleaseIntegrityError,
    load_model_release,
)

app = FastAPI(
    title="GSTRIDE 跌倒风险推理 API",
    description=(
        "研究原型接口：输入六项步态特征，输出回顾性跌倒者识别分数。 "
        "不能用于骨折风险预测、临床诊断或治疗决策。"
    ),
    version="0.1.0",
)

DEFAULT_RELEASE_DIR = str(
    ROOT / "artifacts" / "model_releases" / "gstride_fall_v1_random_forest_full_historical"
)
RELEASE_DIR = Path(os.getenv("MODEL_RELEASE_DIR", DEFAULT_RELEASE_DIR))

_model: Any | None = None
_manifest: dict[str, Any] | None = None
_load_error: str | None = None


def _load_model() -> None:
    global _model, _manifest, _load_error
    try:
        _model, _manifest = load_model_release(RELEASE_DIR)
        _load_error = None
    except (ReleaseIntegrityError, FileNotFoundError, OSError, ValueError) as exc:
        _model = None
        _manifest = None
        _load_error = str(exc)


_load_model()


def _ensure_model_loaded() -> None:
    if _model is None or _manifest is None:
        detail = f"模型未能加载: {_load_error}"
        raise HTTPException(status_code=503, detail=detail)


class PredictRequest(BaseModel):
    """Single record scoring request."""

    participant_id: str | None = Field(default=None, description="可选样本标识")
    step_speed_m_s: float = Field(..., ge=0, description="步速 (m/s)")
    cadence_strides_per_min: float = Field(..., ge=0, description="步频 (步/分钟)")
    stride_length_m: float = Field(..., ge=0, description="步长 (m)")
    double_support_pct: float = Field(..., ge=0, le=100, description="双支撑相百分比 (0-100)")
    swing_to_stance_ratio: float = Field(..., ge=0, description="摆动/支撑比")
    stride_time_cv_pct: float = Field(..., ge=0, description="步时变异系数百分比")


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "GSTRIDE 跌倒风险推理 API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "notice": "研究原型，不能用于临床诊断或治疗决策。",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    _ensure_model_loaded()
    return {
        "status": "ok",
        "model_name": _manifest["model_name"],
        "data_version": _manifest["data_version"],
        "release_id": _manifest.get("release_id"),
        "release_dir": str(RELEASE_DIR),
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    """Score a single record."""
    _ensure_model_loaded()
    record = request.model_dump()
    result = _model.predict_record(record, data_version=_manifest["data_version"])
    return result


@app.post("/predict/batch")
def predict_batch(requests: list[PredictRequest]) -> list[dict[str, Any]]:
    """Score multiple records at once."""
    _ensure_model_loaded()
    if not requests:
        raise HTTPException(status_code=400, detail="请求体不能为空列表")
    records = [r.model_dump() for r in requests]
    return score_records(_model, records, data_version=_manifest["data_version"])
