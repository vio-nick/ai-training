"""HTTP adapter for a fixed GSTRIDE model release (origin/memberB API integrated)."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import Body, FastAPI, HTTPException, Request
import pandas as pd
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from fall_prediction.contracts import ContractValidationError
from fall_prediction.release import load_model_release

ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = Path(
    os.getenv(
        "MODEL_RELEASE_DIR",
        str(
            ROOT
            / "artifacts/model_releases/gstride_fall_v1_random_forest_full_historical"
        ),
    )
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.manifest = None
    try:
        app.state.model, app.state.manifest = load_model_release(RELEASE_DIR)
    except Exception:
        logger.exception("Model release could not be loaded")
    yield


app = FastAPI(title="GSTRIDE 步态评估 API", version="0.2.0", lifespan=lifespan)
origins = [v.strip() for v in os.getenv("CORS_ORIGINS", "").split(",") if v.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    participant_id: str | None = Field(
        default=None, max_length=80, description="可选记录编号"
    )
    step_speed_m_s: float = Field(ge=0, description="步速 (m/s)")
    cadence_strides_per_min: float = Field(
        ge=0, description="步频 (strides/min，跨步/分钟，不是 steps/min)"
    )
    stride_length_m: float = Field(ge=0, description="跨步长 (m)")
    double_support_pct: float = Field(ge=0, le=100, description="双支撑比例 (%)")
    swing_to_stance_ratio: float = Field(ge=0, description="摆动/支撑比，无量纲")
    stride_time_cv_pct: float = Field(ge=0, description="跨步时间变异度 (%)")


class FeatureContribution(BaseModel):
    feature: str
    display_name: str
    unit: str
    value: float | None
    contribution_probability: float
    contribution_probability_percent: float
    direction: Literal["positive", "negative", "neutral"]
    direction_display: str


class Prediction(BaseModel):
    participant_id: str | None = None
    predicted_probability: float = Field(ge=0, le=1)
    predicted_probability_percent: float = Field(ge=0, le=100)
    risk_level: Literal["low", "medium", "high"]
    risk_level_display: str
    predicted_label: int
    decision_threshold: float
    data_version: str
    model_name: str
    feature_contributions: list[FeatureContribution]
    feature_contribution_method: str
    feature_contribution_baseline_probability: float
    feature_contribution_reconstructed_probability: float


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Do not echo raw input: non-finite numbers cannot be JSON encoded.
    return JSONResponse(
        status_code=422,
        content={
            "detail": "请检查输入字段、数值范围和单位。",
            "errors": [
                {"field": ".".join(map(str, e["loc"][1:])), "message": e["msg"]}
                for e in exc.errors()
            ],
        },
    )


@app.exception_handler(ContractValidationError)
async def contract_error(request: Request, exc: ContractValidationError):
    return JSONResponse(
        status_code=422, content={"detail": "输入不符合模型要求，请检查六项步态参数。"}
    )


def ensure_model():
    if getattr(app.state, "model", None) is None:
        raise HTTPException(503, "模型尚未就绪，请先生成模型发布包并重启服务。")
    return app.state.model, app.state.manifest


def score(records):
    model, manifest = ensure_model()
    try:
        return model.predict_frame(
            pd.DataFrame([r.model_dump() for r in records]),
            data_version=manifest["data_version"],
        ).to_dict(orient="records")
    except ContractValidationError:
        raise
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(500, "评分暂时失败，请稍后重试。")


@app.get("/")
def root():
    return {"name": "GSTRIDE 步态评估 API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    _, manifest = ensure_model()
    return {
        "status": "ok",
        "model_name": manifest["model_name"],
        "data_version": manifest["data_version"],
        "release_id": manifest["release_id"],
    }


@app.post("/predict", response_model=Prediction)
def predict(request: PredictRequest):
    return score([request])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(requests: list[PredictRequest] = Body(min_length=1, max_length=100)):
    return score(requests)
