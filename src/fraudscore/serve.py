"""FastAPI service: POST /score, GET /health, GET /model-info.

The artifact (calibrated model + frozen t* + cost params) is loaded once at startup.
/score prices each transaction with the expected-cost rule:

    review  <=>  fraud_probability * amount >= c_review

Strict request contract (pydantic v2): extra fields, negative amounts, non-finite
values, or a V vector that isn't exactly 28 floats all return 422 with field detail.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from fraudscore.features import RAW_FEATURE_COLUMNS

DEFAULT_ARTIFACT_DIR = Path("artifacts")

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    time: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    v: Annotated[list[FiniteFloat], Field(min_length=28, max_length=28)]


class ScoreResponse(BaseModel):
    fraud_probability: float
    expected_fraud_cost: float
    decision: str  # "approve" | "review"
    decision_rule: str
    c_review: float
    model_version: str
    scored_at: str


def _finite_safe(obj):
    """Replace non-finite floats so a 422 echoing NaN/inf input can still serialize."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, list | tuple):
        return [_finite_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _finite_safe(v) for k, v in obj.items()}
    return obj


def _request_to_frame(req: ScoreRequest) -> pd.DataFrame:
    row = {"Time": req.time, "Amount": req.amount}
    row.update({f"V{i + 1}": req.v[i] for i in range(28)})
    return pd.DataFrame([row], columns=RAW_FEATURE_COLUMNS)


def create_app(artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> FastAPI:
    artifact_dir = Path(artifact_dir)
    artifact = joblib.load(artifact_dir / "model.joblib")
    model = artifact["main"].model
    c_review = float(artifact["c_review"])
    version = str(artifact["version"])
    card_path = artifact_dir / "model-card.json"

    app = FastAPI(title="fraudscore", version=version)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Field-level detail per the contract; NaN/inf in the rejected input would
        # break the default handler's JSON encoding, so sanitize first.
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(_finite_safe(exc.errors()))},
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        p_hat = float(model.predict_proba(_request_to_frame(req))[0])
        expected_fraud_cost = p_hat * req.amount
        return ScoreResponse(
            fraud_probability=p_hat,
            expected_fraud_cost=expected_fraud_cost,
            decision="review" if expected_fraud_cost >= c_review else "approve",
            decision_rule="expected_cost",
            c_review=c_review,
            model_version=version,
            scored_at=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": True}

    @app.get("/model-info")
    def model_info() -> dict:
        # The model card, so both decision rules are inspectable (incl. baseline t*).
        return json.loads(card_path.read_text())

    return app
