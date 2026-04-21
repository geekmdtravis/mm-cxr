"""Pydantic request/response schemas for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TabularFeatures(BaseModel):
    """Clinical tabular features submitted alongside a CXR image.

    Values must already be on the same scale the training pipeline used
    (min-max normalized age/follow-up, 0/1 encoded gender/view position).
    """

    model_config = ConfigDict(extra="forbid")

    patientAge: float = Field(..., description="Normalized [0,1] age.")
    patientGender: float = Field(..., description="0=M, 1=F.")
    viewPosition: float = Field(..., description="0=PA, 1=AP.")
    followUpNumber: float = Field(..., description="Normalized [0,1] follow-up index.")


class PredictionResponse(BaseModel):
    """Full two-stage response body returned by ``POST /predict``."""

    abnormal_prob: float
    abnormal: bool
    stage1_threshold: float
    no_finding_prob: float
    pathologies: dict[str, float] | None
    pathology_thresholds: dict[str, float] | None
    pathology_labels: list[str]
    skipped_stage2: bool
    model_versions: dict[str, str]
    latency_ms: dict[str, float]


class Stage1Response(BaseModel):
    """Stage-1-only response — the abnormality gate in isolation."""

    abnormal_prob: float
    abnormal: bool
    stage1_threshold: float
    no_finding_prob: float
    stage1_model: str
    latency_ms: dict[str, float]


class Stage2Response(BaseModel):
    """Stage-2-only response — raw pathology probabilities, not combined.

    These are ``P(pathology | abnormal)`` — the Stage 2 output directly,
    without Stage 1's scaling. Use ``/predict`` for the calibrated
    hierarchy.
    """

    pathologies: dict[str, float]
    pathology_thresholds: dict[str, float]
    pathology_labels: list[str]
    stage2_model: str
    latency_ms: dict[str, float]


class HealthResponse(BaseModel):
    """Liveness + predictor-load status."""

    status: str
    stage1_loaded: bool
    stage2_loaded: bool
    device: str


class VersionResponse(BaseModel):
    """Package + model checkpoint identity."""

    version: str
    stage1_model: str
    stage2_model: str
