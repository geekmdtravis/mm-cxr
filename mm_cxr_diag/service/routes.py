"""FastAPI route handlers."""

from __future__ import annotations

import io
import json
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from mm_cxr_diag import __version__
from mm_cxr_diag.inference import HierarchicalPredictor
from mm_cxr_diag.service.dependencies import get_predictor
from mm_cxr_diag.service.schemas import (
    HealthResponse,
    PredictionResponse,
    Stage1Response,
    Stage2Response,
    TabularFeatures,
    VersionResponse,
)

router = APIRouter()


def _parse_tabular(tabular_json: str) -> TabularFeatures:
    try:
        payload = json.loads(tabular_json)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422, detail=f"tabular is not valid JSON: {e.msg}"
        ) from e
    try:
        return TabularFeatures.model_validate(payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def _open_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=422, detail="image upload is empty")
    try:
        return Image.open(io.BytesIO(data))
    except UnidentifiedImageError as e:
        raise HTTPException(
            status_code=422, detail="image is not a recognizable format"
        ) from e


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Always returns 200 while the process is alive — the fields indicate
    whether the predictor actually loaded. Orchestrators can check
    ``stage1_loaded and stage2_loaded`` to decide readiness."""
    predictor: HierarchicalPredictor | None = getattr(
        request.app.state, "predictor", None
    )
    return HealthResponse(
        status="ok",
        stage1_loaded=predictor is not None,
        stage2_loaded=predictor is not None,
        device=str(getattr(request.app.state, "device", "unknown")),
    )


@router.get("/version", response_model=VersionResponse)
def version(
    predictor: HierarchicalPredictor = Depends(get_predictor),
) -> VersionResponse:
    return VersionResponse(
        version=__version__,
        stage1_model=predictor.stage1.model_name,
        stage2_model=predictor.stage2.model_name,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(
    image: UploadFile = File(..., description="Chest X-ray image (PNG/JPEG)."),
    tabular: str = Form(
        ...,
        description='JSON string of clinical features, e.g. {"patientAge": 0.6,...}',
    ),
    predictor: HierarchicalPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Full hierarchical inference — abnormality gate then pathology
    classifier — with the multiplicative combination baked in."""
    tabular_obj = _parse_tabular(tabular)
    pil = _open_image(image)
    result = predictor.predict(pil, tabular_obj.model_dump())
    return PredictionResponse.model_validate(result.to_dict())


@router.post("/predict/stage1", response_model=Stage1Response)
def predict_stage1(
    image: UploadFile = File(...),
    tabular: str = Form(...),
    predictor: HierarchicalPredictor = Depends(get_predictor),
) -> Stage1Response:
    """Run only the abnormality gate. Useful for triage dashboards."""
    tabular_obj = _parse_tabular(tabular)
    pil = _open_image(image)

    from mm_cxr_diag.inference.transforms import prepare_image, tabular_to_tensor

    img_tensor = prepare_image(pil, transform=predictor.stage1._transform)
    tab_tensor = tabular_to_tensor(tabular_obj.model_dump())

    t0 = time.perf_counter()
    probs = predictor.stage1.predict_batch(img_tensor, tab_tensor)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    abnormal_prob = float(probs[0, 0])
    return Stage1Response(
        abnormal_prob=abnormal_prob,
        abnormal=abnormal_prob >= 0.5,
        stage1_threshold=0.5,
        no_finding_prob=1.0 - abnormal_prob,
        stage1_model=predictor.stage1.model_name,
        latency_ms={"stage1": elapsed_ms, "total": elapsed_ms},
    )


@router.post("/predict/stage2", response_model=Stage2Response)
def predict_stage2(
    image: UploadFile = File(...),
    tabular: str = Form(...),
    predictor: HierarchicalPredictor = Depends(get_predictor),
) -> Stage2Response:
    """Run only the pathology classifier — raw conditional probabilities.

    Use ``/predict`` for operational inference. This endpoint exists for
    debugging/backfilling and explicitly returns ``P(pathology|abnormal)``
    without the Stage 1 scaling.
    """
    tabular_obj = _parse_tabular(tabular)
    pil = _open_image(image)

    from mm_cxr_diag.inference.transforms import prepare_image, tabular_to_tensor

    img_tensor = prepare_image(pil, transform=predictor.stage2._transform)
    tab_tensor = tabular_to_tensor(tabular_obj.model_dump())

    t0 = time.perf_counter()
    probs = predictor.stage2.predict_batch(img_tensor, tab_tensor)[0]
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    pathologies = {
        lbl: float(p) for lbl, p in zip(predictor.pathology_labels, probs, strict=True)
    }
    thresholds = {lbl: 0.5 for lbl in predictor.pathology_labels}
    return Stage2Response(
        pathologies=pathologies,
        pathology_thresholds=thresholds,
        pathology_labels=list(predictor.pathology_labels),
        stage2_model=predictor.stage2.model_name,
        latency_ms={"stage2": elapsed_ms, "total": elapsed_ms},
    )
