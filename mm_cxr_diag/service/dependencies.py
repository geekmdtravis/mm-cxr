"""FastAPI dependency providers for the hierarchical predictor.

Keeping the predictor behind ``Depends(get_predictor)`` makes
``app.dependency_overrides[get_predictor]`` in tests trivial and leaves a
clean hook for later additions (API-key auth, request-scoped logging).
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from mm_cxr_diag.inference import HierarchicalPredictor


def get_predictor(request: Request) -> HierarchicalPredictor:
    """Return the process-wide ``HierarchicalPredictor``.

    The predictor is attached to ``app.state.predictor`` by the service's
    lifespan context manager. If checkpoint loading failed at startup,
    respond with 503 so the caller can retry against a healthy replica.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Predictor is not loaded. See /health for details.",
        )
    return predictor
