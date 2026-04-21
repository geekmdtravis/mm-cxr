"""FastAPI application factory with predictor-loading lifespan."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI

from mm_cxr_diag import __version__
from mm_cxr_diag.inference import HierarchicalPredictor
from mm_cxr_diag.service.routes import router

Device = Literal["cuda", "cpu", "auto"]

logger = logging.getLogger(__name__)


def create_app(
    stage1_ckpt: str | Path | None = None,
    stage2_ckpt: str | Path | None = None,
    device: Device = "auto",
    stage1_threshold: float = 0.5,
    skip_stage2_below: float = 0.02,
) -> FastAPI:
    """Build the FastAPI app.

    Checkpoint paths may be passed in directly (used by the CLI ``serve``
    subcommand and by tests) or picked up from ``MM_CXR_STAGE1_CKPT`` /
    ``MM_CXR_STAGE2_CKPT`` at startup. Tests that don't need real
    checkpoints pass ``None`` for both and override the ``get_predictor``
    dependency via ``app.dependency_overrides``.
    """

    resolved_stage1 = stage1_ckpt or os.environ.get("MM_CXR_STAGE1_CKPT")
    resolved_stage2 = stage2_ckpt or os.environ.get("MM_CXR_STAGE2_CKPT")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.device = device
        app.state.predictor = None

        if resolved_stage1 and resolved_stage2:
            try:
                app.state.predictor = HierarchicalPredictor.from_checkpoints(
                    resolved_stage1,
                    resolved_stage2,
                    device=device,
                    stage1_threshold=stage1_threshold,
                    skip_stage2_below=skip_stage2_below,
                )
                logger.info(
                    "Loaded hierarchical predictor (stage1=%s, stage2=%s, device=%s)",
                    app.state.predictor.stage1.model_name,
                    app.state.predictor.stage2.model_name,
                    device,
                )
            except Exception:
                # Log and continue — /health will report the degraded state,
                # and dependency injection will return 503 for the predict
                # routes. Better than refusing to start the process.
                logger.exception("Failed to load hierarchical predictor")
        else:
            logger.warning(
                "No stage1/stage2 checkpoints configured; predict endpoints "
                "will return 503. Set MM_CXR_STAGE1_CKPT and MM_CXR_STAGE2_CKPT, "
                "or pass paths to create_app()."
            )

        yield
        # Nothing to tear down — torch model memory is reclaimed on exit.

    app = FastAPI(
        title="mm-cxr-diag",
        description="Two-stage multimodal chest X-ray classifier.",
        version=__version__,
        lifespan=lifespan,
    )
    app.include_router(router)
    return app
