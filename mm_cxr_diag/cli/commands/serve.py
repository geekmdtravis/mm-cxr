"""`mm-cxr-diag serve`: start the FastAPI inference service (M5 wires this up)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

Device = Literal["cuda", "cpu", "auto"]


def serve(
    stage1_ckpt: Annotated[
        Path | None,
        typer.Option(
            "--stage1-ckpt",
            exists=True,
            readable=True,
            envvar="MM_CXR_STAGE1_CKPT",
            help="Stage 1 checkpoint (or MM_CXR_STAGE1_CKPT env var).",
        ),
    ] = None,
    stage2_ckpt: Annotated[
        Path | None,
        typer.Option(
            "--stage2-ckpt",
            exists=True,
            readable=True,
            envvar="MM_CXR_STAGE2_CKPT",
            help="Stage 2 checkpoint (or MM_CXR_STAGE2_CKPT env var).",
        ),
    ] = None,
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    reload: Annotated[
        bool,
        typer.Option(
            "--reload/--no-reload",
            help="Auto-reload on source changes (development only).",
        ),
    ] = False,
    device: Annotated[Device, typer.Option("--device")] = "auto",
) -> None:
    """Launch the hierarchical inference service."""
    try:
        import uvicorn

        from mm_cxr_diag.service.app import create_app
    except ImportError as e:
        typer.secho(
            f"`serve` requires the `serve` extra: pip install 'mm-cxr-diag[serve]'. "
            f"Details: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from e

    if stage1_ckpt is None or stage2_ckpt is None:
        typer.secho(
            "Both --stage1-ckpt and --stage2-ckpt are required (env vars "
            "MM_CXR_STAGE1_CKPT / MM_CXR_STAGE2_CKPT are accepted).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    app = create_app(stage1_ckpt=stage1_ckpt, stage2_ckpt=stage2_ckpt, device=device)
    uvicorn.run(app, host=host, port=port, workers=workers, reload=reload)
