"""`mm-cxr-diag predict`: hierarchical inference on one image + tabular features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

import typer
from PIL import Image

from mm_cxr_diag.inference import HierarchicalPredictor
from mm_cxr_diag.inference.transforms import TABULAR_FEATURE_ORDER

OutputFormat = Literal["json", "table"]
Device = Literal["cuda", "cpu", "auto"]


def _load_tabular(
    tabular_path: Path | None, tabular_json: str | None
) -> dict[str, float]:
    if tabular_path is None and tabular_json is None:
        raise typer.BadParameter(
            "Provide either --tabular (path to JSON file) or --tabular-json (inline).",
            param_hint="--tabular / --tabular-json",
        )
    if tabular_path is not None and tabular_json is not None:
        raise typer.BadParameter(
            "Pass --tabular OR --tabular-json, not both.",
            param_hint="--tabular / --tabular-json",
        )
    raw = (
        json.loads(tabular_path.read_text())
        if tabular_path
        else json.loads(tabular_json)
    )
    if not isinstance(raw, dict):
        raise typer.BadParameter(
            f"Tabular input must be a JSON object keyed by {TABULAR_FEATURE_ORDER}, "
            f"got {type(raw).__name__}.",
            param_hint="--tabular / --tabular-json",
        )
    return {k: float(v) for k, v in raw.items()}


def _format_table(result_dict: dict) -> str:
    lines = [
        f"abnormal_prob        : {result_dict['abnormal_prob']:.4f}",
        f"abnormal             : {result_dict['abnormal']}",
        f"stage1_threshold     : {result_dict['stage1_threshold']:.4f}",
        f"no_finding_prob      : {result_dict['no_finding_prob']:.4f}",
        f"skipped_stage2       : {result_dict['skipped_stage2']}",
        f"stage1_model         : {result_dict['model_versions']['stage1']}",
        f"stage2_model         : {result_dict['model_versions']['stage2']}",
        f"latency_ms (total)   : {result_dict['latency_ms'].get('total', 0.0):.1f}",
    ]
    if result_dict["pathologies"] is not None:
        lines.append("")
        lines.append("pathology probabilities (P(abnormal) * P(path|abnormal)):")
        for label in result_dict["pathology_labels"]:
            p = result_dict["pathologies"][label]
            t = result_dict["pathology_thresholds"][label]
            flag = "*" if p >= t else " "
            lines.append(f"  {flag} {label:<20s} p={p:.4f}  threshold={t:.4f}")
    return "\n".join(lines)


def predict(
    stage1_ckpt: Annotated[
        Path,
        typer.Option(
            "--stage1-ckpt",
            exists=True,
            readable=True,
            help="Stage 1 (abnormality gate) checkpoint.",
        ),
    ],
    stage2_ckpt: Annotated[
        Path,
        typer.Option(
            "--stage2-ckpt",
            exists=True,
            readable=True,
            help="Stage 2 (pathology classifier) checkpoint.",
        ),
    ],
    image: Annotated[
        Path,
        typer.Option(
            "--image", exists=True, readable=True, help="Path to a chest X-ray image."
        ),
    ],
    tabular: Annotated[
        Path | None,
        typer.Option(
            "--tabular",
            exists=True,
            readable=True,
            help="Path to a JSON file with clinical features.",
        ),
    ] = None,
    tabular_json: Annotated[
        str | None,
        typer.Option(
            "--tabular-json",
            help='Inline JSON with clinical features, e.g. \'{"patientAge": 0.6, '
            '"patientGender": 1, "viewPosition": 0, "followUpNumber": 0.1}\'.',
        ),
    ] = None,
    stage1_threshold: Annotated[
        float,
        typer.Option(
            "--stage1-threshold",
            min=0.0,
            max=1.0,
            help="Threshold for reporting abnormal (does NOT gate Stage 2).",
        ),
    ] = 0.5,
    skip_stage2_below: Annotated[
        float,
        typer.Option(
            "--skip-stage2-below",
            min=0.0,
            max=0.99,
            help="Short-circuit Stage 2 when P(abnormal) is below this.",
        ),
    ] = 0.02,
    device: Annotated[Device, typer.Option("--device")] = "auto",
    output: Annotated[
        OutputFormat,
        typer.Option("--output", help="Output format."),
    ] = "json",
) -> None:
    """Run Stage 1 + Stage 2 and print the combined prediction."""
    tabular_dict = _load_tabular(tabular, tabular_json)

    predictor = HierarchicalPredictor.from_checkpoints(
        stage1_ckpt,
        stage2_ckpt,
        device=device,
        stage1_threshold=stage1_threshold,
        skip_stage2_below=skip_stage2_below,
    )
    pil = Image.open(image)
    result = predictor.predict(pil, tabular_dict).to_dict()

    if output == "json":
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(_format_table(result))
