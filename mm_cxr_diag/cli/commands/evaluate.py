"""`mm-cxr-diag evaluate`: evaluate a single-stage checkpoint or a two-stage pair."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
import typer

from mm_cxr_diag.data import PATHOLOGY_LABELS, create_dataloader
from mm_cxr_diag.data.dataset import LabelMode
from mm_cxr_diag.data.labels import CLASS_LABELS
from mm_cxr_diag.inference import HierarchicalPredictor
from mm_cxr_diag.inference.persistence import load_model
from mm_cxr_diag.training import evaluate_model, print_evaluation_results, run_inference

Stage = Literal["1", "2", "both"]
Device = Literal["cuda", "cpu", "auto"]


def _resolve_device(device: Device) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _label_mode_for(num_classes: int) -> LabelMode:
    if num_classes == 1:
        return "binary"
    if num_classes == 15:
        return "multilabel_legacy15"
    return "multilabel"


def _class_labels_for(num_classes: int) -> tuple[str, ...]:
    if num_classes == 1:
        return ("Abnormal",)
    if num_classes == 15:
        return CLASS_LABELS
    return PATHOLOGY_LABELS


def evaluate(
    stage: Annotated[Stage, typer.Option("--stage", help="1, 2, or both.")],
    test_csv: Annotated[
        Path,
        typer.Option(
            "--test-csv", exists=True, readable=True, help="Preprocessed test CSV."
        ),
    ],
    images_dir: Annotated[
        Path,
        typer.Option(
            "--images-dir",
            exists=True,
            file_okay=False,
            help="Directory containing the CXR PNGs.",
        ),
    ],
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            help="Checkpoint path for single-stage evaluation (required when "
            "--stage is 1 or 2).",
        ),
    ] = None,
    stage1_ckpt: Annotated[
        Path | None,
        typer.Option("--stage1-ckpt", help="Stage 1 checkpoint (required for 'both')."),
    ] = None,
    stage2_ckpt: Annotated[
        Path | None,
        typer.Option("--stage2-ckpt", help="Stage 2 checkpoint (required for 'both')."),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory to write the text report into. Created if missing.",
        ),
    ] = Path("results/evaluation"),
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 32,
    num_workers: Annotated[int, typer.Option("--num-workers", min=0)] = 4,
    device: Annotated[Device, typer.Option("--device")] = "auto",
) -> None:
    """Run evaluation and write a text report. Metrics: AUC/F1/confusion matrices."""
    resolved_device = _resolve_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stage in ("1", "2"):
        if checkpoint is None:
            raise typer.BadParameter(
                "--checkpoint is required when --stage is 1 or 2.",
                param_hint="--checkpoint",
            )
        model = load_model(checkpoint).to(resolved_device)
        num_classes = model.model.classifier[-1].out_features
        label_mode = _label_mode_for(num_classes)
        class_labels = _class_labels_for(num_classes)

        loader = create_dataloader(
            clinical_data=test_csv,
            cxr_images_dir=images_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            label_mode=label_mode,
        )

        typer.echo(f"Running stage {stage} inference on {len(loader.dataset)} samples")
        preds, labels = run_inference(model, loader, device=resolved_device)
        results = evaluate_model(preds, labels, class_labels=class_labels)
        out_path = output_dir / f"stage{stage}_evaluation.txt"
        print_evaluation_results(results, class_labels, save_path=out_path)
        typer.echo(f"Report saved to {out_path}")
        return

    # stage == "both"
    if stage1_ckpt is None or stage2_ckpt is None:
        raise typer.BadParameter(
            "--stage1-ckpt and --stage2-ckpt are required when --stage is 'both'.",
            param_hint="--stage1-ckpt / --stage2-ckpt",
        )

    predictor = HierarchicalPredictor.from_checkpoints(
        stage1_ckpt, stage2_ckpt, device=resolved_device, skip_stage2_below=0.0
    )
    loader = create_dataloader(
        clinical_data=test_csv,
        cxr_images_dir=images_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        label_mode="multilabel",
    )
    typer.echo(
        f"Running two-stage inference on {len(loader.dataset)} samples "
        f"(multiplicative combination)"
    )

    combined_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for images, tabular, labels in loader:
        probs = predictor.predict_batch(images, tabular)
        combined_probs.append(probs)
        all_labels.append(labels.numpy())

    preds = np.concatenate(combined_probs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    results = evaluate_model(preds, y, class_labels=PATHOLOGY_LABELS)
    out_path = output_dir / "hierarchical_evaluation.txt"
    print_evaluation_results(results, PATHOLOGY_LABELS, save_path=out_path)
    typer.echo(f"Report saved to {out_path}")
