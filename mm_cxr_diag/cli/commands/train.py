"""`mm-cxr-diag train`: train a Stage 1 or Stage 2 model from a config + CSVs."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import torch
import typer

from mm_cxr_diag.data import create_dataloader
from mm_cxr_diag.data.dataset import LabelMode
from mm_cxr_diag.models import CXRModelConfig
from mm_cxr_diag.training import train_model
from mm_cxr_diag.utils.logging import setup_logging

Stage = Literal[1, 2]
Device = Literal["cuda", "cpu", "auto"]


def _resolve_device(device: Device) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _label_mode_for(config: CXRModelConfig) -> LabelMode:
    """num_classes=1 → binary gate, otherwise multi-label."""
    return "binary" if config.num_classes == 1 else "multilabel"


def _make_run_dir(
    base: Path, backbone: str, num_classes: int, name: str | None
) -> Path:
    stage = 1 if num_classes == 1 else 2
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_name = name or stamp
    run_dir = base / f"stage{stage}" / backbone / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            readable=True,
            help="CXRModelConfig YAML (configs/stage{1,2}/<backbone>.yaml).",
        ),
    ],
    train_csv: Annotated[
        Path,
        typer.Option(
            "--train-csv", exists=True, readable=True, help="Preprocessed train CSV."
        ),
    ],
    val_csv: Annotated[
        Path,
        typer.Option(
            "--val-csv", exists=True, readable=True, help="Preprocessed val CSV."
        ),
    ],
    images_dir: Annotated[
        Path,
        typer.Option(
            "--images-dir",
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="Directory containing the CXR PNGs.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Base directory under which the run directory is created. "
            "Layout: {output_dir}/stage{1,2}/<backbone>/<run_name>/.",
        ),
    ] = Path("results/runs"),
    run_name: Annotated[
        str | None,
        typer.Option(
            "--run-name",
            help="Name for this run's subdir; defaults to a UTC timestamp.",
        ),
    ] = None,
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 50,
    lr: Annotated[float, typer.Option("--lr", min=0.0)] = 1e-5,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 32,
    num_workers: Annotated[int, typer.Option("--num-workers", min=0)] = 4,
    patience: Annotated[int, typer.Option("--patience", min=1)] = 5,
    focal_loss: Annotated[
        bool,
        typer.Option(
            "--focal-loss/--no-focal-loss",
            help="Use focal loss with class reweighting. Recommended for Stage 2; "
            "BCE is fine for Stage 1.",
        ),
    ] = False,
    focal_gamma: Annotated[float, typer.Option("--focal-gamma", min=0.0)] = 2.0,
    focal_beta: Annotated[
        float, typer.Option("--focal-beta", min=0.0, max=1.0)
    ] = 0.9999,
    device: Annotated[Device, typer.Option("--device")] = "auto",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run/--no-dry-run",
            help="Build model + one batch to verify wiring, then exit without training.",
        ),
    ] = False,
) -> None:
    """Train one model (Stage 1 binary gate or Stage 2 pathology head)."""
    setup_logging(log_level="info")

    model_config = CXRModelConfig.from_yaml(config)
    label_mode = _label_mode_for(model_config)
    resolved_device = _resolve_device(device)

    typer.echo(
        f"Training {model_config.model} with num_classes={model_config.num_classes} "
        f"(label_mode={label_mode}) on {resolved_device}"
    )

    run_dir = _make_run_dir(
        base=output_dir,
        backbone=model_config.model,
        num_classes=model_config.num_classes,
        name=run_name,
    )
    shutil.copy2(config, run_dir / "config.yaml")
    typer.echo(f"Run directory: {run_dir}")

    train_loader = create_dataloader(
        clinical_data=train_csv,
        cxr_images_dir=images_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        label_mode=label_mode,
        shuffle=True,
    )
    val_loader = create_dataloader(
        clinical_data=val_csv,
        cxr_images_dir=images_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        label_mode=label_mode,
    )

    if dry_run:
        from mm_cxr_diag.models import CXRModel

        model = CXRModel(**model_config.as_dict()).to(resolved_device)
        images, tabular, labels = next(iter(train_loader))
        with torch.no_grad():
            out = model(images.to(resolved_device), tabular.to(resolved_device))
        typer.echo(
            f"Dry run OK: model built, one batch forward returned {tuple(out.shape)}"
        )
        raise typer.Exit()

    best_val_loss, best_val_auc, mean_t, total_t, epochs_run, _model = train_model(
        model_config=model_config,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        patience=patience,
        focal_loss=focal_loss,
        focal_loss_rebal_beta=focal_beta,
        focal_loss_gamma=focal_gamma,
        plot_path=str(run_dir / "training_curves.png"),
        best_model_path=str(run_dir / "best_model.pth"),
        last_model_path=str(run_dir / "last_model.pth"),
        train_val_data_path=str(run_dir / "train_val_data.csv"),
        device=resolved_device,
    )

    typer.echo(
        f"Training complete. Best val_loss={best_val_loss:.4f} "
        f"best val_auc={best_val_auc:.4f} "
        f"epochs_run={epochs_run} "
        f"mean_epoch_time={mean_t:.1f}s total={total_t:.1f}s"
    )
    typer.echo(f"Checkpoints in: {run_dir}")
