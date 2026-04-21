"""Training loop for a single-stage ``CXRModel``.

No module-level configuration side effects. Paths and dataloaders are
passed in by the caller (CLI / notebook / test harness). Focal loss class
weights, when enabled, are recomputed from the training split the caller
provided — critical for Stage 2 where rebalancing must use the
abnormal-only class counts.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from mm_cxr_diag.inference.persistence import save_model
from mm_cxr_diag.losses import FocalLoss, reweight
from mm_cxr_diag.models import CXRModel, CXRModelConfig

logger = logging.getLogger(__name__)

Device = torch.device | Literal["cuda", "cpu"]


def _train_one_epoch(
    model: CXRModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: Device,
    pb_prefix: str,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    all_preds: list = []
    all_labels: list = []

    pbar = tqdm(loader, desc=pb_prefix)
    for images, tabular, labels in pbar:
        images = images.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, tabular)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader)
    epoch_auc = _safe_macro_auc(np.array(all_labels), np.array(all_preds))
    return epoch_loss, epoch_auc


def _validate_one_epoch(
    model: CXRModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: Device,
    pb_prefix: str,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    all_preds: list = []
    all_labels: list = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=pb_prefix)
        for images, tabular, labels in pbar:
            images = images.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device)

            outputs = model(images, tabular)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader)
    epoch_auc = _safe_macro_auc(np.array(all_labels), np.array(all_preds))
    return epoch_loss, epoch_auc


def _safe_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro AUC that handles binary (shape (N,1)) and multilabel alike.

    ``roc_auc_score`` wants shape ``(N,)`` for binary and ``(N, C)`` for
    multilabel. It also raises when a column has a single class; we fall
    back to NaN for those.
    """
    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true[:, 0]
        y_pred = y_pred[:, 0]
    try:
        return float(roc_auc_score(y_true, y_pred, average="macro"))
    except ValueError:
        return float("nan")


def _plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_aucs: list[float],
    val_aucs: list[float],
    title_prefix: str,
    save_path: str,
    figsize: tuple[int, int] = (12, 5),
) -> None:
    plt.figure(figsize=figsize)
    out = Path(save_path)
    if out.suffix != ".png":
        out = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} - Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_aucs, label="Train AUC-ROC")
    plt.plot(val_aucs, label="Val AUC-ROC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC-ROC")
    plt.title(f"{title_prefix} - AUC-ROC")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def train_model(
    model_config: CXRModelConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    criterion: nn.Module | None = None,
    optimizer: optim.Optimizer | None = None,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    epochs: int = 50,
    lr: float = 1e-5,
    patience: int = 5,
    focal_loss: bool = False,
    focal_loss_rebal_beta: float = 0.9999,
    focal_loss_gamma: float = 2.0,
    plot_path: str = "results/plots/training_curves.png",
    best_model_path: str = "results/models/best_model.pth",
    last_model_path: str = "results/models/last_model.pth",
    train_val_data_path: str = "results/train_val_data.csv",
    device: Device = "cuda",
) -> tuple[float, float, float, float, int, CXRModel]:
    """Train a ``CXRModel`` on pre-built dataloaders.

    Returns ``(best_val_loss, best_val_auc, mean_epoch_time, total_time,
    num_epochs_run, model)``.
    """
    model = CXRModel(**model_config.as_dict()).to(device)

    if focal_loss:
        logger.info(
            "Rebalancing and configuring Focal Loss with beta=%s gamma=%s",
            focal_loss_rebal_beta,
            focal_loss_gamma,
        )
        all_labels = np.array([labels for _, _, labels in train_loader.dataset])
        class_counts = np.sum(all_labels, axis=0).tolist()
        criterion = FocalLoss(
            weight=reweight(class_counts, beta=focal_loss_rebal_beta),
            gamma=focal_loss_gamma,
        )
        logger.info("Focal Loss weights: %s", criterion.weight)
    elif criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=patience
        )

    best_val_loss = float("inf")
    best_val_auc = 0.0
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_aucs: list[float] = []
    val_aucs: list[float] = []
    epoch_train_times: list[float] = []
    num_epochs_run = 0

    for epoch in range(epochs):
        epoch_display = epoch + 1
        logger.info("Epoch %d/%d", epoch_display, epochs)

        t0 = time.time()
        train_loss, train_auc = _train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            pb_prefix=f"T-{epoch_display}",
        )
        epoch_train_times.append(time.time() - t0)
        train_losses.append(train_loss)
        train_aucs.append(train_auc)

        val_loss, val_auc = _validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            pb_prefix=f"V-{epoch_display}",
        )
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        logger.info(
            "Epoch %d Train Loss=%.4f AUC=%.4f | Val Loss=%.4f AUC=%.4f",
            epoch_display,
            train_loss,
            train_auc,
            val_loss,
            val_auc,
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(
                "Epoch %d: saving best model to %s", epoch_display, best_model_path
            )
            save_model(model=model, config=model_config, file_path=best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if val_auc > best_val_auc:
            best_val_auc = val_auc

        if patience_counter >= patience:
            logger.info(
                "Epoch %d: early stopping after %d epochs of no improvement",
                epoch_display,
                patience,
            )
            break
        num_epochs_run += 1

    logger.info("Saving final model to %s", last_model_path)
    save_model(model=model, config=model_config, file_path=last_model_path)
    _plot_training_curves(
        train_losses,
        val_losses,
        train_aucs,
        val_aucs,
        title_prefix="Training Curves",
        save_path=plot_path,
    )
    tv = pd.DataFrame(
        {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "train_auc": train_aucs,
            "val_auc": val_aucs,
        }
    )
    if not train_val_data_path.endswith(".csv"):
        base, _ = os.path.splitext(train_val_data_path)
        train_val_data_path = base + ".csv"
    Path(train_val_data_path).parent.mkdir(parents=True, exist_ok=True)
    tv.to_csv(train_val_data_path, index=False)
    logger.info("Training complete.")

    return (
        best_val_loss,
        best_val_auc,
        float(np.mean(epoch_train_times)) if epoch_train_times else 0.0,
        float(np.sum(epoch_train_times)),
        num_epochs_run,
        model,
    )
