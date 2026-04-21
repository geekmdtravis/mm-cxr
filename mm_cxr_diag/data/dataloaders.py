"""Factory for PyTorch DataLoaders backed by ``ChestXrayDataset``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader
from torchvision import transforms

from mm_cxr_diag.data.dataset import ChestXrayDataset, LabelMode

# Dataset statistics (recomputable via `mm-cxr-diag calculate-stats`).
DATASET_MEAN = 0.4995
DATASET_STD = 0.2480

NormalizationMode = Literal["imagenet", "dataset_specific", "none"]

logger = logging.getLogger(__name__)


def create_dataloader(
    clinical_data: Path,
    cxr_images_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    normalization_mode: NormalizationMode = "imagenet",
    label_mode: LabelMode = "multilabel",
    shuffle: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """Build a DataLoader for a train/val/test split."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer.")

    transform_list: list = [transforms.ToTensor()]
    if normalization_mode == "imagenet":
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    elif normalization_mode == "dataset_specific":
        transform_list.append(
            transforms.Normalize(mean=[DATASET_MEAN] * 3, std=[DATASET_STD] * 3)
        )
    elif normalization_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization_mode: {normalization_mode}")

    dataset = ChestXrayDataset(
        clinical_data=clinical_data,
        cxr_images_dir=cxr_images_dir,
        transform=transforms.Compose(transform_list),
        label_mode=label_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
