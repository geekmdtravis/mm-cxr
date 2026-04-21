"""Preprocessing for single-image inference (CLI and REST service).

The training DataLoader applies ``ToTensor + Normalize`` inside
``create_dataloader``. At inference we usually have a single raw PIL image
from an API request, so we replicate those same steps plus an explicit
resize to 224x224 — the input size all five torchvision backbones expect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from PIL import Image
from torchvision import transforms

NormalizationMode = Literal["imagenet", "dataset_specific", "none"]

# Must mirror the values in ``mm_cxr_diag.data.dataloaders``.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_DATASET_MEAN = [0.4995] * 3
_DATASET_STD = [0.2480] * 3

# Canonical order for tabular features — must match
# ``ChestXrayDataset.__getitem__`` and the multimodal head's input layout.
TABULAR_FEATURE_ORDER: tuple[str, ...] = (
    "patientAge",
    "patientGender",
    "viewPosition",
    "followUpNumber",
)


def default_preprocess(
    image_size: int = 224,
    normalization_mode: NormalizationMode = "imagenet",
) -> transforms.Compose:
    """Standard single-image preprocessing pipeline."""
    steps: list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if normalization_mode == "imagenet":
        steps.append(transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD))
    elif normalization_mode == "dataset_specific":
        steps.append(transforms.Normalize(mean=_DATASET_MEAN, std=_DATASET_STD))
    elif normalization_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization_mode: {normalization_mode}")
    return transforms.Compose(steps)


def prepare_image(
    image: Image.Image | torch.Tensor,
    transform: transforms.Compose | None = None,
) -> torch.Tensor:
    """Convert a PIL image (or pre-tensorized image) to a ``(1, 3, H, W)`` batch.

    Raw ``torch.Tensor`` inputs are passed through unchanged aside from
    shape normalization — the caller is assumed to have applied their own
    preprocessing.
    """
    if isinstance(image, Image.Image):
        if transform is None:
            transform = default_preprocess()
        image = image.convert("RGB")
        tensor = transform(image)
    elif isinstance(image, torch.Tensor):
        tensor = image
    else:
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 4:
        raise ValueError(f"Image tensor must be 3- or 4-D, got {tensor.dim()}-D")
    return tensor


def tabular_to_tensor(
    tabular: Mapping[str, float] | Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Convert tabular input into a ``(1, 4)`` float tensor.

    Accepts a dict keyed by :data:`TABULAR_FEATURE_ORDER`, a 4-element
    sequence in that order, or a pre-built tensor.
    """
    if isinstance(tabular, torch.Tensor):
        t = tabular.float()
    elif isinstance(tabular, Mapping):
        try:
            values = [float(tabular[k]) for k in TABULAR_FEATURE_ORDER]
        except KeyError as e:
            raise KeyError(
                f"Tabular dict missing required key: {e.args[0]!r}. "
                f"Expected keys: {TABULAR_FEATURE_ORDER}"
            ) from e
        t = torch.tensor(values, dtype=torch.float32)
    elif isinstance(tabular, Sequence):
        if len(tabular) != len(TABULAR_FEATURE_ORDER):
            raise ValueError(
                f"Tabular sequence must have {len(TABULAR_FEATURE_ORDER)} "
                f"elements ({TABULAR_FEATURE_ORDER}); got {len(tabular)}"
            )
        t = torch.tensor([float(v) for v in tabular], dtype=torch.float32)
    else:
        raise TypeError(f"Unsupported tabular type: {type(tabular).__name__}")

    if t.dim() == 1:
        t = t.unsqueeze(0)
    elif t.dim() != 2:
        raise ValueError(f"Tabular tensor must be 1- or 2-D, got {t.dim()}-D")
    return t
