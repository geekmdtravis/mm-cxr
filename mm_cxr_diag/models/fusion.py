"""Shared multimodal classifier head.

All backbones use the same concat-MLP: concatenate image features with
tabular features, then a sequence of ``Linear → BN → ReLU → Dropout``
layers ending in a ``Linear`` to ``num_classes``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn


def build_mm_classifier(
    image_features: int,
    tabular_features: int,
    num_classes: int,
    hidden_dims: Iterable[int] | None,
    dropout: float,
) -> nn.Sequential:
    """Build the shared multimodal classifier head."""
    dims = tuple(hidden_dims) if hidden_dims is not None else ()

    layers: list[nn.Module] = []
    prev = image_features + tabular_features
    for h in dims:
        layers.extend(
            [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        prev = h
    layers.append(nn.Linear(prev, num_classes))
    return nn.Sequential(*layers)
