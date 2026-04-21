"""DenseNet backbones with a multimodal concat-MLP head."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    DenseNet201_Weights,
    densenet121,
    densenet201,
)

from mm_cxr_diag.models.fusion import build_mm_classifier
from mm_cxr_diag.models.registry import register


class _DenseNetMM(nn.Module):
    """Shared implementation for DenseNet121/201 multimodal heads."""

    def __init__(
        self,
        backbone_factory,
        weights,
        hidden_dims: Iterable[int] | None = (512, 256, 128),
        dropout: float = 0.2,
        num_classes: int = 14,
        tabular_features: int = 4,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.model = backbone_factory(weights=weights)
        image_features = self.model.classifier.in_features

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

        self.model.classifier = nn.Identity()
        self.classifier = build_mm_classifier(
            image_features=image_features,
            tabular_features=tabular_features,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, tabular_data: torch.Tensor) -> torch.Tensor:
        image_features = self.model(x)
        return self.classifier(torch.cat([image_features, tabular_data], dim=1))


@register("densenet121")
class DenseNet121MM(_DenseNetMM):
    """DenseNet-121 backbone with multimodal concat-MLP head."""

    def __init__(self, **kwargs):
        super().__init__(
            backbone_factory=densenet121,
            weights=DenseNet121_Weights.IMAGENET1K_V1,
            **kwargs,
        )


@register("densenet201")
class DenseNet201MM(_DenseNetMM):
    """DenseNet-201 backbone with multimodal concat-MLP head."""

    def __init__(self, **kwargs):
        super().__init__(
            backbone_factory=densenet201,
            weights=DenseNet201_Weights.IMAGENET1K_V1,
            **kwargs,
        )
