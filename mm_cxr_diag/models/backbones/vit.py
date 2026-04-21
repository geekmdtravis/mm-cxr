"""Vision Transformer backbones with a multimodal concat-MLP head."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from torchvision.models import (
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    vit_b_16,
    vit_b_32,
    vit_l_16,
)

from mm_cxr_diag.models.fusion import build_mm_classifier
from mm_cxr_diag.models.registry import register


class _ViTMM(nn.Module):
    """Shared implementation for ViT multimodal heads.

    The ViT backbone's ``.heads`` is replaced with ``Identity`` so that
    ``model(image)`` returns the pooled hidden state (shape
    ``(B, hidden_dim)``). The shared classifier then concatenates tabular
    features and projects to ``num_classes``.
    """

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

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

        image_features = self.model.hidden_dim
        self.model.heads = nn.Identity()
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


@register("vit_b_16")
class ViTB16MM(_ViTMM):
    """ViT-B/16 backbone with multimodal concat-MLP head."""

    def __init__(self, **kwargs):
        super().__init__(
            backbone_factory=vit_b_16,
            weights=ViT_B_16_Weights.IMAGENET1K_V1,
            **kwargs,
        )


@register("vit_b_32")
class ViTB32MM(_ViTMM):
    """ViT-B/32 backbone with multimodal concat-MLP head."""

    def __init__(self, **kwargs):
        super().__init__(
            backbone_factory=vit_b_32,
            weights=ViT_B_32_Weights.IMAGENET1K_V1,
            **kwargs,
        )


@register("vit_l_16")
class ViTL16MM(_ViTMM):
    """ViT-L/16 backbone with multimodal concat-MLP head."""

    def __init__(self, **kwargs):
        super().__init__(
            backbone_factory=vit_l_16,
            weights=ViT_L_16_Weights.IMAGENET1K_V1,
            **kwargs,
        )
