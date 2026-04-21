"""Wrapper model + config for the registry-backed backbones.

``CXRModel`` delegates to a concrete backbone via ``build_model(name, **kw)``
and exposes a uniform ``forward(image, tabular)`` signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import yaml

from mm_cxr_diag.models.registry import MODELS, build_model

SupportedModels = Literal[
    "densenet121",
    "densenet201",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
]


@dataclass
class CXRModelConfig:
    """Serializable configuration for a ``CXRModel``.

    Attributes:
        model: Backbone name. Must be a key of
            :data:`mm_cxr_diag.models.registry.MODELS` (the five MM backbones).
        hidden_dims: Hidden layer widths for the classifier MLP head. ``None``
            means no hidden layers (linear classifier).
        dropout: Dropout rate between classifier layers.
        num_classes: 14 for Stage 2 (pathologies), 1 for Stage 1 (abnormality).
        tabular_features: Number of clinical features fed into the head.
        freeze_backbone: Freeze the backbone's convolutional/transformer
            parameters (classifier head still trains).
    """

    model: SupportedModels
    hidden_dims: tuple[int, ...] | list[int] | None = None
    dropout: float = 0.2
    num_classes: int = 14
    tabular_features: int = 4
    freeze_backbone: bool = False

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> CXRModelConfig:
        """Load a configuration from a YAML file."""
        with open(config_path) as file:
            config = yaml.safe_load(file)
        if "hidden_dims" in config and isinstance(config["hidden_dims"], list):
            config["hidden_dims"] = tuple(config["hidden_dims"])
        return cls(**config)

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "num_classes": self.num_classes,
            "tabular_features": self.tabular_features,
            "freeze_backbone": self.freeze_backbone,
        }

    def __repr__(self) -> str:
        return f"CXRModelConfig({self.as_dict()})"


class CXRModel(nn.Module):
    """Uniform wrapper over the registered multimodal backbones."""

    def __init__(
        self,
        model: SupportedModels,
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        dropout: float = 0.2,
        num_classes: int = 14,
        tabular_features: int = 4,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if model not in MODELS:
            known = ", ".join(sorted(MODELS))
            raise ValueError(f"Model '{model}' is not supported. Known: {known}")
        self.model_name = model
        self.model = build_model(
            model,
            hidden_dims=hidden_dims if hidden_dims is not None else (),
            dropout=dropout,
            num_classes=num_classes,
            tabular_features=tabular_features,
            freeze_backbone=freeze_backbone,
        )

    def forward(
        self, img_batch: torch.Tensor, tabular_batch: torch.Tensor
    ) -> torch.Tensor:
        return self.model(img_batch, tabular_batch)
