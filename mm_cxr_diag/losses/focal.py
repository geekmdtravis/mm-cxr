"""Focal loss with class-frequency reweighting.

For Stage 2, ``reweight`` MUST be called with class counts computed on the
abnormal-only training subset — removing normal studies shifts base rates
(e.g. Hernia ~0.2% of all studies → ~0.5% of abnormal studies).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-label focal loss on BCE-with-logits.

    Args:
        weight: Optional per-class weight tensor of shape ``(num_classes,)``.
        gamma: Focusing parameter. Higher values down-weight easy examples
            more aggressively. ``gamma=0`` reduces to BCE.
    """

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = gamma
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        probs = torch.sigmoid(input)
        focal_weight = torch.pow(
            (1 - probs) * target + probs * (1 - target), self.gamma
        )
        loss = focal_weight * bce
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0).to(input.device)
        return loss.mean()


def reweight(cls_num_list: list[int] | tuple[int, ...], beta: float = 0.9999):
    """Effective-number-of-samples class weighting (Cui et al., 2019)."""
    counts = np.asarray(cls_num_list, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * len(counts)
    return torch.from_numpy(weights).float()
