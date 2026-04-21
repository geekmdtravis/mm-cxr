"""Shared fixtures for integration tests that need untrained checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from mm_cxr_diag.inference.persistence import save_model
from mm_cxr_diag.models import CXRModel, CXRModelConfig


def _save_untrained(
    path: Path, num_classes: int, backbone: str = "densenet121"
) -> Path:
    """Instantiate a fresh CXRModel and serialize its random weights.

    Tiny MLP head (hidden_dims=(16,)) keeps the test fast — we only care
    that load/forward/save work end-to-end, not about learned behavior.
    """
    config = CXRModelConfig(
        model=backbone,
        hidden_dims=(16,),
        dropout=0.1,
        num_classes=num_classes,
        tabular_features=4,
        freeze_backbone=False,
    )
    model = CXRModel(**config.as_dict())
    return save_model(model=model, config=config, file_path=path)


@pytest.fixture
def stage1_checkpoint(tmp_path: Path) -> Path:
    return _save_untrained(tmp_path / "stage1.pth", num_classes=1)


@pytest.fixture
def stage2_checkpoint(tmp_path: Path) -> Path:
    return _save_untrained(tmp_path / "stage2.pth", num_classes=14)
