"""Focused tests for the model registry + ``build_model`` factory."""

from __future__ import annotations

import pytest
import torch
from mm_cxr_diag.models import MODELS, build_model, register

EXPECTED_BACKBONES = {
    "densenet121",
    "densenet201",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
}


def test_registry_is_complete():
    assert set(MODELS) == EXPECTED_BACKBONES


@pytest.mark.parametrize("name", sorted(EXPECTED_BACKBONES))
def test_build_model_forward_multilabel(name: str):
    model = build_model(
        name, hidden_dims=(16,), dropout=0.1, num_classes=14, tabular_features=4
    )
    model.eval()
    out = model(torch.randn(2, 3, 224, 224), torch.randn(2, 4))
    assert out.shape == (2, 14)


@pytest.mark.parametrize("name", sorted(EXPECTED_BACKBONES))
def test_build_model_forward_binary(name: str):
    model = build_model(
        name, hidden_dims=(16,), dropout=0.1, num_classes=1, tabular_features=4
    )
    model.eval()
    out = model(torch.randn(2, 3, 224, 224), torch.randn(2, 4))
    assert out.shape == (2, 1)


def test_unknown_backbone_raises():
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_model("does-not-exist")


def test_register_rejects_duplicate():
    """Re-registering an already-registered name is a programmer error."""
    with pytest.raises(ValueError, match="already registered"):

        @register("densenet121")
        class _Dup:
            pass
