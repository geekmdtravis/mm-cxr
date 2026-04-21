"""Integration tests for SingleStagePredictor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from mm_cxr_diag.inference import SingleStagePredictor
from PIL import Image


def _make_pil(size: int = 64) -> Image.Image:
    arr = np.random.default_rng(0).integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_loads_checkpoint(stage1_checkpoint: Path):
    p = SingleStagePredictor(stage1_checkpoint, device="cpu")
    assert p.num_classes == 1
    assert p.model_name == "densenet121"
    assert p.checkpoint_path == stage1_checkpoint


def test_predict_single_binary(stage1_checkpoint: Path):
    p = SingleStagePredictor(stage1_checkpoint, device="cpu")
    probs = p.predict(
        _make_pil(),
        {"patientAge": 0.5, "patientGender": 0, "viewPosition": 1, "followUpNumber": 0},
    )
    assert probs.shape == (1,)
    assert 0.0 <= float(probs[0]) <= 1.0


def test_predict_single_multilabel(stage2_checkpoint: Path):
    p = SingleStagePredictor(stage2_checkpoint, device="cpu")
    probs = p.predict(_make_pil(), [0.5, 0, 1, 0])
    assert probs.shape == (14,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_predict_batch(stage2_checkpoint: Path):
    p = SingleStagePredictor(stage2_checkpoint, device="cpu")
    images = torch.randn(3, 3, 224, 224)
    tabular = torch.randn(3, 4)
    probs = p.predict_batch(images, tabular)
    assert probs.shape == (3, 14)


def test_tabular_dict_missing_key_raises(stage1_checkpoint: Path):
    p = SingleStagePredictor(stage1_checkpoint, device="cpu")
    with pytest.raises(KeyError):
        p.predict(_make_pil(), {"patientAge": 0.5})


def test_batch_size_mismatch_raises(stage1_checkpoint: Path):
    p = SingleStagePredictor(stage1_checkpoint, device="cpu")
    with pytest.raises(ValueError, match="Batch size mismatch"):
        p.predict_batch(torch.randn(2, 3, 224, 224), torch.randn(3, 4))


def test_nonexistent_checkpoint_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        SingleStagePredictor(tmp_path / "does_not_exist.pth", device="cpu")
