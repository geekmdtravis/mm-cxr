"""Integration tests for HierarchicalPredictor.

Covers:
- Checkpoint loading via ``from_checkpoints``.
- Full ``predict(image, tabular)`` output schema.
- The multiplicative combination ``P(path_i) = sigmoid(s1) * sigmoid(s2_i)``.
- The Stage-2 short-circuit when ``P(abnormal) < skip_stage2_below``.
- Validation that Stage 1 must have ``num_classes=1``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mm_cxr_diag.data.labels import PATHOLOGY_LABELS
from mm_cxr_diag.inference import (
    HierarchicalPredictor,
    SingleStagePredictor,
)
from PIL import Image


def _make_pil(size: int = 64) -> Image.Image:
    arr = np.random.default_rng(1).integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_from_checkpoints_builds(stage1_checkpoint: Path, stage2_checkpoint: Path):
    p = HierarchicalPredictor.from_checkpoints(
        stage1_checkpoint, stage2_checkpoint, device="cpu"
    )
    assert p.stage1.num_classes == 1
    assert p.stage2.num_classes == len(PATHOLOGY_LABELS)
    assert p.pathology_labels == PATHOLOGY_LABELS


def test_predict_full_schema(stage1_checkpoint: Path, stage2_checkpoint: Path):
    p = HierarchicalPredictor.from_checkpoints(
        stage1_checkpoint,
        stage2_checkpoint,
        device="cpu",
        skip_stage2_below=0.0,  # force Stage 2 to run
    )
    result = p.predict(
        _make_pil(),
        {"patientAge": 0.5, "patientGender": 0, "viewPosition": 1, "followUpNumber": 0},
    )
    assert 0.0 <= result.abnormal_prob <= 1.0
    assert isinstance(result.abnormal, bool)
    assert result.stage1_threshold == 0.5
    assert result.no_finding_prob == pytest.approx(1.0 - result.abnormal_prob)
    assert not result.skipped_stage2
    assert result.pathologies is not None
    assert set(result.pathologies) == set(PATHOLOGY_LABELS)
    assert all(0.0 <= v <= 1.0 for v in result.pathologies.values())
    assert result.pathology_thresholds is not None
    assert result.stage1_model == "densenet121"
    assert result.stage2_model == "densenet121"
    assert {"stage1", "stage2", "total"}.issubset(result.latency_ms)

    d = result.to_dict()
    assert d["model_versions"] == {"stage1": "densenet121", "stage2": "densenet121"}
    assert len(d["pathology_labels"]) == len(PATHOLOGY_LABELS)


def test_multiplicative_combination(
    stage1_checkpoint: Path, stage2_checkpoint: Path, monkeypatch
):
    """Combined prob must equal P(abnormal) * P(pathology | abnormal)."""
    p = HierarchicalPredictor.from_checkpoints(
        stage1_checkpoint, stage2_checkpoint, device="cpu", skip_stage2_below=0.0
    )

    # Force deterministic stage outputs
    fake_stage1 = np.array([[0.8]], dtype=np.float32)
    fake_stage2 = np.full((1, 14), 0.5, dtype=np.float32)
    monkeypatch.setattr(p.stage1, "predict_batch", lambda images, tabular: fake_stage1)
    monkeypatch.setattr(p.stage2, "predict_batch", lambda images, tabular: fake_stage2)

    result = p.predict(_make_pil(), [0.5, 0, 1, 0])

    assert result.abnormal_prob == pytest.approx(0.8)
    assert result.pathologies is not None
    # 0.8 * 0.5 = 0.4 for every pathology
    for label in PATHOLOGY_LABELS:
        assert result.pathologies[label] == pytest.approx(0.4, rel=1e-5)


def test_stage2_short_circuited_when_normal(
    stage1_checkpoint: Path, stage2_checkpoint: Path, monkeypatch
):
    p = HierarchicalPredictor.from_checkpoints(
        stage1_checkpoint, stage2_checkpoint, device="cpu", skip_stage2_below=0.02
    )
    monkeypatch.setattr(
        p.stage1,
        "predict_batch",
        lambda images, tabular: np.array([[0.01]], dtype=np.float32),
    )

    stage2_called = {"n": 0}

    def _tripwire(*a, **kw):
        stage2_called["n"] += 1
        return np.zeros((1, 14), dtype=np.float32)

    monkeypatch.setattr(p.stage2, "predict_batch", _tripwire)

    result = p.predict(_make_pil(), [0, 0, 0, 0])
    assert result.skipped_stage2 is True
    assert result.pathologies is None
    assert result.pathology_thresholds is None
    assert result.latency_ms["stage2"] == 0.0
    assert stage2_called["n"] == 0


def test_batch_predict(stage1_checkpoint: Path, stage2_checkpoint: Path):
    import torch

    p = HierarchicalPredictor.from_checkpoints(
        stage1_checkpoint, stage2_checkpoint, device="cpu"
    )
    combined = p.predict_batch(torch.randn(4, 3, 224, 224), torch.randn(4, 4))
    assert combined.shape == (4, 14)
    assert np.all((combined >= 0.0) & (combined <= 1.0))


def test_stage1_num_classes_validated(stage2_checkpoint: Path):
    """Stage 1 predictor must have num_classes=1."""
    wrong = SingleStagePredictor(stage2_checkpoint, device="cpu")  # has 14
    correct = SingleStagePredictor(stage2_checkpoint, device="cpu")
    with pytest.raises(ValueError, match="num_classes=1"):
        HierarchicalPredictor(stage1=wrong, stage2=correct)


def test_stage2_num_classes_must_match_labels(
    stage1_checkpoint: Path, stage2_checkpoint: Path
):
    s1 = SingleStagePredictor(stage1_checkpoint, device="cpu")
    s2 = SingleStagePredictor(stage2_checkpoint, device="cpu")
    with pytest.raises(ValueError, match="does not match"):
        HierarchicalPredictor(stage1=s1, stage2=s2, pathology_labels=("A", "B"))


def test_custom_thresholds_round_trip(stage1_checkpoint: Path, stage2_checkpoint: Path):
    s1 = SingleStagePredictor(stage1_checkpoint, device="cpu")
    s2 = SingleStagePredictor(stage2_checkpoint, device="cpu")
    custom = {lbl: 0.3 for lbl in PATHOLOGY_LABELS}
    p = HierarchicalPredictor(
        stage1=s1, stage2=s2, stage2_thresholds=custom, skip_stage2_below=0.0
    )
    result = p.predict(_make_pil(), [0, 0, 0, 0])
    assert result.pathology_thresholds == custom


def test_stage2_thresholds_missing_keys_raises(
    stage1_checkpoint: Path, stage2_checkpoint: Path
):
    s1 = SingleStagePredictor(stage1_checkpoint, device="cpu")
    s2 = SingleStagePredictor(stage2_checkpoint, device="cpu")
    with pytest.raises(ValueError, match="missing keys"):
        HierarchicalPredictor(
            stage1=s1, stage2=s2, stage2_thresholds={"Atelectasis": 0.5}
        )
