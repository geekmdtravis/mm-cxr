"""Two-stage hierarchical predictor.

Stage 1 outputs ``P(abnormal | image, tabular)``.
Stage 2 outputs ``P(pathology_i | image, tabular)`` on studies Stage 2 was
trained on (ground-truth abnormal + a small slice of Stage-1 false positives).

At inference we combine multiplicatively per the law of total probability:

    P(pathology_i) = sigmoid(s1) * sigmoid(s2_i)
    P(No Finding)  = 1 - sigmoid(s1)

The design intentionally avoids hard gating at a high threshold — that
discards rare-pathology recall on borderline studies, which is the exact
failure mode this hierarchy was designed to fix.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image

from mm_cxr_diag.data.labels import PATHOLOGY_LABELS
from mm_cxr_diag.inference.predictor import SingleStagePredictor
from mm_cxr_diag.inference.transforms import prepare_image, tabular_to_tensor

Device = torch.device | Literal["cuda", "cpu", "auto"]


@dataclass
class HierarchicalPrediction:
    """Result of a two-stage prediction.

    Attributes:
        abnormal_prob: Stage 1 sigmoid probability of abnormal.
        abnormal: Whether ``abnormal_prob >= stage1_threshold``.
        stage1_threshold: Threshold used to binarize the Stage 1 output.
        pathologies: Combined per-pathology probabilities
            (``sigmoid(s1) * sigmoid(s2_i)``). ``None`` when Stage 2 was
            short-circuited because Stage 1 was extremely confident normal.
        pathology_thresholds: Per-pathology operating thresholds applied to
            the combined probabilities. ``None`` when ``pathologies`` is.
        pathology_labels: Names in the same order as ``pathologies`` keys
            (14 pathologies, no ``No Finding``).
        no_finding_prob: ``1 - abnormal_prob`` — reported separately to
            preserve a coherent calibrated abnormal/normal axis.
        skipped_stage2: True when Stage 2 was short-circuited.
        stage1_model: Backbone name of the loaded Stage 1 checkpoint.
        stage2_model: Backbone name of the loaded Stage 2 checkpoint.
        latency_ms: Per-stage wall-clock timing.
    """

    abnormal_prob: float
    abnormal: bool
    stage1_threshold: float
    no_finding_prob: float
    pathologies: dict[str, float] | None
    pathology_thresholds: dict[str, float] | None
    pathology_labels: tuple[str, ...]
    skipped_stage2: bool
    stage1_model: str
    stage2_model: str
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serializable view — used by the CLI ``predict`` and the API."""
        return {
            "abnormal_prob": self.abnormal_prob,
            "abnormal": self.abnormal,
            "stage1_threshold": self.stage1_threshold,
            "no_finding_prob": self.no_finding_prob,
            "pathologies": self.pathologies,
            "pathology_thresholds": self.pathology_thresholds,
            "pathology_labels": list(self.pathology_labels),
            "skipped_stage2": self.skipped_stage2,
            "model_versions": {
                "stage1": self.stage1_model,
                "stage2": self.stage2_model,
            },
            "latency_ms": self.latency_ms,
        }


class HierarchicalPredictor:
    """Compose two ``SingleStagePredictor``s into a hierarchical pipeline.

    Args:
        stage1: Predictor for the abnormality gate (``num_classes=1``).
        stage2: Predictor for the 14-way pathology head.
        stage1_threshold: Threshold for reporting ``abnormal`` (does NOT
            gate Stage 2 execution — it's reported for context only).
        stage2_thresholds: Operating thresholds per pathology applied to
            the combined probability. When ``None``, all thresholds default
            to 0.5.
        skip_stage2_below: Short-circuit Stage 2 when ``P(abnormal)`` is
            below this probability. Defaults to 0.02 — well below any
            reasonable operating point, used only to save compute on
            unambiguously normal studies. Set to 0 to always run Stage 2.
        pathology_labels: Names of the 14 pathologies, in the order the
            Stage 2 model outputs them. Defaults to :data:`PATHOLOGY_LABELS`.

    Raises:
        ValueError: if ``stage1.num_classes != 1`` or if ``stage2.num_classes
            != len(pathology_labels)``.
    """

    def __init__(
        self,
        stage1: SingleStagePredictor,
        stage2: SingleStagePredictor,
        stage1_threshold: float = 0.5,
        stage2_thresholds: Mapping[str, float] | None = None,
        skip_stage2_below: float = 0.02,
        pathology_labels: Sequence[str] = PATHOLOGY_LABELS,
    ):
        if stage1.num_classes != 1:
            raise ValueError(
                f"Stage 1 predictor must have num_classes=1 "
                f"(binary abnormality gate); got {stage1.num_classes}"
            )
        if stage2.num_classes != len(pathology_labels):
            raise ValueError(
                f"Stage 2 predictor num_classes={stage2.num_classes} does not "
                f"match len(pathology_labels)={len(pathology_labels)}"
            )
        if not 0.0 <= skip_stage2_below < 1.0:
            raise ValueError("skip_stage2_below must be in [0, 1)")

        self._stage1 = stage1
        self._stage2 = stage2
        self._stage1_threshold = float(stage1_threshold)
        self._skip_stage2_below = float(skip_stage2_below)
        self._pathology_labels = tuple(pathology_labels)

        if stage2_thresholds is None:
            self._stage2_thresholds = {lbl: 0.5 for lbl in self._pathology_labels}
        else:
            missing = set(self._pathology_labels) - set(stage2_thresholds)
            if missing:
                raise ValueError(f"stage2_thresholds missing keys: {sorted(missing)}")
            self._stage2_thresholds = {
                lbl: float(stage2_thresholds[lbl]) for lbl in self._pathology_labels
            }

    @classmethod
    def from_checkpoints(
        cls,
        stage1_ckpt: str | Path,
        stage2_ckpt: str | Path,
        *,
        device: Device = "auto",
        stage1_threshold: float = 0.5,
        stage2_thresholds: Mapping[str, float] | None = None,
        skip_stage2_below: float = 0.02,
        pathology_labels: Sequence[str] = PATHOLOGY_LABELS,
    ) -> HierarchicalPredictor:
        """Convenience factory: load both stages from ``.pth`` files."""
        return cls(
            stage1=SingleStagePredictor(stage1_ckpt, device=device),
            stage2=SingleStagePredictor(stage2_ckpt, device=device),
            stage1_threshold=stage1_threshold,
            stage2_thresholds=stage2_thresholds,
            skip_stage2_below=skip_stage2_below,
            pathology_labels=pathology_labels,
        )

    @property
    def stage1(self) -> SingleStagePredictor:
        return self._stage1

    @property
    def stage2(self) -> SingleStagePredictor:
        return self._stage2

    @property
    def pathology_labels(self) -> tuple[str, ...]:
        return self._pathology_labels

    def predict(
        self,
        image: Image.Image | torch.Tensor,
        tabular: Mapping[str, float] | Sequence[float] | torch.Tensor,
    ) -> HierarchicalPrediction:
        """Run the full hierarchy on a single ``(image, tabular)`` input."""
        # Preprocess once; both stages receive identical inputs.
        img_tensor = prepare_image(image, transform=self._stage1._transform)
        tab_tensor = tabular_to_tensor(tabular)

        t0 = time.perf_counter()
        stage1_probs = self._stage1.predict_batch(img_tensor, tab_tensor)
        t1 = time.perf_counter()

        abnormal_prob = float(stage1_probs[0, 0])
        abnormal = abnormal_prob >= self._stage1_threshold

        skipped = abnormal_prob < self._skip_stage2_below
        pathologies: dict[str, float] | None = None
        pathology_thresholds: dict[str, float] | None = None
        t2 = t1

        if not skipped:
            stage2_probs = self._stage2.predict_batch(img_tensor, tab_tensor)
            t2 = time.perf_counter()
            combined = abnormal_prob * stage2_probs[0]
            pathologies = {
                lbl: float(p)
                for lbl, p in zip(self._pathology_labels, combined, strict=True)
            }
            pathology_thresholds = dict(self._stage2_thresholds)

        return HierarchicalPrediction(
            abnormal_prob=abnormal_prob,
            abnormal=abnormal,
            stage1_threshold=self._stage1_threshold,
            no_finding_prob=1.0 - abnormal_prob,
            pathologies=pathologies,
            pathology_thresholds=pathology_thresholds,
            pathology_labels=self._pathology_labels,
            skipped_stage2=skipped,
            stage1_model=self._stage1.model_name,
            stage2_model=self._stage2.model_name,
            latency_ms={
                "stage1": round((t1 - t0) * 1000.0, 3),
                "stage2": round((t2 - t1) * 1000.0, 3) if not skipped else 0.0,
                "total": round((t2 - t0) * 1000.0, 3),
            },
        )

    def predict_batch(
        self,
        images: torch.Tensor,
        tabular: torch.Tensor,
    ) -> np.ndarray:
        """Batch variant for evaluation loops.

        Returns a ``(B, num_pathologies)`` array of combined probabilities.
        Does NOT short-circuit (always runs Stage 2 for every row, since
        short-circuiting per-item in a batched forward makes little sense).
        """
        s1_probs = self._stage1.predict_batch(images, tabular)
        s2_probs = self._stage2.predict_batch(images, tabular)
        return s1_probs * s2_probs
