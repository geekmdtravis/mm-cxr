"""Shared pytest fixtures for the mm-cxr-diag test suite.

The ``synthetic_dataset`` fixture replaces copy-pasted setUp boilerplate in
several tests. It produces:

- A tmpdir with ``n`` synthetic PNGs in ``<tmp>/images/``.
- A preprocessed clinical CSV at ``<tmp>/<name>.csv`` with the exact 14
  pathology ``label_*`` columns the production dataset expects, plus the
  four tabular features.

Per-backbone smoke tests also build tiny ``CXRModelConfig``s for each
registered backbone via ``tiny_configs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mm_cxr_diag.data.labels import LABEL_COLUMN_PREFIX, PATHOLOGY_LABELS
from mm_cxr_diag.models import MODELS, CXRModelConfig
from PIL import Image

_PATHOLOGY_COLUMNS: list[str] = [
    LABEL_COLUMN_PREFIX + name.lower().replace(" ", "_") for name in PATHOLOGY_LABELS
]


@dataclass
class SyntheticSplit:
    """Handle for a synthetic train/val/test split produced by the fixture."""

    root: Path
    images_dir: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path


def _make_images(
    images_dir: Path,
    count: int,
    rng: np.random.Generator,
    size: int = 224,
) -> list[str]:
    """Synthetic PNGs sized to match production expectations.

    Default is 224 because the ViT backbones have a fixed patch grid and
    reject smaller inputs; DenseNets handle anything via adaptive pooling.
    Keeping the fixture at 224 lets one fixture serve every backbone.
    """
    names: list[str] = []
    for i in range(count):
        name = f"img_{i:04d}.png"
        names.append(name)
        arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        Image.fromarray(arr).save(images_dir / name)
    return names


def _write_split_csv(
    path: Path, image_names: list[str], rng: np.random.Generator
) -> None:
    n = len(image_names)
    df = pd.DataFrame(
        {
            "imageIndex": image_names,
            "patientAge": rng.uniform(0.1, 0.9, n),
            "patientGender": rng.integers(0, 2, n),
            "viewPosition": rng.integers(0, 2, n),
            "followUpNumber": rng.uniform(0.0, 1.0, n),
        }
    )
    for col in _PATHOLOGY_COLUMNS:
        df[col] = rng.integers(0, 2, n)
    # Synthetic "No Finding" — consistent with derive_binary_label invariant.
    df["label_no_finding"] = (df[_PATHOLOGY_COLUMNS].sum(axis=1) == 0).astype(int)
    df.to_csv(path, index=False)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> SyntheticSplit:
    """Create a tiny train/val/test dataset on disk for integration tests.

    Sizes are kept intentionally tiny (8 / 4 / 4) — these fixtures are for
    wiring verification, not for teaching the model anything.
    """
    rng = np.random.default_rng(0)
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    train_names = _make_images(images_dir, 8, rng)
    val_names = _make_images(images_dir, 4, rng)
    test_names = _make_images(images_dir, 4, rng)

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    _write_split_csv(train_csv, train_names, rng)
    _write_split_csv(val_csv, val_names, rng)
    _write_split_csv(test_csv, test_names, rng)

    return SyntheticSplit(
        root=tmp_path,
        images_dir=images_dir,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
    )


@pytest.fixture(params=sorted(MODELS))
def backbone(request: pytest.FixtureRequest) -> str:
    """Parametrizes across every registered backbone.

    Used with ``@pytest.mark.slow`` tests that compare behavior across
    architectures. On the default fast path, those tests are skipped.
    """
    return request.param


def tiny_config(backbone: str, *, num_classes: int = 14) -> CXRModelConfig:
    """Build a small-head config for integration tests (shared helper)."""
    return CXRModelConfig(
        model=backbone,
        hidden_dims=(16,),
        dropout=0.1,
        num_classes=num_classes,
        tabular_features=4,
        freeze_backbone=False,
    )
