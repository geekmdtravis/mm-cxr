"""Per-backbone trainer smoke tests.

Each backbone is trained for 2 epochs on a tiny synthetic fixture to
verify that the full (dataloader → model → loss → optimizer → checkpoint
→ reload) pipeline is wired correctly for every registered architecture.

Marked ``slow`` because each backbone materializes ImageNet-pretrained
weights (even though we don't actually train meaningfully). The default
``ci.yml`` run skips these; they execute on the nightly ``ci-slow``
schedule and on demand via ``pytest -m slow``.
"""

from __future__ import annotations

import pytest
from mm_cxr_diag.data import create_dataloader
from mm_cxr_diag.inference.persistence import load_model
from mm_cxr_diag.training import train_model
from tests.conftest import SyntheticSplit, tiny_config


@pytest.mark.slow
@pytest.mark.parametrize("num_classes", [1, 14], ids=["stage1", "stage2"])
def test_trainer_smoke(
    backbone: str,
    num_classes: int,
    synthetic_dataset: SyntheticSplit,
    tmp_path,
):
    """Full pipeline: build loaders → train → save → reload → run on val."""
    label_mode = "binary" if num_classes == 1 else "multilabel"

    train_loader = create_dataloader(
        clinical_data=synthetic_dataset.train_csv,
        cxr_images_dir=synthetic_dataset.images_dir,
        batch_size=2,
        num_workers=0,
        label_mode=label_mode,
    )
    val_loader = create_dataloader(
        clinical_data=synthetic_dataset.val_csv,
        cxr_images_dir=synthetic_dataset.images_dir,
        batch_size=2,
        num_workers=0,
        label_mode=label_mode,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    config = tiny_config(backbone, num_classes=num_classes)
    best_val_loss, _best_auc, _mean_t, _total_t, _epochs, model = train_model(
        model_config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        lr=1e-3,
        patience=10,
        plot_path=str(run_dir / "curves.png"),
        best_model_path=str(run_dir / "best.pth"),
        last_model_path=str(run_dir / "last.pth"),
        train_val_data_path=str(run_dir / "tv.csv"),
        device="cpu",
    )

    # Numerical correctness isn't the point — we assert shape/pipeline only.
    assert model is not None
    assert (run_dir / "best.pth").exists()
    assert (run_dir / "last.pth").exists()
    assert (run_dir / "tv.csv").exists()
    assert (run_dir / "curves.png").exists()
    assert best_val_loss >= 0.0

    # Round-trip the checkpoint.
    reloaded = load_model(run_dir / "best.pth")
    assert reloaded.model_name == backbone
    assert reloaded.model.classifier[-1].out_features == num_classes
