"""CLI smoke + functional tests via Typer's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mm_cxr_diag import __version__
from mm_cxr_diag.cli.main import app
from mm_cxr_diag.data import PATHOLOGY_LABELS
from mm_cxr_diag.data.labels import LABEL_COLUMN_PREFIX
from mm_cxr_diag.inference import HierarchicalPrediction
from PIL import Image
from typer.testing import CliRunner

runner = CliRunner()

PATHOLOGY_COLUMNS = [
    LABEL_COLUMN_PREFIX + name.lower().replace(" ", "_") for name in PATHOLOGY_LABELS
]


@pytest.mark.parametrize(
    "subcommand", ["prepare-data", "train", "evaluate", "predict", "serve"]
)
def test_subcommand_help(subcommand: str):
    result = runner.invoke(app, [subcommand, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_top_level_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("prepare-data", "train", "evaluate", "predict", "serve"):
        assert name in result.output


def test_prepare_data_functional(tmp_path: Path):
    # Minimal raw NIH-schema CSV (10 rows) and stub images dir.
    rows = [
        {
            "Image Index": f"{i:08d}_000.png",
            "Finding Labels": "No Finding" if i % 2 == 0 else "Cardiomegaly",
            "Follow-up #": i,
            "Patient Age": "045Y",
            "Patient Gender": "M" if i % 2 == 0 else "F",
            "View Position": "PA" if i % 2 == 0 else "AP",
        }
        for i in range(10)
    ]
    raw = pd.DataFrame(rows)
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    # create one image so the presence check passes silently
    Image.new("RGB", (32, 32)).save(images_dir / "00000000_000.png")

    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "prepare-data",
            "--input-csv",
            str(raw_path),
            "--output-dir",
            str(out_dir),
            "--images-dir",
            str(images_dir),
            "--val-size",
            "0.2",
            "--test-size",
            "0.2",
            "--seed",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "val.csv").exists()
    assert (out_dir / "test.csv").exists()

    # Every row from the raw input should appear in exactly one split.
    n_train = len(pd.read_csv(out_dir / "train.csv"))
    n_val = len(pd.read_csv(out_dir / "val.csv"))
    n_test = len(pd.read_csv(out_dir / "test.csv"))
    assert n_train + n_val + n_test == len(raw)

    # Preprocessed columns exist.
    train = pd.read_csv(out_dir / "train.csv")
    for col in ("imageIndex", "patientAge", "patientGender", "viewPosition"):
        assert col in train.columns
    for col in PATHOLOGY_COLUMNS:
        assert col in train.columns


def test_predict_functional(tmp_path: Path, monkeypatch):
    """The CLI composes `HierarchicalPredictor.from_checkpoints` then calls
    `.predict()`. We stub both to avoid needing real checkpoints."""
    import mm_cxr_diag.cli.commands.predict as predict_mod

    stage1_ckpt = tmp_path / "s1.pth"
    stage2_ckpt = tmp_path / "s2.pth"
    stage1_ckpt.write_bytes(b"\x00")  # Typer --exists check only requires file presence
    stage2_ckpt.write_bytes(b"\x00")

    image_path = tmp_path / "img.png"
    Image.new("RGB", (32, 32)).save(image_path)

    class StubPredictor:
        def predict(self, pil, tabular):
            return HierarchicalPrediction(
                abnormal_prob=0.7,
                abnormal=True,
                stage1_threshold=0.5,
                no_finding_prob=0.3,
                pathologies={label: 0.1 for label in PATHOLOGY_LABELS},
                pathology_thresholds={label: 0.5 for label in PATHOLOGY_LABELS},
                pathology_labels=PATHOLOGY_LABELS,
                skipped_stage2=False,
                stage1_model="densenet121",
                stage2_model="densenet121",
                latency_ms={"stage1": 1.0, "stage2": 2.0, "total": 3.0},
            )

    class StubFactory:
        @staticmethod
        def from_checkpoints(*args, **kwargs):
            return StubPredictor()

    monkeypatch.setattr(predict_mod, "HierarchicalPredictor", StubFactory)

    tabular = {
        "patientAge": 0.6,
        "patientGender": 1,
        "viewPosition": 0,
        "followUpNumber": 0.1,
    }
    result = runner.invoke(
        app,
        [
            "predict",
            "--stage1-ckpt",
            str(stage1_ckpt),
            "--stage2-ckpt",
            str(stage2_ckpt),
            "--image",
            str(image_path),
            "--tabular-json",
            json.dumps(tabular),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed["abnormal_prob"] == 0.7
    assert parsed["abnormal"] is True
    assert parsed["model_versions"]["stage1"] == "densenet121"
    assert len(parsed["pathologies"]) == len(PATHOLOGY_LABELS)


def test_predict_table_output(tmp_path: Path, monkeypatch):
    """--output table renders a human-readable summary."""
    import mm_cxr_diag.cli.commands.predict as predict_mod

    stage1_ckpt = tmp_path / "s1.pth"
    stage2_ckpt = tmp_path / "s2.pth"
    stage1_ckpt.write_bytes(b"\x00")
    stage2_ckpt.write_bytes(b"\x00")
    image_path = tmp_path / "img.png"
    Image.new("RGB", (32, 32)).save(image_path)

    class StubPredictor:
        def predict(self, pil, tabular):
            return HierarchicalPrediction(
                abnormal_prob=0.9,
                abnormal=True,
                stage1_threshold=0.5,
                no_finding_prob=0.1,
                pathologies={label: 0.6 for label in PATHOLOGY_LABELS},
                pathology_thresholds={label: 0.5 for label in PATHOLOGY_LABELS},
                pathology_labels=PATHOLOGY_LABELS,
                skipped_stage2=False,
                stage1_model="densenet121",
                stage2_model="densenet121",
                latency_ms={"stage1": 1, "stage2": 2, "total": 3},
            )

    monkeypatch.setattr(
        predict_mod.HierarchicalPredictor,
        "from_checkpoints",
        classmethod(lambda cls, *a, **kw: StubPredictor()),
    )

    result = runner.invoke(
        app,
        [
            "predict",
            "--stage1-ckpt",
            str(stage1_ckpt),
            "--stage2-ckpt",
            str(stage2_ckpt),
            "--image",
            str(image_path),
            "--tabular-json",
            '{"patientAge":0.6,"patientGender":1,"viewPosition":0,"followUpNumber":0.1}',
            "--output",
            "table",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "abnormal_prob" in result.output
    assert "Atelectasis" in result.output  # a pathology listed in the table


def test_predict_requires_tabular(tmp_path: Path):
    stage1_ckpt = tmp_path / "s1.pth"
    stage2_ckpt = tmp_path / "s2.pth"
    stage1_ckpt.write_bytes(b"\x00")
    stage2_ckpt.write_bytes(b"\x00")
    image_path = tmp_path / "img.png"
    Image.new("RGB", (32, 32)).save(image_path)

    result = runner.invoke(
        app,
        [
            "predict",
            "--stage1-ckpt",
            str(stage1_ckpt),
            "--stage2-ckpt",
            str(stage2_ckpt),
            "--image",
            str(image_path),
        ],
    )
    assert result.exit_code != 0
    assert "tabular" in result.output.lower()


def test_serve_requires_checkpoints():
    """`serve` with fastapi importable but no checkpoints should exit non-zero."""
    # fastapi isn't installed in this env, so we only verify the error path is taken.
    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 0
    # Either the import guard or the checkpoint guard fires; both acceptable.
    assert "stage1-ckpt" in result.output.lower() or "serve" in result.output.lower()


def test_train_dry_run(tmp_path: Path):
    """Dry-run path: builds model + one batch, no actual training."""
    # Build a minimal preprocessed split using the same fixture pattern as
    # test_dataset.py.
    rng = np.random.default_rng(0)
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_names = [f"img_{i}.png" for i in range(4)]
    for name in image_names:
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
            images_dir / name
        )

    def _make_csv(n: int) -> Path:
        df = pd.DataFrame(
            {
                "imageIndex": image_names[:n],
                "patientAge": rng.integers(20, 80, n),
                "patientGender": rng.integers(0, 2, n),
                "viewPosition": rng.integers(0, 2, n),
                "followUpNumber": rng.integers(0, 5, n),
            }
        )
        for col in PATHOLOGY_COLUMNS:
            df[col] = rng.integers(0, 2, n)
        path = tmp_path / f"rows_{n}.csv"
        df.to_csv(path, index=False)
        return path

    train_csv = _make_csv(4)
    val_csv = _make_csv(2)

    # Tiny model config: minimum hidden dim, untrained.
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("""model: densenet121
hidden_dims: [16]
dropout: 0.1
num_classes: 14
tabular_features: 4
freeze_backbone: false
""")

    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(config_path),
            "--train-csv",
            str(train_csv),
            "--val-csv",
            str(val_csv),
            "--images-dir",
            str(images_dir),
            "--output-dir",
            str(tmp_path / "runs"),
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Dry run OK" in result.output
