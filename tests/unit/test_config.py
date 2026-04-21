"""Unit tests for RuntimeSettings and CXRModelConfig YAML round-trips."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from mm_cxr_diag.config import RuntimeSettings, ensure_dirs
from mm_cxr_diag.models import CXRModelConfig


@pytest.fixture
def _clear_env(monkeypatch):
    """Clear any MM_CXR_* env vars that might pollute a test."""
    for k in list(os.environ):
        if k.startswith("MM_CXR_"):
            monkeypatch.delenv(k, raising=False)
    # Avoid accidental .env pickup from the repo root.
    monkeypatch.chdir("/tmp")


def test_runtime_settings_has_no_side_effects_on_import():
    """Instantiating RuntimeSettings() must not create filesystem artifacts."""
    before = set(Path("/tmp").iterdir())
    s = RuntimeSettings()
    after = set(Path("/tmp").iterdir())
    assert before == after
    assert s.device == "auto"
    assert s.seed == 42
    assert s.log_level == "info"


def test_resolve_paths_fills_defaults(tmp_path: Path, _clear_env):
    s = RuntimeSettings(project_root=tmp_path).resolve_paths()
    assert s.artifacts_dir == tmp_path / "artifacts"
    assert s.logs_dir == tmp_path / "logs"
    assert s.runs_dir == tmp_path / "results" / "runs"
    assert s.checkpoints_dir == tmp_path / "results" / "checkpoints"
    # Nothing should have been created yet.
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "logs").exists()


def test_resolve_paths_respects_env_overrides(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MM_CXR_ARTIFACTS_DIR", str(tmp_path / "custom_artifacts"))
    monkeypatch.setenv("MM_CXR_DEVICE", "cpu")
    monkeypatch.setenv("MM_CXR_SEED", "7")
    s = RuntimeSettings(project_root=tmp_path).resolve_paths()
    assert s.artifacts_dir == tmp_path / "custom_artifacts"
    assert s.device == "cpu"
    assert s.seed == 7


def test_ensure_dirs_creates_tree(tmp_path: Path, _clear_env):
    s = RuntimeSettings(project_root=tmp_path)
    ensure_dirs(s)
    assert s.artifacts_dir.exists()
    assert s.logs_dir.exists()
    assert s.runs_dir.exists()
    assert s.checkpoints_dir.exists()


def test_cxr_model_config_yaml_round_trip(tmp_path: Path):
    config = CXRModelConfig(
        model="densenet121",
        hidden_dims=[512, 256],
        dropout=0.3,
        num_classes=14,
        tabular_features=4,
        freeze_backbone=True,
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.as_dict()))

    loaded = CXRModelConfig.from_yaml(str(path))
    assert loaded.model == "densenet121"
    # `from_yaml` upgrades list → tuple for stable hashing of hidden_dims.
    assert loaded.hidden_dims == (512, 256)
    assert loaded.dropout == 0.3
    assert loaded.num_classes == 14
    assert loaded.tabular_features == 4
    assert loaded.freeze_backbone is True
