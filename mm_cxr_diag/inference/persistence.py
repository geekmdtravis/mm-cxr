"""Save/load ``CXRModel`` checkpoints as ``(state_dict, config_dict)`` pairs."""

from __future__ import annotations

from pathlib import Path

import torch

from mm_cxr_diag.models import CXRModel, CXRModelConfig


def save_model(
    model: CXRModel,
    config: CXRModelConfig,
    file_path: str | Path,
) -> Path:
    """Serialize ``model.state_dict()`` + ``config.as_dict()`` to disk."""
    path = Path(file_path)
    if path.suffix != ".pth":
        path = path.with_suffix(".pth")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config.as_dict()}, path)
    return path


def load_model(file_path: str | Path) -> CXRModel:
    """Load a ``CXRModel`` checkpoint saved by :func:`save_model`."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist.")
    if not path.is_file():
        raise FileNotFoundError(f"Path {path} is not a file.")

    save_info = torch.load(path, map_location="cpu", weights_only=False)
    model = CXRModel(**save_info["config"])
    model.load_state_dict(save_info["model"])
    return model
