"""Single-stage predictor: one checkpoint → one image → sigmoid probs.

Used standalone for either-stage-alone inference and as a building block
for :class:`mm_cxr_diag.inference.hierarchical.HierarchicalPredictor`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mm_cxr_diag.inference.persistence import load_model
from mm_cxr_diag.inference.transforms import (
    default_preprocess,
    prepare_image,
    tabular_to_tensor,
)
from mm_cxr_diag.models import CXRModel, CXRModelConfig

Device = torch.device | Literal["cuda", "cpu", "auto"]


def _resolve_device(device: Device) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


class SingleStagePredictor:
    """Load one ``CXRModel`` checkpoint and run inference on single items.

    Args:
        checkpoint_path: ``.pth`` file saved by
            :func:`mm_cxr_diag.inference.persistence.save_model`.
        device: ``"cuda"``, ``"cpu"``, ``"auto"``, or a ``torch.device``.
            ``"auto"`` picks CUDA when available.
        transform: Image transform applied to PIL inputs. Defaults to
            ``default_preprocess()`` (Resize 224 + ToTensor + ImageNet norm).

    Typical usage::

        p = SingleStagePredictor("stage2.pth")
        probs = p.predict(pil_image, {"patientAge": 0.6, ...})
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: Device = "auto",
        transform: transforms.Compose | None = None,
    ):
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self._checkpoint_path = path
        self._device = _resolve_device(device)
        self._transform = transform if transform is not None else default_preprocess()

        self._model: CXRModel = load_model(path).to(self._device)
        self._model.eval()

        # Re-derive the config from the model (load_model already used it
        # to construct; we keep a copy for metadata access).
        raw = torch.load(path, map_location="cpu", weights_only=False)
        self._config = CXRModelConfig(**raw["config"])

    @property
    def config(self) -> CXRModelConfig:
        return self._config

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def num_classes(self) -> int:
        return int(self._config.num_classes)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    def predict(
        self,
        image: Image.Image | torch.Tensor,
        tabular: Mapping[str, float] | Sequence[float] | torch.Tensor,
    ) -> np.ndarray:
        """Return sigmoid probabilities for a single item, shape ``(num_classes,)``."""
        probs = self.predict_batch(
            prepare_image(image, transform=self._transform),
            tabular_to_tensor(tabular),
        )
        return probs[0]

    def predict_batch(
        self,
        images: torch.Tensor,
        tabular: torch.Tensor,
    ) -> np.ndarray:
        """Return sigmoid probabilities for a batch, shape ``(B, num_classes)``.

        ``images`` must be a ``(B, 3, H, W)`` tensor already preprocessed.
        ``tabular`` must be ``(B, 4)``.
        """
        if images.dim() != 4:
            raise ValueError(f"images must be 4-D (B,C,H,W), got {images.dim()}-D")
        if tabular.dim() != 2:
            raise ValueError(f"tabular must be 2-D (B,F), got {tabular.dim()}-D")
        if images.shape[0] != tabular.shape[0]:
            raise ValueError(
                f"Batch size mismatch: images={images.shape[0]} "
                f"vs tabular={tabular.shape[0]}"
            )

        images = images.to(self._device)
        tabular = tabular.to(self._device)
        with torch.no_grad():
            logits = self._model(images, tabular)
            probs = torch.sigmoid(logits)
        return probs.detach().cpu().numpy()
