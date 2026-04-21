"""Model architectures and the registry-backed factory."""

# Import backbones so that their @register decorators fire on package import.
from mm_cxr_diag.models import backbones as _backbones  # noqa: F401
from mm_cxr_diag.models.cxr_model import CXRModel, CXRModelConfig, SupportedModels
from mm_cxr_diag.models.fusion import build_mm_classifier
from mm_cxr_diag.models.registry import MODELS, build_model, register

__all__ = [
    "MODELS",
    "CXRModel",
    "CXRModelConfig",
    "SupportedModels",
    "build_mm_classifier",
    "build_model",
    "register",
]
