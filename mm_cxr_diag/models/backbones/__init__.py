"""Backbone implementations. Importing this package triggers registration
of every backbone into ``mm_cxr_diag.models.registry.MODELS``."""

from mm_cxr_diag.models.backbones import densenet as _densenet  # noqa: F401
from mm_cxr_diag.models.backbones import vit as _vit  # noqa: F401
