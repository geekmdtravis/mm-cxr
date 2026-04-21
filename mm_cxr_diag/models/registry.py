"""Model registry: a dict keyed by backbone name → nn.Module class.

Replaces the 10-way if-elif factory that previously lived in
``src/models/cxr_model.py``. Backbones self-register with ``@register(name)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

MODELS: dict[str, type] = {}


def register(name: str):
    """Decorator: register an nn.Module class under ``name`` in ``MODELS``."""

    def deco(cls: type) -> type:
        if name in MODELS:
            raise ValueError(f"Backbone '{name}' is already registered.")
        MODELS[name] = cls
        return cls

    return deco


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered backbone by name.

    Raises:
        ValueError: if ``name`` is not registered. Available names are
            listed in the error message.
    """
    if name not in MODELS:
        known = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown backbone '{name}'. Registered: {known}")
    return MODELS[name](**kwargs)
