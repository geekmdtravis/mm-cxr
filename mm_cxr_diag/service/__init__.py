"""FastAPI service layer for hierarchical inference.

All imports here are guarded by the ``serve`` extra: importing
``mm_cxr_diag.service`` at all requires ``fastapi``, ``uvicorn[standard]``,
and ``python-multipart``. The top-level package stays CUDA/HTTP-agnostic.
"""

from mm_cxr_diag.service.app import create_app

__all__ = ["create_app"]
