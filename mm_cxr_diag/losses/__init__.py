"""Loss functions used by the trainer."""

from mm_cxr_diag.losses.focal import FocalLoss, reweight

__all__ = ["FocalLoss", "reweight"]
