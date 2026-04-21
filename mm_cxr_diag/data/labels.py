"""NIH ChestX-ray14 label conventions.

Single source of truth for class label ordering and the rule that collapses
the 14-label multi-label vector into a binary abnormal/normal target used
by the Stage 1 gate.
"""

from __future__ import annotations

import pandas as pd

PATHOLOGY_LABELS: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax",
)
"""The 14 pathology classes — Stage 2 output order."""

CLASS_LABELS: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax",
)
"""Legacy 15-class label order (14 pathologies + 'No Finding').

Used by the single-stage model. Preserved for comparison against the
hierarchical system. The pathology-only order is ``PATHOLOGY_LABELS``.
"""

# Column name prefix used by the preprocessed CSVs.
LABEL_COLUMN_PREFIX = "label_"


def derive_binary_label(row: pd.Series | dict) -> int:
    """Collapse a multi-label row to binary abnormal (1) vs normal (0).

    A study is abnormal iff any of the 14 pathology flags is set.
    The ``label_no_finding`` column is ignored — NIH-14's NLP-mined labels
    occasionally set both ``No Finding`` and a pathology on the same study;
    we prefer the pathology flags.
    """
    for name in PATHOLOGY_LABELS:
        col = LABEL_COLUMN_PREFIX + name.lower().replace(" ", "_")
        val = row.get(col, 0)
        if float(val) > 0.5:
            return 1
    return 0
