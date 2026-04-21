"""Preprocessing utilities for the NIH ChestX-ray14 metadata CSV.

These transforms are element-wise and do not cause train/test leakage.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch

_VALID_LABELS = (
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "effusion",
    "emphysema",
    "fibrosis",
    "hernia",
    "infiltration",
    "mass",
    "no finding",
    "nodule",
    "pleural_thickening",
    "pneumonia",
    "pneumothorax",
)

_LABEL_COLUMNS = (
    "label_atelectasis",
    "label_cardiomegaly",
    "label_consolidation",
    "label_edema",
    "label_effusion",
    "label_emphysema",
    "label_fibrosis",
    "label_hernia",
    "label_infiltration",
    "label_mass",
    "label_no_finding",
    "label_nodule",
    "label_pleural_thickening",
    "label_pneumonia",
    "label_pneumothorax",
)


def generate_image_labels(finding_labels: str) -> torch.Tensor:
    """One-hot encode a pipe-delimited NIH 'Finding Labels' string."""
    fl = finding_labels.lower()
    if fl.strip() == "":
        raise ValueError("Finding labels cannot be an empty string.")

    for label in fl.split("|"):
        if label not in _VALID_LABELS:
            raise ValueError(f"Invalid finding label: {label}")

    out = torch.zeros(15, dtype=torch.float32)
    out[0] = 1 if "atelectasis" in fl else 0
    out[1] = 1 if "cardiomegaly" in fl else 0
    out[2] = 1 if "consolidation" in fl else 0
    out[3] = 1 if "edema" in fl else 0
    out[4] = 1 if "effusion" in fl else 0
    out[5] = 1 if "emphysema" in fl else 0
    out[6] = 1 if "fibrosis" in fl else 0
    out[7] = 1 if "hernia" in fl else 0
    out[8] = 1 if "infiltration" in fl else 0
    out[9] = 1 if "mass" in fl else 0
    out[10] = 1 if "no finding" in fl else 0
    out[11] = 1 if "nodule" in fl else 0
    out[12] = 1 if "pleural_thickening" in fl else 0
    out[13] = 1 if "pneumonia" in fl else 0
    out[14] = 1 if "pneumothorax" in fl else 0
    return out


def convert_agestr_to_years(agestr: str) -> float:
    """Convert an NIH age string like ``057Y`` or ``009M`` to a float in years."""
    s = agestr.strip().lower()
    if not s:
        raise ValueError("Age string cannot be empty.")
    if len(s) != 4:
        raise ValueError(f"Invalid age string length: {agestr}")
    if not (s[:-1].isdigit() and s[-1] in {"y", "m", "d", "w"}):
        raise ValueError(f"Invalid age string format: {agestr}")

    value = float(s[:-1])
    if s.endswith("y"):
        return value
    if s.endswith("m"):
        return value / 12
    if s.endswith("d"):
        return value / 365
    if s.endswith("w"):
        return value / 52
    raise ValueError(f"Invalid age string format: {agestr}")


def create_working_tabular_df(df: pd.DataFrame) -> pd.DataFrame:
    """Transform the raw NIH metadata CSV into the preprocessed schema.

    Returns a DataFrame with columns: ``imageIndex``, ``followUpNumber``,
    ``patientAge``, ``patientGender`` (0=M, 1=F), ``viewPosition`` (0=PA,
    1=AP), and 15 ``label_*`` one-hot columns.
    """
    working = pd.DataFrame()
    working["imageIndex"] = df["Image Index"]
    working["followUpNumber"] = df["Follow-up #"]
    working["patientAge"] = df["Patient Age"].apply(convert_agestr_to_years)
    working["patientGender"] = df["Patient Gender"].str.upper().map({"M": 0, "F": 1})
    working["viewPosition"] = df["View Position"].str.upper().map({"PA": 0, "AP": 1})

    for idx, row in df.iterrows():
        labels = generate_image_labels(row["Finding Labels"])
        if idx == 0:
            for name in _LABEL_COLUMNS:
                working[name] = 0
        for col, value in zip(_LABEL_COLUMNS, labels, strict=True):
            working.at[idx, col] = value.item()

    return working


def randomize_df(df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """Shuffle rows and reset the index."""
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random row split. ``test_size`` must be strictly between 0 and 1."""
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1 exclusive")
    train_df = df.sample(frac=1 - test_size, random_state=seed)
    test_df = df.drop(train_df.index)
    return train_df, test_df
