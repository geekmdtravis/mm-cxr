"""NIH ChestX-ray14 dataset.

Returns ``(image, tabular_features, labels)`` tuples. Labels shape depends
on the Stage: 14-dim multi-label (Stage 2) or 1-dim binary (Stage 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from mm_cxr_diag.data.labels import LABEL_COLUMN_PREFIX, PATHOLOGY_LABELS

LabelMode = Literal["multilabel", "multilabel_legacy15", "binary"]


class ChestXrayDataset(Dataset):
    """Custom dataset for NIH ChestX-ray14 images + clinical tabular features.

    Args:
        clinical_data: Path to CSV with columns ``imageIndex``, ``patientAge``,
            ``patientGender``, ``viewPosition``, ``followUpNumber``, and
            ``label_*`` columns for each pathology.
        cxr_images_dir: Directory containing image files named by
            ``imageIndex``.
        transform: Optional torchvision transform. Defaults to ``ToTensor``.
        label_mode: Which label vector the dataset should emit.
            - ``"multilabel"``: 14-dim vector in the order of
              :data:`PATHOLOGY_LABELS`. Default.
            - ``"multilabel_legacy15"``: the original 15-dim vector
              (14 pathologies + ``No Finding``). Used to reproduce the
              single-stage baseline.
            - ``"binary"``: 1-dim vector, ``1`` if any pathology flag is set
              (see :func:`derive_binary_label`). Used for Stage 1.
    """

    def __init__(
        self,
        clinical_data: Path,
        cxr_images_dir: Path,
        transform: transforms.Compose | None = None,
        label_mode: LabelMode = "multilabel",
    ):
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.images_dir = Path(cxr_images_dir)
        self.tabular_df = pd.read_csv(clinical_data)
        self.label_mode = label_mode
        self._pathology_cols = [
            LABEL_COLUMN_PREFIX + name.lower().replace(" ", "_")
            for name in PATHOLOGY_LABELS
        ]

    def __len__(self) -> int:
        return len(self.tabular_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.tabular_df.iloc[idx]

        img_path = self.images_dir / row["imageIndex"]
        _image = Image.open(img_path).convert("RGB")
        image = self.transform(_image)

        if self.label_mode == "multilabel":
            labels = torch.tensor(
                [float(row[c]) for c in self._pathology_cols], dtype=torch.float32
            )
        elif self.label_mode == "multilabel_legacy15":
            labels = torch.tensor(
                row.filter(like=LABEL_COLUMN_PREFIX).to_numpy(dtype=float),
                dtype=torch.float32,
            )
        elif self.label_mode == "binary":
            abnormal = any(float(row[c]) > 0.5 for c in self._pathology_cols)
            labels = torch.tensor([1.0 if abnormal else 0.0], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

        tabular_features = torch.tensor(
            [
                float(row["patientAge"]),
                float(row["patientGender"]),
                float(row["viewPosition"]),
                float(row["followUpNumber"]),
            ],
            dtype=torch.float32,
        )

        return image, tabular_features, labels
