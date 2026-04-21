"""Unit tests for ChestXrayDataset."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mm_cxr_diag.data import PATHOLOGY_LABELS, ChestXrayDataset
from mm_cxr_diag.data.labels import LABEL_COLUMN_PREFIX
from PIL import Image
from torchvision import transforms

PATHOLOGY_COLUMNS = [
    LABEL_COLUMN_PREFIX + name.lower().replace(" ", "_") for name in PATHOLOGY_LABELS
]


def _make_fixture(num_samples: int = 3, image_size=(64, 64)):
    """Create a tmpdir with ``num_samples`` synthetic images and a clinical CSV."""
    test_dir = Path(tempfile.mkdtemp())
    images_dir = test_dir / "images"
    images_dir.mkdir()

    image_names = []
    for i in range(num_samples):
        name = f"image_{i}.png"
        image_names.append(name)
        Image.fromarray(
            np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        ).save(images_dir / name)

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "imageIndex": image_names,
            "patientAge": rng.integers(20, 80, num_samples),
            "patientGender": rng.integers(0, 2, num_samples),
            "viewPosition": rng.integers(0, 2, num_samples),
            "followUpNumber": rng.integers(0, 5, num_samples),
        }
    )
    for col in PATHOLOGY_COLUMNS:
        df[col] = rng.integers(0, 2, num_samples)
    df["label_no_finding"] = (df[PATHOLOGY_COLUMNS].sum(axis=1) == 0).astype(int)

    csv_path = test_dir / "clinical_data.csv"
    df.to_csv(csv_path, index=False)
    return test_dir, images_dir, csv_path, image_size


class TestChestXrayDataset(unittest.TestCase):
    def setUp(self):
        self.num_samples = 3
        (
            self.test_dir,
            self.images_dir,
            self.clinical_data_path,
            self.image_size,
        ) = _make_fixture(num_samples=self.num_samples)
        self.dataset = ChestXrayDataset(
            clinical_data=self.clinical_data_path, cxr_images_dir=self.images_dir
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_init(self):
        self.assertIsInstance(self.dataset, ChestXrayDataset)
        self.assertEqual(len(self.dataset.tabular_df), self.num_samples)

    def test_len(self):
        self.assertEqual(len(self.dataset), self.num_samples)

    def test_getitem_multilabel(self):
        image, tabular_features, labels = self.dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(tabular_features, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(image.shape, (3, *self.image_size))
        self.assertEqual(tabular_features.shape, (4,))
        self.assertEqual(labels.shape, (14,))

    def test_getitem_legacy15(self):
        ds = ChestXrayDataset(
            clinical_data=self.clinical_data_path,
            cxr_images_dir=self.images_dir,
            label_mode="multilabel_legacy15",
        )
        _, _, labels = ds[0]
        self.assertEqual(labels.shape, (15,))

    def test_getitem_binary(self):
        ds = ChestXrayDataset(
            clinical_data=self.clinical_data_path,
            cxr_images_dir=self.images_dir,
            label_mode="binary",
        )
        _, _, labels = ds[0]
        self.assertEqual(labels.shape, (1,))
        self.assertIn(labels.item(), (0.0, 1.0))

    def test_custom_transform(self):
        custom = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((32, 32))]
        )
        ds = ChestXrayDataset(
            clinical_data=self.clinical_data_path,
            cxr_images_dir=self.images_dir,
            transform=custom,
        )
        image, _, _ = ds[0]
        self.assertEqual(image.shape, (3, 32, 32))

    def test_invalid_index(self):
        with self.assertRaises(IndexError):
            _ = self.dataset[len(self.dataset)]

    def test_invalid_label_mode(self):
        ds = ChestXrayDataset(
            clinical_data=self.clinical_data_path,
            cxr_images_dir=self.images_dir,
            label_mode="bogus",  # type: ignore[arg-type]
        )
        with self.assertRaises(ValueError):
            _ = ds[0]


if __name__ == "__main__":
    unittest.main()
