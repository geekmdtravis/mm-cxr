"""Unit tests for create_dataloader."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mm_cxr_diag.data import PATHOLOGY_LABELS, create_dataloader
from mm_cxr_diag.data.labels import LABEL_COLUMN_PREFIX
from PIL import Image
from torch.utils.data import DataLoader

PATHOLOGY_COLUMNS = [
    LABEL_COLUMN_PREFIX + name.lower().replace(" ", "_") for name in PATHOLOGY_LABELS
]


class TestDataloaders(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.images_dir = self.test_dir / "images"
        self.images_dir.mkdir()

        self.num_samples = 10
        self.image_size = (64, 64)
        self.image_names = []

        for i in range(self.num_samples):
            name = f"image_{i}.png"
            self.image_names.append(name)
            Image.fromarray(
                np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
            ).save(self.images_dir / name)

        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "imageIndex": self.image_names,
                "patientAge": rng.integers(20, 80, self.num_samples),
                "patientGender": rng.integers(0, 2, self.num_samples),
                "viewPosition": rng.integers(0, 2, self.num_samples),
                "followUpNumber": rng.integers(0, 5, self.num_samples),
            }
        )
        for col in PATHOLOGY_COLUMNS:
            df[col] = rng.integers(0, 2, self.num_samples)

        self.clinical_data_path = self.test_dir / "clinical_data.csv"
        df.to_csv(self.clinical_data_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_create_dataloader(self):
        loader = create_dataloader(
            clinical_data=self.clinical_data_path,
            cxr_images_dir=self.images_dir,
            batch_size=4,
            num_workers=0,
        )
        self.assertIsInstance(loader, DataLoader)
        self.assertEqual(loader.batch_size, 4)

    def test_batch_generation(self):
        batch_size = 4
        loader = create_dataloader(
            clinical_data=self.clinical_data_path,
            cxr_images_dir=self.images_dir,
            batch_size=batch_size,
            num_workers=0,
        )
        images, tabular, labels = next(iter(loader))
        self.assertEqual(images.shape, (batch_size, 3, *self.image_size))
        self.assertEqual(tabular.shape, (batch_size, 4))
        self.assertEqual(labels.shape, (batch_size, 14))

    def test_binary_label_mode(self):
        loader = create_dataloader(
            clinical_data=self.clinical_data_path,
            cxr_images_dir=self.images_dir,
            batch_size=4,
            num_workers=0,
            label_mode="binary",
        )
        _, _, labels = next(iter(loader))
        self.assertEqual(labels.shape, (4, 1))

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            create_dataloader(
                clinical_data=self.clinical_data_path,
                cxr_images_dir=self.images_dir,
                batch_size=0,
            )
        with self.assertRaises(ValueError):
            create_dataloader(
                clinical_data=self.clinical_data_path,
                cxr_images_dir=self.images_dir,
                batch_size=-1,
            )

    def test_invalid_num_workers(self):
        with self.assertRaises(ValueError):
            create_dataloader(
                clinical_data=self.clinical_data_path,
                cxr_images_dir=self.images_dir,
                num_workers=-1,
            )

    def test_invalid_normalization_mode(self):
        with self.assertRaises(ValueError):
            create_dataloader(
                clinical_data=self.clinical_data_path,
                cxr_images_dir=self.images_dir,
                normalization_mode="bogus",  # type: ignore[arg-type]
            )

    def test_multi_worker_loading(self):
        loader = create_dataloader(
            clinical_data=self.clinical_data_path,
            cxr_images_dir=self.images_dir,
            batch_size=4,
            num_workers=2,
        )
        for images, tabular, labels in loader:
            self.assertIsInstance(images, torch.Tensor)
            self.assertIsInstance(tabular, torch.Tensor)
            self.assertIsInstance(labels, torch.Tensor)
            break


if __name__ == "__main__":
    unittest.main()
