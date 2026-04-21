"""Unit tests for CXRModel + CXRModelConfig (registry-backed)."""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from mm_cxr_diag.models import MODELS, CXRModel, CXRModelConfig, build_model

BACKBONES = ("densenet121", "densenet201", "vit_b_16", "vit_b_32", "vit_l_16")


class TestCXRModelConfig(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = Path(self.test_dir) / "test_config.yaml"
        self.config_path.write_text("""
model: vit_b_16
hidden_dims: [256, 128]
dropout: 0.5
num_classes: 10
tabular_features: 8
freeze_backbone: true
""")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_default_initialization(self):
        config = CXRModelConfig(model="densenet121")
        self.assertEqual(config.model, "densenet121")
        self.assertIsNone(config.hidden_dims)
        self.assertEqual(config.dropout, 0.2)
        self.assertEqual(config.num_classes, 14)
        self.assertEqual(config.tabular_features, 4)
        self.assertFalse(config.freeze_backbone)

    def test_custom_initialization(self):
        config = CXRModelConfig(
            model="vit_b_16",
            hidden_dims=(256, 128),
            dropout=0.5,
            num_classes=10,
            tabular_features=8,
            freeze_backbone=True,
        )
        self.assertEqual(config.model, "vit_b_16")
        self.assertEqual(config.hidden_dims, (256, 128))
        self.assertEqual(config.dropout, 0.5)
        self.assertEqual(config.num_classes, 10)
        self.assertEqual(config.tabular_features, 8)
        self.assertTrue(config.freeze_backbone)

    def test_from_yaml(self):
        config = CXRModelConfig.from_yaml(str(self.config_path))
        self.assertEqual(config.model, "vit_b_16")
        self.assertEqual(config.hidden_dims, (256, 128))
        self.assertEqual(config.dropout, 0.5)
        self.assertEqual(config.num_classes, 10)
        self.assertEqual(config.tabular_features, 8)
        self.assertTrue(config.freeze_backbone)


class TestCXRModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.img_size = 224
        self.tabular_features = 4
        self.num_classes = 14

        self.img_batch = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        self.tabular_batch = torch.randn(self.batch_size, self.tabular_features)

    def test_registry_has_expected_backbones(self):
        self.assertEqual(set(MODELS), set(BACKBONES))

    def test_init_all_backbones(self):
        for name in BACKBONES:
            model = CXRModel(model=name)
            self.assertIsInstance(model, CXRModel)
            self.assertEqual(model.model_name, name)

    def test_invalid_model_name(self):
        with self.assertRaises(ValueError):
            CXRModel(model="invalid_model")

    def test_forward_all_backbones_multilabel(self):
        for name in BACKBONES:
            model = CXRModel(model=name)
            output = model(self.img_batch, self.tabular_batch)
            self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_forward_all_backbones_binary(self):
        for name in BACKBONES:
            model = CXRModel(model=name, num_classes=1)
            output = model(self.img_batch, self.tabular_batch)
            self.assertEqual(output.shape, (self.batch_size, 1))

    def test_custom_dimensions(self):
        model = CXRModel(model="densenet121", hidden_dims=(256, 128, 64), dropout=0.3)
        output = model(self.img_batch, self.tabular_batch)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_freeze_backbone(self):
        """When freeze_backbone=True, only classifier params should train."""
        model = CXRModel(model="densenet121", freeze_backbone=True)
        # The torchvision backbone lives at model.model.model; the classifier
        # head lives at model.model.classifier.
        for p in model.model.model.parameters():
            self.assertFalse(p.requires_grad)
        for p in model.model.classifier.parameters():
            self.assertTrue(p.requires_grad)

    def test_build_model_helper(self):
        m = build_model("densenet121", num_classes=1, hidden_dims=(32,))
        out = m(self.img_batch, self.tabular_batch)
        self.assertEqual(out.shape, (self.batch_size, 1))


if __name__ == "__main__":
    unittest.main()
