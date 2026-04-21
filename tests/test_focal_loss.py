"""Unit tests for the focal loss module"""

import unittest

import numpy as np
import torch
import torch.nn.functional as F
from mm_cxr_diag.losses import FocalLoss, reweight


def logit(p: torch.Tensor) -> torch.Tensor:
    """Convert probability to logit."""
    return torch.log(p / (1 - p))


class TestFocalLoss(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.class_size_list = [645, 387, 232, 139]
        self.per_class_weight = reweight(self.class_size_list)

    def test_reweighting(self) -> None:
        """Test the reweighting function."""
        self.assertTrue(torch.is_tensor(self.per_class_weight))
        self.assertEqual(self.per_class_weight.shape[0], len(self.class_size_list))
        expected = np.array([0.4043, 0.6652, 1.1011, 1.8294])
        np.testing.assert_array_almost_equal(self.per_class_weight.numpy(), expected, 4)

    def test_focal_loss_equals_bce_loss(self) -> None:
        """Test that focal loss equals BCE loss when gamma=0."""
        inputs = logit(
            torch.tensor(
                [
                    [0.95, 0.05, 0.12, 0.05],
                    [0.09, 0.95, 0.36, 0.11],
                    [0.06, 0.12, 0.56, 0.07],
                    [0.09, 0.15, 0.25, 0.45],
                ],
                dtype=torch.float32,
            )
        )

        targets = torch.zeros_like(inputs)
        targets[0, 0] = 1
        targets[1, 1] = 1
        targets[2, 2] = 1
        targets[3, 3] = 1

        focalloss = FocalLoss(weight=self.per_class_weight, gamma=0.0)
        focal_loss = focalloss.forward(input=inputs, target=targets)

        bce_loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            weight=self.per_class_weight.unsqueeze(0).expand_as(inputs),
            reduction="mean",
        )

        self.assertAlmostEqual(bce_loss.item(), focal_loss.item(), places=6)

    def test_focal_loss_reduces_easy_examples(self):
        """Test that focal loss down-weights easy examples."""
        inputs = logit(torch.tensor([[0.95, 0.05], [0.51, 0.49]], dtype=torch.float32))
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

        bce_loss = FocalLoss(gamma=0.0)
        focal_loss = FocalLoss(gamma=2.0)

        bce_ratio = bce_loss(inputs[1:], targets[1:]) / bce_loss(
            inputs[:1], targets[:1]
        )
        focal_ratio = focal_loss(inputs[1:], targets[1:]) / focal_loss(
            inputs[:1], targets[:1]
        )

        self.assertGreater(focal_ratio, bce_ratio)
