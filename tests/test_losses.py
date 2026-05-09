# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import unittest
from pytorch_segmentation_models_trainer.custom_losses.loss import (
    KnowledgeDistillationLoss,
    MixUpAugmentationLoss,
    LabelSmoothingLoss,
    SmoothCrossEntropyLoss,
    dice_loss,
    tversky_loss,
    focal_tversky_loss,
)


class TestLosses(unittest.TestCase):
    def test_knowledge_distillation_loss(self):
        criterion = nn.CrossEntropyLoss()
        kd_loss = KnowledgeDistillationLoss(alpha=0.5, T=2.0, criterion=criterion)

        logits = torch.randn(2, 4, 10, 10)
        target = torch.randint(0, 4, (2, 10, 10))
        teacher_logits = torch.randn(2, 4, 10, 10)

        loss = kd_loss(logits, target, teacher_logits)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)

    def test_mixup_augmentation_loss(self):
        criterion = nn.CrossEntropyLoss()
        mixup_loss = MixUpAugmentationLoss(criterion)

        logits = torch.randn(2, 4, 10, 10)
        target_a = torch.randint(0, 4, (2, 10, 10))
        target_b = torch.randint(0, 4, (2, 10, 10))
        lmbd = 0.7

        loss = mixup_loss(logits, (target_a, target_b, lmbd))
        self.assertIsInstance(loss, torch.Tensor)

    def test_label_smoothing_loss(self):
        ls_loss = LabelSmoothingLoss(n_classes=4, smoothing=0.1, dim=1)
        logits = torch.randn(2, 4, 10, 10)
        target = torch.randint(0, 4, (2, 10, 10))

        loss = ls_loss(logits, target)
        self.assertIsInstance(loss, torch.Tensor)

    def test_smooth_cross_entropy_loss(self):
        sce_loss = SmoothCrossEntropyLoss(smoothing=0.1)
        # SmoothCrossEntropyLoss now handles (B, C, H, W)
        logits = torch.randn(2, 4, 10, 10)
        target = torch.randint(0, 4, (2, 10, 10))

        loss = sce_loss(logits, target)
        self.assertIsInstance(loss, torch.Tensor)

    def test_dice_loss(self):
        y_pred = torch.sigmoid(torch.randn(2, 1, 10, 10))
        y_true = torch.randint(0, 2, (2, 1, 10, 10)).float()
        loss = dice_loss(y_pred, y_true)
        self.assertEqual(loss.shape, (2, 1))
        self.assertTrue((loss >= 0).all())

    def test_tversky_loss(self):
        y_pred = torch.sigmoid(torch.randn(2, 1, 10, 10))
        y_true = torch.randint(0, 2, (2, 1, 10, 10)).float()
        loss = tversky_loss(y_pred, y_true, alpha=0.5, beta=0.5)
        self.assertEqual(loss.shape, (2, 1))

    def test_focal_tversky_loss(self):
        y_pred = torch.sigmoid(torch.randn(2, 1, 10, 10))
        y_true = torch.randint(0, 2, (2, 1, 10, 10)).float()
        loss = focal_tversky_loss(y_pred, y_true)
        self.assertEqual(loss.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
