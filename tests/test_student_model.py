# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pytorch_segmentation_models_trainer.model_loader.student_model import (
    StudentSegmentationModel,
)


class TestStudentModel(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create(
            {
                "model": {
                    "_target_": "torch.nn.Conv2d",
                    "in_channels": 3,
                    "out_channels": 2,
                    "kernel_size": 1,
                },
                "loss": {"_target_": "torch.nn.CrossEntropyLoss", "reduction": "none"},
                "hyperparameters": {
                    "batch_size": 2,
                    "accelerator": "cpu",
                    "devices": 1,
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                },
                "metrics": [
                    {
                        "_target_": "torchmetrics.JaccardIndex",
                        "task": "multiclass",
                        "num_classes": 2,
                    }
                ],
                "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-3},
            }
        )

    def test_student_model_init(self):
        # Test that it instantiates correctly using the base Model logic
        model = StudentSegmentationModel(self.cfg)
        self.assertIsInstance(model.model, nn.Conv2d)
        self.assertIsInstance(model.loss_function, nn.CrossEntropyLoss)
        self.assertFalse(model.use_soft_labels)

    def test_student_model_training_step_weights(self):
        # Mock dependencies to avoid full instantiation issues in test
        model = StudentSegmentationModel(self.cfg)
        model.log = MagicMock()
        model.log_dict = MagicMock()

        # Batch size 2
        images = torch.randn(2, 3, 32, 32)
        labels = torch.randint(0, 2, (2, 32, 32))
        weights = torch.tensor([1.0, 10.0])

        batch = (images, labels, weights)

        # Training step
        loss = model.training_step(batch, 0)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss > 0)

        # Verify weighting logic manually
        logits = model(images)
        loss_pixel = nn.CrossEntropyLoss(reduction="none")(logits, labels)
        loss_per_image = loss_pixel.mean(dim=(1, 2))
        expected_loss = (loss_per_image * weights).mean()

        self.assertTrue(torch.allclose(loss, expected_loss))

    def test_student_model_soft_labels(self):
        self.cfg.use_soft_labels = True
        model = StudentSegmentationModel(self.cfg)
        model.log = MagicMock()
        model.log_dict = MagicMock()

        images = torch.randn(2, 3, 32, 32)
        # Soft labels: probabilities (B, C, H, W)
        soft_labels = torch.softmax(torch.randn(2, 2, 32, 32), dim=1)
        weights = torch.tensor([1.0, 1.0])

        batch = (images, soft_labels, weights)

        loss = model.training_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss >= 0)


if __name__ == "__main__":
    unittest.main()
