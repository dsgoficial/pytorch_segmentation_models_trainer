# -*- coding: utf-8 -*-
"""
Unit tests for model.py
"""

import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
import torchmetrics
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.model_loader.model import Model


class TestModelLoader(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create(
            {
                "model": {"_target_": "torch.nn.Identity"},
                "loss": {"_target_": "torch.nn.BCEWithLogitsLoss"},
                "hyperparameters": {
                    "batch_size": 2,
                    "accelerator": "cpu",
                    "devices": 1,
                },
                "metrics": [
                    {"_target_": "torchmetrics.JaccardIndex", "task": "binary"}
                ],
            }
        )

    def test_model_init(self):
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.model.instantiate"
        ) as mock_inst:
            # model, loss, metric
            mock_model = nn.Identity()
            mock_loss = nn.BCEWithLogitsLoss()
            mock_metric = torchmetrics.JaccardIndex(task="binary")

            mock_inst.side_effect = [mock_model, mock_loss, mock_metric]

            model_pl = Model(self.cfg)
            self.assertIsNotNone(model_pl.model)
            self.assertIsNotNone(model_pl.loss_function)

    def test_shared_step(self):
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.model.instantiate"
        ) as mock_inst:
            mock_model = lambda x: torch.sigmoid(
                torch.randn(x.shape[0], 1, x.shape[2], x.shape[3])
            )
            mock_loss = nn.BCEWithLogitsLoss()
            mock_metric = torchmetrics.JaccardIndex(task="binary", ignore_index=255)
            mock_inst.side_effect = [mock_model, mock_loss, mock_metric]

            model_pl = Model(self.cfg)
            model_pl.log = MagicMock()
            model_pl.log_dict = MagicMock()

            batch = {
                "image": torch.randn(2, 3, 32, 32),
                "mask": torch.randint(0, 2, (2, 1, 32, 32)).float(),
            }
            loss = model_pl.training_step(batch, 0)
            self.assertIsInstance(loss, torch.Tensor)

    def test_configure_optimizers(self):
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.model.instantiate"
        ) as mock_inst:
            mock_model = nn.Conv2d(3, 1, 3)
            mock_loss = nn.BCEWithLogitsLoss()
            mock_metric = torchmetrics.JaccardIndex(task="binary")
            mock_inst.side_effect = [mock_model, mock_loss, mock_metric]

            self.cfg.optimizer = {"_target_": "torch.optim.Adam", "lr": 1e-3}
            model_pl = Model(self.cfg)

            with patch(
                "pytorch_segmentation_models_trainer.model_loader.model.instantiate",
                return_value=torch.optim.Adam(model_pl.parameters(), lr=1e-3),
            ):
                optimizers, schedulers = model_pl.configure_optimizers()
                self.assertEqual(len(optimizers), 1)

    def test_get_tta_augmentations(self):
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.model.instantiate"
        ) as mock_inst:
            mock_model = nn.Identity()
            mock_loss = nn.BCEWithLogitsLoss()
            mock_metric = torchmetrics.JaccardIndex(task="binary")
            mock_inst.side_effect = [mock_model, mock_loss, mock_metric]

            self.cfg.tta_mode = "d4"
            model_pl = Model(self.cfg)
            augs = model_pl._get_tta_augmentations()
            self.assertEqual(len(augs), 8)


if __name__ == "__main__":
    unittest.main()
