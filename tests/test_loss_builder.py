# -*- coding: utf-8 -*-
"""
Unit tests for loss_builder.py
"""

import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.custom_losses.loss_builder import (
    LossWrapper,
    MultiLoss,
    build_compound_loss_from_config,
    validate_loss_config,
)
from pytorch_segmentation_models_trainer.custom_losses.base_loss import Loss


class TestLossBuilder(unittest.TestCase):
    def test_loss_wrapper_torch_loss(self):
        torch_loss = nn.BCEWithLogitsLoss()
        wrapper = LossWrapper(torch_loss, name="bce")
        self.assertEqual(wrapper.name, "bce")
        self.assertFalse(wrapper.is_custom_loss)

        pred = torch.randn(2, 1, 10, 10)
        gt = torch.empty(2, 1, 10, 10).random_(2)
        val, info = wrapper(pred, gt)
        self.assertIsInstance(val, torch.Tensor)
        self.assertEqual(info, {})

    def test_loss_wrapper_custom_loss(self):
        class MyCustomLoss(Loss):
            def __init__(self):
                super().__init__(name="custom")

            def forward(self, p, g, normalize=True):
                return (p - g).mean(), {"extra": 1}

        custom_loss = MyCustomLoss()
        wrapper = LossWrapper(custom_loss)
        self.assertEqual(wrapper.name, "custom")
        self.assertTrue(wrapper.is_custom_loss)

        p = torch.ones(1)
        g = torch.zeros(1)
        val, info = wrapper(p, g)
        self.assertEqual(val.item(), 1.0)
        self.assertEqual(info["extra"], 1)

    def test_validate_loss_config(self):
        valid_cfg = OmegaConf.create(
            {
                "losses": [
                    {
                        "loss": {"_target_": "torch.nn.L1Loss", "name": "l1"},
                        "weight": 1.0,
                    },
                    {
                        "loss": {"_target_": "torch.nn.MSELoss", "name": "mse"},
                        "weight": 0.5,
                    },
                ]
            }
        )
        self.assertTrue(validate_loss_config(valid_cfg))

        # Duplicate names
        invalid_cfg = OmegaConf.create(
            {
                "losses": [
                    {"loss": {"_target_": "torch.nn.L1Loss", "name": "l1"}},
                    {"loss": {"_target_": "torch.nn.MSELoss", "name": "l1"}},
                ]
            }
        )
        with self.assertRaises(ValueError):
            validate_loss_config(invalid_cfg)

    @patch("pytorch_segmentation_models_trainer.custom_losses.loss_builder.instantiate")
    def test_build_compound_loss(self, mock_inst):
        # Mock instantiate to return a simple nn.Module
        mock_inst.return_value = nn.L1Loss()

        cfg = OmegaConf.create(
            {"losses": [{"loss": {"_target_": "fake", "name": "l1"}, "weight": 1.0}]}
        )

        multi_loss = build_compound_loss_from_config(cfg)
        self.assertEqual(len(multi_loss.loss_funcs), 1)
        self.assertIsInstance(multi_loss.loss_funcs[0], LossWrapper)


if __name__ == "__main__":
    unittest.main()
