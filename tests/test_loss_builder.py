# -*- coding: utf-8 -*-
"""
Unit tests for loss_builder.py
"""

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.custom_losses.loss_builder import (
    LossWrapper,
    MultiLoss,
    build_compound_loss_from_config,
    build_loss_from_config,
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

            def reset_norm(self):
                self.reset_called = True

            def update_norm(self, pred_batch, gt_batch, nums):
                self.update_args = (pred_batch, gt_batch, nums)

            def sync(self, world_size):
                self.sync_world_size = world_size

        custom_loss = MyCustomLoss()
        wrapper = LossWrapper(custom_loss)
        self.assertEqual(wrapper.name, "custom")
        self.assertTrue(wrapper.is_custom_loss)

        p = torch.ones(1)
        g = torch.zeros(1)
        val, info = wrapper(p, g)
        self.assertEqual(val.item(), 1.0)
        self.assertEqual(info["extra"], 1)
        wrapper.reset_norm()
        wrapper.update_norm("p", "g", 3)
        wrapper.sync(2)
        self.assertTrue(custom_loss.reset_called)
        self.assertEqual(custom_loss.update_args, ("p", "g", 3))
        self.assertEqual(custom_loss.sync_world_size, 2)

    def test_loss_wrapper_fallback_to_single_argument(self):
        class SingleArgLoss(nn.Module):
            def forward(self, pred):
                return pred.sum()

        wrapper = LossWrapper(SingleArgLoss(), name="single")
        val, info = wrapper(torch.ones(2), torch.zeros(2))
        self.assertEqual(val.item(), 2.0)
        self.assertEqual(info, {})

    def test_loss_wrapper_raises_when_both_calls_fail(self):
        class BrokenLoss(nn.Module):
            def forward(self, *args, **kwargs):
                raise TypeError("broken")

        wrapper = LossWrapper(BrokenLoss(), name="broken")
        with self.assertRaises(TypeError):
            wrapper(torch.ones(1), torch.zeros(1))

    def test_loss_wrapper_repr(self):
        wrapper = LossWrapper(nn.MSELoss(), name="mse")
        self.assertEqual(repr(wrapper), "LossWrapper(mse, custom=False)")

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

        empty_cfg = OmegaConf.create({})
        with self.assertRaises(ValueError):
            validate_loss_config(empty_cfg)

        no_losses_cfg = OmegaConf.create({"losses": []})
        with self.assertRaises(ValueError):
            validate_loss_config(no_losses_cfg)

        missing_target_cfg = OmegaConf.create({"losses": [{"loss": {"name": "l1"}}]})
        with self.assertRaises(ValueError):
            validate_loss_config(missing_target_cfg)

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
        self.assertEqual(multi_loss.weights, [1.0])

    @patch("pytorch_segmentation_models_trainer.custom_losses.loss_builder.instantiate")
    def test_build_compound_loss_custom_loss_and_preprocesses(self, mock_inst):
        class MyLoss(Loss):
            def __init__(self):
                super().__init__(name="custom")

            def forward(self, pred, gt, normalize=True):
                return torch.tensor(0.0), {}

        mock_inst.return_value = MyLoss()
        cfg = OmegaConf.create(
            {
                "epoch_thresholds": [0, 1],
                "pre_processes": ["p1"],
                "losses": [
                    {
                        "loss": {"_target_": "fake", "name": "custom"},
                        "weight": [1.0, 2.0],
                    }
                ],
            }
        )

        multi_loss = build_compound_loss_from_config(cfg)
        self.assertEqual(len(multi_loss.weights), 1)
        self.assertTrue(callable(multi_loss.weights[0]))
        self.assertEqual(multi_loss.pre_processes, ["p1"])
        self.assertEqual(float(multi_loss.weights[0](0.0)), 1.0)
        self.assertEqual(float(multi_loss.weights[0](1.0)), 2.0)

    def test_build_compound_loss_requires_losses_field(self):
        with self.assertRaises(ValueError):
            build_compound_loss_from_config(OmegaConf.create({}))

    def test_build_compound_loss_requires_loss_field(self):
        cfg = OmegaConf.create({"losses": [{}]})
        with self.assertRaises(ValueError):
            build_compound_loss_from_config(cfg)

    @patch("pytorch_segmentation_models_trainer.custom_losses.loss_builder.instantiate")
    def test_build_loss_from_config_new_and_legacy_and_invalid(self, mock_inst):
        mock_inst.return_value = nn.L1Loss()
        new_cfg = OmegaConf.create(
            {
                "loss_params": {
                    "compound_loss": {
                        "losses": [
                            {
                                "loss": {"_target_": "fake", "name": "l1"},
                                "weight": 1.0,
                            }
                        ]
                    }
                }
            }
        )
        self.assertIsInstance(build_loss_from_config(new_cfg), MultiLoss)

        legacy_cfg = OmegaConf.create({"loss_params": {"multi_loss": {"x": 1}}})
        with patch(
            "pytorch_segmentation_models_trainer.custom_losses.base_loss.build_combined_loss",
            return_value="legacy",
        ) as mock_build:
            self.assertEqual(build_loss_from_config(legacy_cfg), "legacy")
            mock_build.assert_called_once_with(legacy_cfg)

        invalid_cfg = OmegaConf.create({"loss_params": {}})
        with self.assertRaises(ValueError):
            build_loss_from_config(invalid_cfg)


if __name__ == "__main__":
    unittest.main()
