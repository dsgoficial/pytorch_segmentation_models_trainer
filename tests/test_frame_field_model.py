# -*- coding: utf-8 -*-
"""
Tests for FrameFieldModel / FrameFieldSegmentationPLModel.

- Inference tests use dummy tensors (fast, no I/O).
- Training tests mock Trainer.fit() to avoid running the full pipeline.
"""

import os
from importlib import import_module
from unittest.mock import MagicMock, patch

import hydra
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from hydra import compose, initialize
from parameterized import parameterized

from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldModel,
    FrameFieldSegmentationPLModel,
)
from pytorch_segmentation_models_trainer.model_loader.model import Model
from pytorch_segmentation_models_trainer.train import train
from pytorch_segmentation_models_trainer.custom_models import models as pytorch_smt_cm

from tests.utils import CustomTestCase

input_model_list = [
    (smp.Unet, {}),
    (smp.DeepLabV3Plus, {}),
    (smp.FPN, {}),
    (smp.PSPNet, {}),
    (smp.PAN, {}),
    (pytorch_smt_cm.UNetResNet, {}),
    (pytorch_smt_cm.HRNetOCRW48, {}),
    (pytorch_smt_cm.HRNetOCRW48, {"pretrained": "cityscapes"}),
    (pytorch_smt_cm.HRNetOCRW48, {"pretrained": "lip"}),
    (pytorch_smt_cm.HRNetOCRW48, {"pretrained": "pascal"}),
]

input_model_overrides_list = [
    (
        [
            "model.segmentation_model._target_=segmentation_models_pytorch.Unet",
            "hyperparameters.epochs=1",
        ],
    ),
    (
        [
            "model.segmentation_model._target_=segmentation_models_pytorch.DeepLabV3Plus",
            "hyperparameters.epochs=1",
        ],
    ),
    (
        [
            "model.segmentation_model._target_=segmentation_models_pytorch.FPN",
            "hyperparameters.epochs=1",
        ],
    ),
    (
        [
            "model.segmentation_model._target_=segmentation_models_pytorch.PSPNet",
            "hyperparameters.epochs=1",
        ],
    ),
    (
        [
            "model.segmentation_model._target_=segmentation_models_pytorch.PAN",
            "hyperparameters.epochs=1",
        ],
    ),
]

input_overrides_list = [
    (
        [
            "model.segmentation_model._target_=pytorch_segmentation_models_trainer.custom_models.models.UNetResNet",
            "hyperparameters.epochs=1",
        ],
    ),
    (
        [
            "model.segmentation_model._target_=pytorch_segmentation_models_trainer.custom_models.models.HRNetOCRW48",
            "+model.segmentation_model.pretrained=cityscapes",
            "hyperparameters.epochs=1",
        ],
    ),
]

current_dir = os.path.dirname(__file__)
frame_field_root_dir = os.path.join(
    current_dir, "testing_data", "data", "frame_field_data"
)


class Test_FrameFieldModel(CustomTestCase):
    def make_inference(self, sample, frame_field_model):
        with torch.no_grad():
            out = frame_field_model(sample)
        self.assertEqual(
            out["seg"].shape,
            torch.Size([sample.shape[0], 3, sample.shape[-2], sample.shape[-1]]),
        )
        self.assertEqual(
            out["crossfield"].shape,
            torch.Size([sample.shape[0], 4, sample.shape[-2], sample.shape[-1]]),
        )

    # ------------------------------------------------------------------ #
    # Inference tests — keep as-is (fast, use dummy tensors)
    # ------------------------------------------------------------------ #

    def test_create_instance(self) -> None:
        model = smp.Unet()
        frame_field_model = FrameFieldModel(model)
        self.assertIsNotNone(frame_field_model)

    @parameterized.expand(input_model_list)
    def test_create_inference_from_model(self, input_model, model_args) -> None:
        model = input_model(**model_args)
        frame_field_model = FrameFieldModel(model)
        self.make_inference(torch.ones([2, 3, 256, 256]), frame_field_model)

    def test_create_model_from_cfg(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(config_name="frame_field_pl_model.yaml")
            frame_field_model = hydra.utils.instantiate(cfg, _recursive_=False)
        self.make_inference(torch.ones([2, 3, 256, 256]), frame_field_model)

    def test_create_pl_model(self) -> None:
        csv_path = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_frame_field.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + csv_path,
                    "train_dataset.root_dir=" + frame_field_root_dir,
                    "val_dataset.input_csv_path=" + csv_path,
                    "val_dataset.root_dir=" + frame_field_root_dir,
                ],
            )
            module_path, class_name = cfg.pl_model._target_.rsplit(".", 1)
            module = import_module(module_path)
            frame_field_model = getattr(module, class_name)(cfg)
        self.assertIsInstance(frame_field_model, FrameFieldSegmentationPLModel)

    # ------------------------------------------------------------------ #
    # Training tests — mock Trainer.fit() to avoid actual training I/O
    # ------------------------------------------------------------------ #

    @parameterized.expand(input_model_overrides_list)
    @patch("pytorch_segmentation_models_trainer.train.Trainer")
    @patch.object(Model, "setup")
    def test_train_frame_field_model(
        self, overrides_list, mock_setup, MockTrainer
    ) -> None:
        mock_trainer_instance = MagicMock(spec=pl.Trainer)
        MockTrainer.return_value = mock_trainer_instance
        csv_path = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_frame_field.yaml",
                overrides=overrides_list
                + [
                    "train_dataset.input_csv_path=" + csv_path,
                    "train_dataset.root_dir=" + frame_field_root_dir,
                    "val_dataset.input_csv_path=" + csv_path,
                    "val_dataset.root_dir=" + frame_field_root_dir,
                ],
            )
            result = train(cfg)
        mock_trainer_instance.fit.assert_called_once()

    @patch("pytorch_segmentation_models_trainer.train.Trainer")
    @patch.object(Model, "setup")
    def test_train_frame_field_model_with_callback(
        self, mock_setup, MockTrainer
    ) -> None:
        mock_trainer_instance = MagicMock(spec=pl.Trainer)
        MockTrainer.return_value = mock_trainer_instance
        csv_path = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_frame_field_with_callback.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + csv_path,
                    "train_dataset.root_dir=" + frame_field_root_dir,
                    "val_dataset.input_csv_path=" + csv_path,
                    "val_dataset.root_dir=" + frame_field_root_dir,
                ],
            )
            result = train(cfg)
        mock_trainer_instance.fit.assert_called_once()

    @parameterized.expand(input_overrides_list)
    @patch("pytorch_segmentation_models_trainer.train.Trainer")
    @patch.object(Model, "setup")
    def test_train_custom_models_with_frame_field(
        self, overrides_list, mock_setup, MockTrainer
    ) -> None:
        mock_trainer_instance = MagicMock(spec=pl.Trainer)
        MockTrainer.return_value = mock_trainer_instance
        csv_path = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_frame_field_custom_model.yaml",
                overrides=overrides_list
                + [
                    "train_dataset.input_csv_path=" + csv_path,
                    "train_dataset.root_dir=" + frame_field_root_dir,
                    "val_dataset.input_csv_path=" + csv_path,
                    "val_dataset.root_dir=" + frame_field_root_dir,
                ],
            )
            result = train(cfg)
        mock_trainer_instance.fit.assert_called_once()

    # ------------------------------------------------------------------ #
    # test_step tests — verify FrameFieldSegmentationPLModel.test_step()
    # ------------------------------------------------------------------ #

    def _make_ff_batch(self):
        """Minimal FrameField-style batch with required keys."""
        B, H, W = 2, 64, 64
        C = 3  # channels must match gt_polygons_image channels
        return {
            "image": torch.randn(B, 3, H, W),
            "gt_polygons_image": torch.zeros(B, C, H, W),
            "class_freq": torch.ones(
                B, C
            ),  # [B, C] required by compute_seg_loss_weights
            "gt_crossfield_angle": torch.zeros(
                B, 1, H, W
            ),  # required by compute_gt_field
        }

    def test_frame_field_validation_step_returns_loss(self) -> None:
        csv_path = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_frame_field.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + csv_path,
                    "train_dataset.root_dir=" + frame_field_root_dir,
                    "val_dataset.input_csv_path=" + csv_path,
                    "val_dataset.root_dir=" + frame_field_root_dir,
                ],
            )
            from importlib import import_module

            module_path, class_name = cfg.pl_model._target_.rsplit(".", 1)
            module = import_module(module_path)
            ff_model = getattr(module, class_name)(cfg)

        ff_model.log = MagicMock()
        batch = self._make_ff_batch()
        loss = ff_model.validation_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)

    def test_frame_field_test_step_returns_loss(self) -> None:
        csv_path = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_frame_field.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + csv_path,
                    "train_dataset.root_dir=" + frame_field_root_dir,
                    "val_dataset.input_csv_path=" + csv_path,
                    "val_dataset.root_dir=" + frame_field_root_dir,
                ],
            )
            from importlib import import_module

            module_path, class_name = cfg.pl_model._target_.rsplit(".", 1)
            module = import_module(module_path)
            ff_model = getattr(module, class_name)(cfg)

        ff_model.log = MagicMock()
        batch = self._make_ff_batch()
        loss = ff_model.test_step(batch, 0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)

    def test_frame_field_test_step_logs_test_prefix(self) -> None:
        csv_path = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_frame_field.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + csv_path,
                    "train_dataset.root_dir=" + frame_field_root_dir,
                    "val_dataset.input_csv_path=" + csv_path,
                    "val_dataset.root_dir=" + frame_field_root_dir,
                ],
            )
            from importlib import import_module

            module_path, class_name = cfg.pl_model._target_.rsplit(".", 1)
            module = import_module(module_path)
            ff_model = getattr(module, class_name)(cfg)

        ff_model.log = MagicMock()
        ff_model.log_dict = MagicMock()
        batch = self._make_ff_batch()
        ff_model.test_step(batch, 0)
        logged_keys = [call.args[0] for call in ff_model.log.call_args_list]
        self.assertIn("loss/test", logged_keys)
        self.assertFalse(any("/val" in k for k in logged_keys))
