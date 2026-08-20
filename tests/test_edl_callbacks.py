# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pathlib import Path

from pytorch_segmentation_models_trainer.custom_callbacks.edl_callbacks import (
    EvidentialWarmupCallback,
    EvidentialUncertaintyVisualizationCallback,
)
from pytorch_segmentation_models_trainer.custom_models.edl_wrapper import (
    EvidentialWrapper,
)


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 64, 3), nn.Conv2d(64, 128, 3))

    def set_encoder_trainable(self, trainable):
        for p in self.encoder.parameters():
            p.requires_grad = trainable


class MockPLModule:
    def __init__(self, model):
        self.model = model
        self._device = torch.device("cpu")

    def set_encoder_trainable(self, trainable):
        if hasattr(self.model, "set_encoder_trainable"):
            self.model.set_encoder_trainable(trainable)

    @property
    def device(self):
        return self._device

    def eval(self):
        pass

    def train(self, mode=True):
        pass


class TestEvidentialWarmupCallback(unittest.TestCase):
    def setUp(self):
        self.trainer = MagicMock(spec=pl.Trainer)
        self.model = MockModel()
        self.pl_module = MockPLModule(self.model)

    def test_init(self):
        cb = EvidentialWarmupCallback(
            warmup_epochs=2, freeze_encoder=True, partial_unfreeze_epoch=4
        )
        self.assertEqual(cb.warmup_epochs, 2)
        self.assertTrue(cb.freeze_encoder)
        self.assertEqual(cb.partial_unfreeze_epoch, 4)

    def test_on_fit_start_no_freeze(self):
        cb = EvidentialWarmupCallback(freeze_encoder=False)
        cb.on_fit_start(self.trainer, self.pl_module)
        self.assertEqual(cb._phase, "init")

    def test_on_fit_start_freeze_warmup(self):
        self.trainer.current_epoch = 0
        cb = EvidentialWarmupCallback(warmup_epochs=2, freeze_encoder=True)
        cb.on_fit_start(self.trainer, self.pl_module)
        self.assertEqual(cb._phase, "warmup")
        for p in self.model.encoder.parameters():
            self.assertFalse(p.requires_grad)

    def test_on_fit_start_freeze_full(self):
        self.trainer.current_epoch = 5
        cb = EvidentialWarmupCallback(warmup_epochs=2, freeze_encoder=True)
        cb.on_fit_start(self.trainer, self.pl_module)
        self.assertEqual(cb._phase, "full")

    def test_on_train_epoch_start_no_freeze(self):
        cb = EvidentialWarmupCallback(freeze_encoder=False)
        cb.on_train_epoch_start(self.trainer, self.pl_module)
        self.assertEqual(cb._phase, "init")

    def test_on_train_epoch_start_warmup_to_partial(self):
        self.trainer.current_epoch = 2
        cb = EvidentialWarmupCallback(
            warmup_epochs=2, freeze_encoder=True, partial_unfreeze_epoch=4
        )
        cb._phase = "warmup"
        cb.on_train_epoch_start(self.trainer, self.pl_module)
        self.assertEqual(cb._phase, "partial")
        # Check if last stage is unfrozen
        for p in list(self.model.encoder.children())[-1].parameters():
            self.assertTrue(p.requires_grad)

    def test_on_train_epoch_start_to_full(self):
        self.trainer.current_epoch = 4
        cb = EvidentialWarmupCallback(
            warmup_epochs=2, freeze_encoder=True, partial_unfreeze_epoch=4
        )
        cb._phase = "partial"
        cb.on_train_epoch_start(self.trainer, self.pl_module)
        self.assertEqual(cb._phase, "full")
        for p in self.model.encoder.parameters():
            self.assertTrue(p.requires_grad)

    def test_set_encoder_trainable_edl_wrapper(self):
        mock_wrapper = MagicMock(spec=EvidentialWrapper)
        self.pl_module.model = mock_wrapper
        cb = EvidentialWarmupCallback()
        cb._set_encoder_trainable(self.pl_module, trainable=True)
        mock_wrapper.unfreeze_encoder.assert_called_once()
        cb._set_encoder_trainable(self.pl_module, trainable=False)
        mock_wrapper._freeze_encoder.assert_called_once()

    def test_partial_unfreeze_leaf_module(self):
        cb = EvidentialWarmupCallback()
        leaf_encoder = nn.Conv2d(3, 10, 3)
        for p in leaf_encoder.parameters():
            p.requires_grad = False
        mock_model = MagicMock()
        mock_model.encoder = leaf_encoder
        self.pl_module.model = mock_model

        cb._partial_unfreeze_encoder(self.pl_module)
        for p in leaf_encoder.parameters():
            self.assertTrue(p.requires_grad)

    def test_partial_unfreeze_edl_wrapper(self):
        cb = EvidentialWarmupCallback()
        mock_wrapper = MagicMock(spec=EvidentialWrapper)
        mock_encoder = nn.Sequential(nn.Conv2d(3, 10, 3))
        mock_wrapper._find_encoder.return_value = mock_encoder
        self.pl_module.model = mock_wrapper

        cb._partial_unfreeze_encoder(self.pl_module)
        mock_wrapper._find_encoder.assert_called_once()
        for p in mock_encoder.parameters():
            self.assertTrue(p.requires_grad)

    def test_partial_unfreeze_no_encoder(self):
        cb = EvidentialWarmupCallback()
        mock_model = MagicMock()
        # Explicitly set to None to avoid MagicMock auto-creating mocks
        mock_model.encoder = None
        mock_model.backbone = None
        self.pl_module.model = mock_model
        with self.assertLogs(
            "pytorch_segmentation_models_trainer.custom_callbacks.edl_callbacks",
            level="WARNING",
        ) as cm:
            cb._partial_unfreeze_encoder(self.pl_module)
        self.assertIn("could not find encoder", cm.output[0])

    def test_set_encoder_trainable_no_method(self):
        cb = EvidentialWarmupCallback()

        # Create an object without the method
        class PlainPLModule:
            def __init__(self):
                self.model = MagicMock()

        pl_module = PlainPLModule()
        with self.assertLogs(
            "pytorch_segmentation_models_trainer.custom_callbacks.edl_callbacks",
            level="WARNING",
        ) as cm:
            cb._set_encoder_trainable(pl_module, trainable=True)
        self.assertIn("could not freeze/unfreeze encoder", cm.output[0])


class TestEvidentialUncertaintyVisualizationCallback(unittest.TestCase):
    def setUp(self):
        self.trainer = MagicMock(spec=pl.Trainer)
        self.model = MagicMock(spec=EvidentialWrapper)
        self.pl_module = MockPLModule(self.model)

        self.batch = {
            "image": torch.randn(4, 3, 32, 32),
            "mask": torch.randint(0, 2, (4, 32, 32)),
        }

    def test_on_validation_batch_start(self):
        cb = EvidentialUncertaintyVisualizationCallback()
        cb.on_validation_batch_start(self.trainer, self.pl_module, self.batch, 0)
        self.assertIsNotNone(cb._val_batch)

        # Second batch should not overwrite
        new_batch = {"image": torch.zeros(1)}
        cb.on_validation_batch_start(self.trainer, self.pl_module, new_batch, 1)
        torch.testing.assert_close(cb._val_batch["image"], self.batch["image"])

    def test_on_validation_epoch_end_skips(self):
        cb = EvidentialUncertaintyVisualizationCallback(log_every_n_epochs=5)

        # Skip by epoch
        self.trainer.current_epoch = 1
        cb.on_validation_epoch_end(self.trainer, self.pl_module)
        self.model.assert_not_called()

        # Skip by no batch
        self.trainer.current_epoch = 5
        cb._val_batch = None
        cb.on_validation_epoch_end(self.trainer, self.pl_module)
        self.model.assert_not_called()

        # Skip by not EDL wrapper
        cb._val_batch = self.batch
        self.pl_module.model = MagicMock()  # Not EvidentialWrapper
        cb.on_validation_epoch_end(self.trainer, self.pl_module)
        self.pl_module.model.assert_not_called()

    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.edl_callbacks.plt.subplots"
    )
    def test_on_validation_epoch_end_execution(self, mock_subplots):
        mock_fig = MagicMock()
        mock_axes = MagicMock()  # Use a single MagicMock that supports indexing
        mock_subplots.return_value = (mock_fig, mock_axes)

        cb = EvidentialUncertaintyVisualizationCallback(
            num_images=2, log_every_n_epochs=1
        )
        cb._val_batch = self.batch
        self.trainer.current_epoch = 0

        # Mock model output
        self.model.side_effect = lambda x: {
            "probs": torch.softmax(torch.randn(x.shape[0], 2, 32, 32), dim=1),
            "uncertainty": torch.rand(x.shape[0], 1, 32, 32),
        }

        with patch.object(cb, "_log_figure") as mock_log:
            cb.on_validation_epoch_end(self.trainer, self.pl_module)
            mock_log.assert_called_once()

    def test_denormalize(self):
        cb = EvidentialUncertaintyVisualizationCallback(
            norm_params={"mean": [0.5], "std": [0.2]}
        )
        img = torch.tensor([[[0.0]]])  # 1,1,1
        denorm = cb._denormalize(img)
        self.assertAlmostEqual(denorm.item(), 0.5)

        cb_no_norm = EvidentialUncertaintyVisualizationCallback(norm_params=None)
        self.assertTrue(torch.equal(cb_no_norm._denormalize(img), img))

    def test_make_figure_various_shapes(self):
        cb = EvidentialUncertaintyVisualizationCallback(num_images=1)
        images = torch.randn(1, 1, 32, 32)
        output = {
            "probs": torch.randn(1, 2, 32, 32),
            "uncertainty": torch.randn(1, 1, 32, 32),
        }
        masks = torch.randn(1, 2, 32, 32)  # floating point masks

        fig = cb._make_figure(images, output, masks, 1)
        self.assertIsNotNone(fig)

        # Test with 3+ channels
        images_rgb = torch.randn(1, 3, 32, 32)
        fig_rgb = cb._make_figure(images_rgb, output, masks, 1)
        self.assertIsNotNone(fig_rgb)

        # Test with 2 channels
        images_2ch = torch.randn(1, 2, 32, 32)
        fig_2ch = cb._make_figure(images_2ch, output, masks, 1)
        self.assertIsNotNone(fig_2ch)

        # Test with 3D integer mask (B, 1, H, W -> gt.dim() == 3)
        masks_3d = torch.randint(0, 2, (1, 1, 32, 32))
        fig_3d_mask = cb._make_figure(images, output, masks_3d, 1)
        self.assertIsNotNone(fig_3d_mask)

    def test_log_figure_tensorboard(self):
        cb = EvidentialUncertaintyVisualizationCallback()
        mock_logger = MagicMock()
        mock_logger.experiment.add_image = MagicMock()
        self.trainer.logger = mock_logger

        fig = MagicMock()
        fig.canvas.tostring_rgb.return_value = np.zeros(
            (100 * 100 * 3,), dtype=np.uint8
        ).tobytes()
        fig.canvas.get_width_height.return_value = (100, 100)

        cb._log_figure(self.trainer, fig, 0)
        mock_logger.experiment.add_image.assert_called_once()

    def test_log_figure_wandb(self):
        cb = EvidentialUncertaintyVisualizationCallback()
        mock_logger = MagicMock()
        # Ensure it doesn't find add_image to trigger wandb branch
        if hasattr(mock_logger.experiment, "add_image"):
            del mock_logger.experiment.add_image
        self.trainer.logger = mock_logger

        fig = MagicMock()
        fig.canvas.tostring_rgb.return_value = np.zeros(
            (100 * 100 * 3,), dtype=np.uint8
        ).tobytes()
        fig.canvas.get_width_height.return_value = (100, 100)

        # Mock sys.modules for wandb

        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb._log_figure(self.trainer, fig, 0)
            mock_logger.experiment.log.assert_called_once()

    def test_log_figure_wandb_error(self):
        cb = EvidentialUncertaintyVisualizationCallback()
        mock_logger = MagicMock()
        if hasattr(mock_logger.experiment, "add_image"):
            del mock_logger.experiment.add_image
        # Trigger AttributeError in log_figure
        mock_logger.experiment.log.side_effect = AttributeError("Mock error")
        self.trainer.logger = mock_logger
        self.trainer.log_dir = None

        fig = MagicMock()
        fig.canvas.tostring_rgb.return_value = np.zeros(
            (100 * 100 * 3,), dtype=np.uint8
        ).tobytes()
        fig.canvas.get_width_height.return_value = (100, 100)

        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb._log_figure(self.trainer, fig, 0)
            # Should not raise error, just pass

    def test_log_figure_fallback(self):
        import shutil

        tmp_dir = Path("tmp_test_logs_fallback")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        cb = EvidentialUncertaintyVisualizationCallback()
        self.trainer.logger = None
        self.trainer.log_dir = str(tmp_dir)

        fig = MagicMock()
        fig.canvas.tostring_rgb.return_value = np.zeros(
            (100 * 100 * 3,), dtype=np.uint8
        ).tobytes()
        fig.canvas.get_width_height.return_value = (100, 100)
        # Mock savefig to actually create a file
        fig.savefig.side_effect = (
            lambda path, **kwargs: Path(path).parent.mkdir(parents=True, exist_ok=True)
            or Path(path).touch()
        )

        cb._log_figure(self.trainer, fig, 0)
        self.assertTrue((tmp_dir / "edl_uncertainty" / "epoch_0000.png").exists())

        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
