# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pathlib import Path

from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import (
    WarmupCallback,
    FrameFieldOnlyCrossfieldWarmupCallback,
    ComputeWeightNormLossesCallback,
    FrameFieldComputeWeightNormLossesCallback,
    FrameFieldPolygonizerCallback,
    ActiveSkeletonsPolygonizerCallback,
    ModPolymapperPolygonizerCallback,
    EMACallback,
    MixStyleCallback,
)


class TestTrainingCallbacks(unittest.TestCase):
    def setUp(self):
        self.trainer = MagicMock()
        self.trainer.current_epoch = 0
        self.trainer.global_rank = 0
        self.trainer.world_size = 1

        self.pl_module = MagicMock()
        self.pl_module.device = torch.device("cpu")
        self.pl_module.cfg = MagicMock()
        self.pl_module.cfg.hyperparameters.batch_size = 2
        self.pl_module.trainer.world_size = 1

    def test_warmup_callback(self):
        cb = WarmupCallback(warmup_epochs=2)

        # Test on_fit_start already warmed up
        self.trainer.current_epoch = 2
        cb.on_fit_start(self.trainer, self.pl_module)
        self.assertTrue(cb.warmed_up)

        # Test lifecycle
        cb = WarmupCallback(warmup_epochs=2)
        self.trainer.current_epoch = 0
        cb.on_fit_start(self.trainer, self.pl_module)
        self.assertFalse(cb.warmed_up)

        cb.on_train_epoch_start(self.trainer, self.pl_module)
        # Should not freeze yet because current_epoch (0) < warmup_epochs - 1 (1)
        self.pl_module.set_encoder_trainable.assert_not_called()

        self.trainer.current_epoch = 1
        cb.on_train_epoch_start(self.trainer, self.pl_module)
        self.pl_module.set_encoder_trainable.assert_called_with(trainable=False)

        cb.on_train_epoch_end(self.trainer, self.pl_module)
        self.pl_module.set_encoder_trainable.assert_called_with(trainable=True)
        self.assertTrue(cb.warmed_up)

    def test_frame_field_warmup_callback(self):
        cb = FrameFieldOnlyCrossfieldWarmupCallback(warmup_epochs=1)
        cb.on_fit_start(self.trainer, self.pl_module)

        self.trainer.current_epoch = 0
        cb.on_train_epoch_start(self.trainer, self.pl_module)
        self.pl_module.set_all_but_crossfield_trainable.assert_called_with(
            trainable=False
        )

        cb.on_train_epoch_end(self.trainer, self.pl_module)
        self.pl_module.set_all_but_crossfield_trainable.assert_called_with(
            trainable=True
        )

    def test_compute_weight_norm_losses_callback_skips(self):
        cb = ComputeWeightNormLossesCallback()

        # 1. Skip if already initialized
        cb.loss_norm_is_initializated = True
        cb.on_train_start(self.trainer, self.pl_module)

        # 2. Skip if should not normalize
        cb.loss_norm_is_initializated = False
        self.pl_module.check_if_should_normalize.return_value = False
        cb.on_train_start(self.trainer, self.pl_module)

        # 3. Skip if no loss_params
        del self.pl_module.check_if_should_normalize
        del self.pl_module.cfg.loss_params
        cb.on_train_start(self.trainer, self.pl_module)

        # 4. Skip if already updated
        self.pl_module.cfg.loss_params = MagicMock()
        self.pl_module.loss_function.norm_updated = True
        cb.on_train_start(self.trainer, self.pl_module)

    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.DataLoader"
    )
    def test_compute_weight_norm_losses_execution(self, mock_dl):
        cb = ComputeWeightNormLossesCallback()
        self.pl_module.loss_function.norm_updated = False
        self.pl_module.cfg.loss_params.compound_loss.normalization_params = {
            "max_samples": 4
        }

        # Mock dataloader
        mock_batch = MagicMock()
        mock_batch.values.return_value = (
            torch.zeros(2, 3, 10, 10),
            torch.zeros(2, 1, 10, 10),
        )
        mock_dl.return_value = [mock_batch, mock_batch]

        cb.on_train_start(self.trainer, self.pl_module)
        self.assertTrue(cb.loss_norm_is_initializated)
        self.pl_module.loss_function.update_norm.assert_called()

    def test_frame_field_compute_weight_norm_losses_callback(self):
        cb = FrameFieldComputeWeightNormLossesCallback()
        self.pl_module.cfg.loss_params.multiloss.normalization_params.min_samples = 2
        self.pl_module.cfg.loss_params.multiloss.normalization_params.max_samples = 4
        self.pl_module.train_dataloader.return_value = [MagicMock()]

        cb.on_fit_start(self.trainer, self.pl_module)
        self.assertTrue(cb.loss_norm_is_initializated)
        self.pl_module.compute_loss_norms.assert_called()

    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.instantiate_polygonizer"
    )
    def test_frame_field_polygonizer_callback(self, mock_inst):
        cb = FrameFieldPolygonizerCallback()
        mock_poly = MagicMock()
        mock_inst.return_value = mock_poly

        batch = {"path": ["test.tif"]}
        outputs = (torch.zeros(1, 1, 10, 10), torch.zeros(1, 2, 10, 10))

        cb.on_predict_batch_end(self.trainer, self.pl_module, outputs, batch, 0)
        mock_poly.process.assert_called()

    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.instantiate_polygonizer"
    )
    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.active_skeletons.polygonize"
    )
    def test_active_skeletons_polygonizer_callback(self, mock_poly_func, mock_inst):
        cb = ActiveSkeletonsPolygonizerCallback()
        mock_writer = MagicMock()
        mock_inst.return_value.data_writer = mock_writer
        # Mock shapely polygon
        mock_poly = MagicMock()
        mock_poly.geom_type = "Polygon"
        mock_poly_func.return_value = ([mock_poly], [0.9])

        batch = {"path": ["test.tif"]}
        outputs = (torch.zeros(1, 1, 10, 10), torch.zeros(1, 2, 10, 10))

        cb.on_predict_batch_end(self.trainer, self.pl_module, outputs, batch, 0)
        mock_writer.write_data.assert_called()

    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.instantiate_polygonizer"
    )
    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.rasterio.open"
    )
    def test_mod_polymapper_polygonizer_callback(self, mock_raster, mock_inst):
        cb = ModPolymapperPolygonizerCallback(convert_output_to_world_coords=True)
        mock_poly = MagicMock()
        mock_inst.return_value = mock_poly

        # Mock rasterio profile
        mock_raster.return_value.__enter__.return_value.profile = {"crs": "epsg:4326"}

        batch = {"path": ["test.tif"]}
        outputs = [{"polygonrnn_output": torch.zeros(1, 10)}]

        cb.on_predict_batch_end(self.trainer, self.pl_module, outputs, batch, 0)
        mock_poly.process.assert_called()

    def test_ema_callback_skips(self):
        cb = EMACallback(decay=0.999)
        self.trainer.global_step = 0
        cb._last_global_step = 0

        # Should skip because step hasn't changed
        cb.on_train_batch_end(self.trainer, self.pl_module, None, None, 0)
        self.assertEqual(cb._last_global_step, 0)

    def test_mixstyle_callback_skips(self):
        cb = MixStyleCallback(p=0.0)  # p=0 means skip
        cb._training = True
        output = torch.randn(2, 3, 10, 10)
        res = cb._mixstyle_hook(None, None, output)
        self.assertTrue(torch.equal(res, output))

        # Multi-output skip
        res_multi = cb._mixstyle_hook(None, None, (output, output))
        self.assertEqual(len(res_multi), 2)

    def test_mixstyle_callback_no_stages(self):
        cb = MixStyleCallback()
        self.pl_module.model.encoder = MagicMock()
        # Mock encoder as having no 'stages' or 'model.stages'
        del self.pl_module.model.encoder.stages
        del self.pl_module.model.encoder.model.stages

        with self.assertLogs(
            "pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks",
            level="WARNING",
        ) as cm:
            cb.on_fit_start(self.trainer, self.pl_module)
        self.assertIn("could not find encoder stages", cm.output[0])

    def test_ema_callback_update(self):
        cb = EMACallback(decay=0.9)
        # Mock pl_module with parameters
        param = nn.Parameter(torch.ones(10))
        self.pl_module.named_parameters.return_value = [("weight", param)]

        # 1. on_fit_start to init shadow
        cb.on_fit_start(self.trainer, self.pl_module)
        self.assertIn("weight", cb._shadow)

        # 2. on_train_batch_end to update
        self.trainer.global_step = 1
        cb.on_train_batch_end(self.trainer, self.pl_module, None, None, 0)
        torch.testing.assert_close(cb._shadow["weight"], torch.ones(10))

        # 3. Validation swap
        cb.on_validation_epoch_start(self.trainer, self.pl_module)
        self.assertIn("weight", cb._original)

        cb.on_validation_epoch_end(self.trainer, self.pl_module)
        self.assertEqual(len(cb._original), 0)

    def test_mixstyle_callback_execution(self):
        cb = MixStyleCallback(p=1.0)  # Always apply
        cb._training = True
        output = torch.randn(4, 3, 10, 10)
        res = cb._mixstyle_hook(None, None, output)
        self.assertEqual(res.shape, output.shape)
        self.assertFalse(torch.equal(res, output))

    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.instantiate_polygonizer"
    )
    def test_frame_field_polygonizer_empty_outputs(self, mock_inst):
        cb = FrameFieldPolygonizerCallback()
        batch = {"path": ["test.tif"]}
        # Empty/None outputs
        cb.on_predict_batch_end(self.trainer, self.pl_module, (None, None), batch, 0)
        mock_inst.assert_not_called()

    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.instantiate_polygonizer"
    )
    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.active_skeletons.polygonize"
    )
    def test_active_skeletons_polygonizer_error_handling(
        self, mock_poly_func, mock_inst
    ):
        cb = ActiveSkeletonsPolygonizerCallback()
        # Trigger exception in polygonize
        mock_poly_func.side_effect = RuntimeError("Mock polygonize error")

        batch = {"path": ["test.tif"]}
        outputs = (torch.zeros(1, 1, 10, 10), torch.zeros(1, 2, 10, 10))

        # Should catch error and try per-image
        cb.on_predict_batch_end(self.trainer, self.pl_module, outputs, batch, 0)

    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.instantiate_polygonizer"
    )
    def test_frame_field_polygonizer_error_handling(self, mock_inst):
        cb = FrameFieldPolygonizerCallback()
        mock_poly = MagicMock()
        mock_inst.return_value = mock_poly
        # Trigger exception in process
        mock_poly.process.side_effect = RuntimeError("Mock process error")

        batch = {"path": ["test.tif"]}
        outputs = (torch.zeros(1, 1, 10, 10), torch.zeros(1, 2, 10, 10))

        cb.on_predict_batch_end(self.trainer, self.pl_module, outputs, batch, 0)

    def test_ema_callback_checkpoint(self):
        cb = EMACallback()
        cb._shadow = {"weight": torch.ones(10)}
        checkpoint = {"state_dict": {"weight": torch.zeros(10)}}

        cb.on_save_checkpoint(self.trainer, self.pl_module, checkpoint)
        # Should have updated checkpoint state_dict with shadow
        torch.testing.assert_close(checkpoint["state_dict"]["weight"], torch.ones(10))

    def test_ema_callback_state_dict(self):
        cb = EMACallback(decay=0.99)
        cb._shadow = {"w": torch.ones(1)}
        sd = cb.state_dict()
        self.assertEqual(sd["decay"], 0.99)

        cb2 = EMACallback()
        cb2.load_state_dict(sd)
        self.assertEqual(cb2.decay, 0.99)
        self.assertIn("w", cb2._shadow)

    def test_mixstyle_callback_lifecycle(self):
        cb = MixStyleCallback()
        cb.on_train_batch_start(None, None, None, 0)
        self.assertTrue(cb._training)
        cb.on_train_batch_end(None, None, None, None, 0)
        self.assertFalse(cb._training)

        cb.on_validation_epoch_start(None, None)
        self.assertFalse(cb._training)

    def test_mixstyle_callback_fit_end(self):
        cb = MixStyleCallback()
        mock_hook = MagicMock()
        cb._hooks = [mock_hook]
        cb.on_fit_end(None, None)
        mock_hook.remove.assert_called_once()
        self.assertEqual(len(cb._hooks), 0)

    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.instantiate_polygonizer"
    )
    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.active_skeletons.polygonize"
    )
    def test_active_skeletons_polygonizer_writer_error(self, mock_poly_func, mock_inst):
        cb = ActiveSkeletonsPolygonizerCallback()
        mock_writer = MagicMock()
        mock_inst.return_value.data_writer = mock_writer
        mock_poly = MagicMock()
        mock_poly.geom_type = "Polygon"
        mock_poly_func.return_value = ([mock_poly], [0.9])

        # Trigger exception in writer
        mock_writer.write_data.side_effect = RuntimeError("Mock writer error")

        batch = {"path": ["test.tif"]}
        outputs = (torch.zeros(1, 1, 10, 10), torch.zeros(1, 2, 10, 10))
        cb.on_predict_batch_end(self.trainer, self.pl_module, outputs, batch, 0)
        mock_writer.write_data.assert_called()

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.barrier")
    @patch(
        "pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks.DataLoader"
    )
    def test_compute_weight_norm_distributed(self, mock_dl, mock_barrier, mock_init):
        cb = ComputeWeightNormLossesCallback()
        self.pl_module.trainer.world_size = 2
        self.trainer.global_rank = 0
        self.pl_module.loss_function.norm_updated = False
        self.pl_module.cfg.loss_params.compound_loss.normalization_params = {
            "max_samples": 2
        }

        # Mock dataloader
        mock_batch = MagicMock()
        mock_batch.values.return_value = (
            torch.zeros(2, 3, 10, 10),
            torch.zeros(2, 1, 10, 10),
        )
        mock_dl.return_value = [mock_batch]

        cb.on_train_start(self.trainer, self.pl_module)
        self.pl_module.loss_function.sync.assert_called_with(2)


if __name__ == "__main__":
    unittest.main()
