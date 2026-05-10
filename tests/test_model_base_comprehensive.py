# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pytorch_segmentation_models_trainer.model_loader.model import Model


class TestModelBaseComprehensive(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create(
            {
                "model": {
                    "_target_": "torch.nn.Conv2d",
                    "in_channels": 3,
                    "out_channels": 2,
                    "kernel_size": 1,
                },
                "loss": {"_target_": "torch.nn.CrossEntropyLoss"},
                "optimizer": {
                    "_target_": "torch.optim.Adam",
                    "lr": 1e-3,
                    "weight_decay": 0.05,
                },
                "hyperparameters": {
                    "batch_size": 2,
                    "lr": 1e-3,
                    "accelerator": "cpu",
                    "devices": 1,
                },
                "seed": 42,
            }
        )

    @patch("pytorch_segmentation_models_trainer.model_loader.model.instantiate")
    def test_model_init_inference_mode(self, mock_inst):
        model = Model(self.cfg, inference_mode=True)
        self.assertTrue(hasattr(model, "model"))

    def test_compute_device_count_various_configs(self):
        with patch.object(Model, "get_model"):
            with patch.object(Model, "get_loss_function"):
                self.cfg.hyperparameters.devices = "auto"
                model = Model(self.cfg)
                self.assertEqual(model._compute_device_count(), 1)

    def test_compute_steps_from_config_samples_per_epoch(self):
        self.cfg.train_dataset = {
            "samples_per_epoch": 20,
            "input_csv_path": "dummy.csv",
        }
        with patch.object(Model, "get_model"):
            with patch.object(Model, "get_loss_function"):
                with patch("os.path.exists", return_value=True):
                    model = Model(self.cfg)
                    self.assertEqual(model._compute_steps_from_config(), 10)

    def test_set_encoder_trainable(self):
        with patch.object(Model, "get_model"):
            with patch.object(Model, "get_loss_function"):
                model = Model(self.cfg)
                mock_inner = nn.Conv2d(3, 2, 1)
                model.model = MagicMock()
                model.model.encoder = mock_inner
                model.set_encoder_trainable(False)
                for p in mock_inner.parameters():
                    self.assertFalse(p.requires_grad)

    def test_get_optimizer_layer_decay(self):
        self.cfg.hyperparameters.layer_decay = 0.8
        with patch.object(Model, "get_loss_function"):
            model = Model(self.cfg)
            model.model = nn.Sequential(nn.Conv2d(3, 2, 1))

            def named_params():
                yield "encoder.stages.0.weight", nn.Parameter(torch.randn(1, 1))
                yield "decoder.weight", nn.Parameter(torch.randn(1, 1))

            model.named_parameters = named_params
            opt = model.get_optimizer()
            self.assertIsInstance(opt, torch.optim.Adam)

    def test_configure_optimizers_onecycle_auto(self):
        self.cfg.scheduler_list = [
            {
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.OneCycleLR",
                    "steps_per_epoch": "auto",
                    "max_lr": 0.01,
                    "epochs": 10,
                }
            }
        ]
        with patch.object(Model, "get_model"):
            with patch.object(Model, "get_loss_function"):
                model = Model(self.cfg)
                model.model = nn.Conv2d(3, 2, 1)
                with patch.object(
                    Model, "_compute_steps_from_config", return_value=100
                ):
                    with patch(
                        "pytorch_segmentation_models_trainer.model_loader.model.instantiate"
                    ) as mock_inst:
                        mock_inst.side_effect = [
                            torch.optim.Adam(model.parameters()),
                            MagicMock(),
                        ]
                        opts, scheds = model.configure_optimizers()
                        self.assertEqual(len(scheds), 1)

    def test_shared_step_with_metrics(self):
        with patch.object(Model, "get_model") as mock_get_model:
            with patch.object(Model, "get_loss_function") as mock_get_loss:
                mock_loss = MagicMock(spec=nn.Module)
                mock_loss.is_dual_head_loss = False
                mock_get_loss.return_value = mock_loss
                model = Model(self.cfg)
                model.log = MagicMock()
                model.log_dict = MagicMock()
                model.train_metrics = MagicMock()
                model.val_metrics = MagicMock()
                model.model = MagicMock(spec=nn.Module)
                if hasattr(model.model, "ohem_ratio"):
                    del model.model.ohem_ratio
                batch = {
                    "image": torch.randn(2, 3, 8, 8),
                    "mask": torch.randint(0, 2, (2, 8, 8)),
                }
                model.forward = MagicMock(return_value=torch.randn(2, 2, 8, 8))
                mock_loss.return_value = torch.tensor(0.5)
                loss = model.training_step(batch, 0)
                self.assertEqual(loss, 0.5)
                val_loss = model.validation_step(batch, 0)
                self.assertEqual(val_loss, 0.5)

    def test_generator(self):
        with patch.object(Model, "get_model"):
            with patch.object(Model, "get_loss_function"):
                model = Model(self.cfg)
                g = model._make_dataloader_generator()
                self.assertIsInstance(g, torch.Generator)

                model.cfg.seed = None
                self.assertIsNone(model._make_dataloader_generator())

    def test_unpack_batch(self):
        with patch.object(Model, "get_model"):
            with patch.object(Model, "get_loss_function"):
                model = Model(self.cfg)
                batch = {"image": torch.tensor(1), "mask": torch.tensor(2)}
                img, mask = model._unpack_batch(batch)
                self.assertEqual(img, 1)

    def test_compute_loss_multiloss(self):
        with patch.object(Model, "get_model"):
            with patch.object(Model, "get_loss_function"):
                model = Model(self.cfg)
                mock_multiloss = MagicMock()
                mock_multiloss.return_value = (
                    torch.tensor(0.5),
                    {"loss_0": torch.tensor(0.2)},
                    {"extra": "info"},
                )
                model.use_compound_loss = True
                model.loss_function = mock_multiloss
                loss, ind, extra = model._compute_loss(torch.randn(1), torch.randn(1))
                self.assertEqual(loss, 0.5)


if __name__ == "__main__":
    unittest.main()
