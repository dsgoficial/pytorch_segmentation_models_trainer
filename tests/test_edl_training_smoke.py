# -*- coding: utf-8 -*-
"""
Smoke tests for EDL integration with the Model (LightningModule).

Tests:
  - _shared_step with EvidentialWrapper produces a scalar loss
  - Uncertainty is logged in the output
  - Metrics use probs (not the raw dict)
  - Training from scratch (freeze_encoder=False) and fine-tuning (True) both work
  - EvidentialWarmupCallback freeze/unfreeze lifecycle
"""

import pytest
import torch
from omegaconf import OmegaConf
from unittest.mock import MagicMock

from pytorch_segmentation_models_trainer.custom_models.edl_wrapper import (
    EvidentialWrapper,
)
from pytorch_segmentation_models_trainer.custom_callbacks.edl_callbacks import (
    EvidentialWarmupCallback,
)
from pytorch_segmentation_models_trainer.model_loader.model import Model

# resnet18 UNet needs input >= 32x32; use 64x64 to be safe
B, K, H, W = 2, 4, 64, 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_edl_cfg(freeze_encoder=False):
    # Use resnet18 from SMP — small and publicly importable by Hydra
    return OmegaConf.create(
        {
            "model": {
                "_target_": "pytorch_segmentation_models_trainer.custom_models.edl_wrapper.EvidentialWrapper",
                "freeze_encoder": freeze_encoder,
                "model": {
                    "_target_": "segmentation_models_pytorch.Unet",
                    "encoder_name": "resnet18",
                    "encoder_weights": None,
                    "in_channels": 3,
                    "classes": K,
                },
            },
            "loss_params": {
                "compound_loss": {
                    "epoch_thresholds": [0, 5, 10],
                    "losses": [
                        {
                            "loss": {
                                "_target_": "pytorch_segmentation_models_trainer.custom_losses.edl_loss.EvidentialMSELoss",
                                "name": "edl_mse",
                                "num_classes": K,
                            },
                            "weight": 1.0,
                        },
                        {
                            "loss": {
                                "_target_": "pytorch_segmentation_models_trainer.custom_losses.edl_loss.EvidentialKLLoss",
                                "name": "edl_kl",
                                "num_classes": K,
                            },
                            "weight": [0.0, 0.0, 1.0],
                        },
                    ],
                }
            },
            "hyperparameters": {
                "batch_size": B,
                "devices": 1,
                "accelerator": "cpu",
            },
        }
    )


def _make_batch():
    return {
        "image": torch.randn(B, 3, H, W),
        "mask": torch.zeros(B, H, W, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# _shared_step tests
# ---------------------------------------------------------------------------


class TestEDLSharedStep:
    @pytest.fixture
    def model(self):
        cfg = _make_edl_cfg(freeze_encoder=False)
        return Model(cfg, inference_mode=False)

    def test_training_step_returns_scalar_loss(self, model):
        batch = _make_batch()
        loss = model._shared_step(batch, "train")
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_validation_step_returns_scalar_loss(self, model):
        batch = _make_batch()
        loss = model._shared_step(batch, "val")
        assert loss.ndim == 0

    def test_uncertainty_is_logged(self, model):
        batch = _make_batch()
        logged = {}

        def _fake_log(name, value, **kwargs):
            logged[name] = value

        model.log = _fake_log
        model._shared_step(batch, "train")
        assert "edl/train_uncertainty" in logged

    def test_loss_backward_does_not_error(self, model):
        batch = _make_batch()
        loss = model._shared_step(batch, "train")
        loss.backward()  # should not raise

    def test_freeze_encoder_variant(self):
        cfg = _make_edl_cfg(freeze_encoder=True)
        model = Model(cfg, inference_mode=False)
        # Encoder should be frozen
        encoder = model.model._find_encoder()
        assert encoder is not None
        for p in encoder.parameters():
            assert not p.requires_grad
        # Decoder should still be trainable
        for p in model.model.model.decoder.parameters():
            assert p.requires_grad


# ---------------------------------------------------------------------------
# EvidentialWarmupCallback tests
# ---------------------------------------------------------------------------


class TestEvidentialWarmupCallback:
    def _make_pl_module(self, freeze_encoder=True):
        cfg = _make_edl_cfg(freeze_encoder=freeze_encoder)
        return Model(cfg, inference_mode=False)

    def _make_trainer(self, current_epoch=0):
        trainer = MagicMock()
        trainer.current_epoch = current_epoch
        return trainer

    def test_on_fit_start_freezes_encoder(self):
        cb = EvidentialWarmupCallback(warmup_epochs=5, freeze_encoder=True)
        pl_module = self._make_pl_module(freeze_encoder=False)  # start unfrozen
        trainer = self._make_trainer(current_epoch=0)
        cb.on_fit_start(trainer, pl_module)
        encoder = pl_module.model._find_encoder()
        for p in encoder.parameters():
            assert not p.requires_grad

    def test_on_fit_start_no_freeze_when_from_scratch(self):
        cb = EvidentialWarmupCallback(warmup_epochs=5, freeze_encoder=False)
        pl_module = self._make_pl_module(freeze_encoder=False)
        trainer = self._make_trainer(current_epoch=0)
        cb.on_fit_start(trainer, pl_module)
        # Encoder should remain trainable
        encoder = pl_module.model._find_encoder()
        for p in encoder.parameters():
            assert p.requires_grad

    def test_partial_unfreeze_at_warmup_epoch(self):
        cb = EvidentialWarmupCallback(
            warmup_epochs=5, freeze_encoder=True, partial_unfreeze_epoch=10
        )
        pl_module = self._make_pl_module(freeze_encoder=True)
        trainer_start = self._make_trainer(current_epoch=0)
        cb.on_fit_start(trainer_start, pl_module)

        # Simulate reaching warmup_epochs
        trainer_warmup = self._make_trainer(current_epoch=5)
        cb.on_train_epoch_start(trainer_warmup, pl_module)
        # At least some encoder params should now be trainable
        encoder = pl_module.model._find_encoder()
        trainable_params = [p for p in encoder.parameters() if p.requires_grad]
        assert len(trainable_params) > 0

    def test_full_unfreeze_at_partial_unfreeze_epoch(self):
        cb = EvidentialWarmupCallback(
            warmup_epochs=5, freeze_encoder=True, partial_unfreeze_epoch=10
        )
        pl_module = self._make_pl_module(freeze_encoder=True)
        trainer_start = self._make_trainer(current_epoch=0)
        cb.on_fit_start(trainer_start, pl_module)

        cb._phase = "partial"
        trainer_full = self._make_trainer(current_epoch=10)
        cb.on_train_epoch_start(trainer_full, pl_module)
        encoder = pl_module.model._find_encoder()
        for p in encoder.parameters():
            assert p.requires_grad
