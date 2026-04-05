# -*- coding: utf-8 -*-
"""
Tests for the hardened training loop (Phase 6).

Validates batch unpacking, configurable keys, extra key tolerance,
and end-to-end training/validation steps with a lightweight SMP model.
"""
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.model_loader.model import Model


# ─── Helpers ─────────────────────────────────────────────────────────────────

B, H, W, CLASSES = 2, 64, 64, 2


def _make_cfg(image_key="image", mask_key="mask", **extra):
    cfg = OmegaConf.create({
        "model": {
            "_target_": "segmentation_models_pytorch.Unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": CLASSES,
        },
        "loss": {
            "_target_": "torch.nn.CrossEntropyLoss",
        },
        "image_key": image_key,
        "mask_key": mask_key,
        "hyperparameters": {
            "batch_size": B,
            "devices": 1,
            "accelerator": "cpu",
        },
        **extra,
    })
    return cfg


def _make_model(cfg=None):
    cfg = cfg or _make_cfg()
    return Model(cfg, inference_mode=False)


def _make_batch(image_key="image", mask_key="mask", extra_keys=None):
    batch = {
        image_key: torch.randn(B, 3, H, W),
        mask_key: torch.zeros(B, H, W, dtype=torch.long),
    }
    if extra_keys:
        for k in extra_keys:
            batch[k] = torch.zeros(B)
    return batch


# ─── Tests: _unpack_batch ─────────────────────────────────────────────────────

class TestUnpackBatch:
    def test_default_keys(self):
        model = _make_model()
        batch = _make_batch()
        images, masks = model._unpack_batch(batch)
        assert images.shape == (B, 3, H, W)
        assert masks.shape == (B, H, W)

    def test_custom_image_key(self):
        cfg = _make_cfg(image_key="pixel_values", mask_key="label")
        model = _make_model(cfg)
        batch = _make_batch(image_key="pixel_values", mask_key="label")
        images, masks = model._unpack_batch(batch)
        assert images.shape == (B, 3, H, W)

    def test_extra_keys_ignored(self):
        model = _make_model()
        batch = _make_batch(extra_keys=["filename", "metadata"])
        images, masks = model._unpack_batch(batch)
        assert images.shape == (B, 3, H, W)
        assert masks.shape == (B, H, W)

    def test_tuple_batch_fallback(self):
        model = _make_model()
        images_t = torch.randn(B, 3, H, W)
        masks_t = torch.zeros(B, H, W, dtype=torch.long)
        images, masks = model._unpack_batch((images_t, masks_t))
        assert torch.equal(images, images_t)

    def test_missing_key_raises(self):
        model = _make_model()
        batch = {"wrong_key": torch.randn(B, 3, H, W)}
        with pytest.raises(KeyError):
            model._unpack_batch(batch)


# ─── Tests: _prepare_preds_for_metrics ────────────────────────────────────────

class TestPreparePreds:
    def test_squeezes_binary(self):
        model = _make_model()
        preds = torch.randn(B, 1, H, W)
        result = model._prepare_preds_for_metrics(preds)
        assert result.shape == (B, H, W)

    def test_multiclass_unchanged(self):
        model = _make_model()
        preds = torch.randn(B, CLASSES, H, W)
        result = model._prepare_preds_for_metrics(preds)
        assert result.shape == (B, CLASSES, H, W)

    def test_non_tensor_returns_none(self, caplog):
        model = _make_model()
        with caplog.at_level(logging.WARNING):
            result = model._prepare_preds_for_metrics({"logits": torch.randn(B, CLASSES, H, W)})
        assert result is None
        assert "not a torch.tensor" in caplog.text.lower()


# ─── Tests: training_step / validation_step end-to-end ───────────────────────

class TestTrainingStepEndToEnd:
    def test_training_step_returns_scalar_loss(self):
        model = _make_model()
        batch = _make_batch()
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_validation_step_returns_scalar_loss(self):
        model = _make_model()
        batch = _make_batch()
        # Mock the log method to avoid PL trainer requirement
        model.log = MagicMock()
        loss = model.validation_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_training_step_with_extra_batch_keys(self):
        model = _make_model()
        batch = _make_batch(extra_keys=["filename"])
        model.log = MagicMock()
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)

    def test_training_step_custom_keys(self):
        cfg = _make_cfg(image_key="pixel_values", mask_key="label")
        model = _make_model(cfg)
        batch = _make_batch(image_key="pixel_values", mask_key="label")
        model.log = MagicMock()
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)


# ─── Tests: set_encoder_trainable ─────────────────────────────────────────────

class TestSetEncoderTrainable:
    def test_smp_model_encoder_can_be_frozen(self):
        model = _make_model()
        model.set_encoder_trainable(trainable=False)
        # At least some encoder params should be frozen
        encoder_params = [
            p for n, p in model.model.named_parameters() if "encoder" in n
        ]
        assert any(not p.requires_grad for p in encoder_params)

    def test_smp_model_encoder_can_be_unfrozen(self):
        model = _make_model()
        # Freeze first
        for p in model.model.parameters():
            p.requires_grad = False
        # Then unfreeze encoder
        model.set_encoder_trainable(trainable=True)
        encoder_params = [
            p for n, p in model.model.named_parameters() if "encoder" in n
        ]
        assert any(p.requires_grad for p in encoder_params)
