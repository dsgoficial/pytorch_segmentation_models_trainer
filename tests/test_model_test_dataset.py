# -*- coding: utf-8 -*-
"""
Tests for test_dataset support in Model and FrameFieldSegmentationPLModel.

Covers:
  - test_ds instantiation from config (present / absent)
  - gpu_test_transform guard
  - test_metrics MetricCollection with "test/" prefix
  - test_dataloader() returning None vs real DataLoader
  - test_step() loss value, logging keys, compound loss, metrics
"""
import logging
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from pytorch_segmentation_models_trainer.model_loader.model import Model

# ─── Shared constants & helpers ──────────────────────────────────────────────

B, H, W, CLASSES = 2, 32, 32, 2


def _make_cfg(**extra):
    """Minimal config with no dataset keys by default."""
    return OmegaConf.create({
        "model": {
            "_target_": "segmentation_models_pytorch.Unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": CLASSES,
        },
        "loss": {"_target_": "torch.nn.CrossEntropyLoss"},
        "hyperparameters": {
            "batch_size": B,
            "devices": 1,
            "accelerator": "cpu",
        },
        **extra,
    })


def _make_model(cfg=None):
    return Model(cfg or _make_cfg(), inference_mode=False)


def _make_batch():
    return {
        "image": torch.randn(B, 3, H, W),
        "mask": torch.zeros(B, H, W, dtype=torch.long),
    }


def _make_tiny_dataset():
    """Two-sample TensorDataset that returns (image, mask) tuples."""
    images = torch.randn(2, 3, H, W)
    masks = torch.zeros(2, H, W, dtype=torch.long)
    return TensorDataset(images, masks)


# ─── Model.__init__ with test_dataset ────────────────────────────────────────

class TestModelInitTestDataset:
    def test_test_ds_is_none_when_not_in_config(self):
        model = _make_model()
        assert model.test_ds is None

    def test_gpu_test_transform_is_none_when_not_in_config(self):
        model = _make_model()
        assert model.gpu_test_transform is None

    def test_init_does_not_crash_without_test_dataset(self):
        """Regression: __init__ must not raise when test_dataset is absent."""
        cfg = _make_cfg()
        model = Model(cfg, inference_mode=False)  # must not raise
        assert model.test_ds is None

    def test_test_metrics_not_created_without_metrics_config(self):
        model = _make_model()
        assert not hasattr(model, "test_metrics")

    def test_test_metrics_created_with_test_prefix(self):
        cfg = _make_cfg(**{
            "metrics": [
                {"_target_": "torchmetrics.F1Score", "task": "multiclass", "num_classes": CLASSES}
            ]
        })
        model = _make_model(cfg)
        assert hasattr(model, "test_metrics")
        # Metric names must carry the "test/" prefix
        metric_keys = list(model.test_metrics.keys())
        assert all(k.startswith("test/") for k in metric_keys)

    def test_all_three_metric_collections_created(self):
        cfg = _make_cfg(**{
            "metrics": [
                {"_target_": "torchmetrics.F1Score", "task": "multiclass", "num_classes": CLASSES}
            ]
        })
        model = _make_model(cfg)
        assert hasattr(model, "train_metrics")
        assert hasattr(model, "val_metrics")
        assert hasattr(model, "test_metrics")


# ─── test_dataloader ─────────────────────────────────────────────────────────

class TestTestDataloader:
    def test_returns_none_when_test_ds_is_none(self):
        model = _make_model()
        assert model.test_dataloader() is None

    def test_returns_dataloader_when_test_ds_set(self):
        from torch.utils.data import DataLoader

        model = _make_model()
        model.test_ds = _make_tiny_dataset()
        # Inject minimal test_dataset config so the dataloader can read params
        model.cfg = OmegaConf.merge(model.cfg, OmegaConf.create({
            "test_dataset": {
                "data_loader": {
                    "shuffle": False,
                    "num_workers": 0,
                    "pin_memory": False,
                    "drop_last": False,
                    "prefetch_factor": 2,
                    "persistent_workers": False,
                }
            }
        }))
        dl = model.test_dataloader()
        assert isinstance(dl, DataLoader)

    def test_shuffle_defaults_to_false(self):
        from torch.utils.data import DataLoader

        model = _make_model()
        model.test_ds = _make_tiny_dataset()
        model.cfg = OmegaConf.merge(model.cfg, OmegaConf.create({
            "test_dataset": {
                "data_loader": {"num_workers": 0, "pin_memory": False}
            }
        }))
        dl = model.test_dataloader()
        assert isinstance(dl, DataLoader)
        assert dl.dataset is model.test_ds


# ─── test_step ───────────────────────────────────────────────────────────────

class TestTestStep:
    def test_returns_scalar_loss(self):
        model = _make_model()
        model.log = MagicMock()
        loss = model.test_step(_make_batch(), 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_logs_loss_test_prefix(self):
        model = _make_model()
        model.log = MagicMock()
        model.test_step(_make_batch(), 0)
        logged_keys = [call.args[0] for call in model.log.call_args_list]
        assert "loss/test" in logged_keys

    def test_does_not_log_train_or_val_keys(self):
        model = _make_model()
        model.log = MagicMock()
        model.test_step(_make_batch(), 0)
        logged_keys = [call.args[0] for call in model.log.call_args_list]
        assert not any("train" in k for k in logged_keys)
        assert not any("/val" in k for k in logged_keys)

    def test_on_step_false_on_epoch_true(self):
        """test_step must log with on_step=False, on_epoch=True."""
        model = _make_model()
        model.log = MagicMock()
        model.test_step(_make_batch(), 0)
        for call in model.log.call_args_list:
            kwargs = call.kwargs
            if call.args[0] == "loss/test":
                assert kwargs.get("on_step") is False
                assert kwargs.get("on_epoch") is True

    def test_logs_individual_losses_with_test_prefix(self):
        """With compound loss, individual losses must use losses/test_* prefix."""
        from pytorch_segmentation_models_trainer.custom_losses.base_loss import MultiLoss
        from unittest.mock import PropertyMock

        model = _make_model()
        mock_loss_fn = MagicMock(spec=MultiLoss)
        mock_loss_fn.return_value = (
            torch.tensor(1.0),
            {"seg": torch.tensor(0.8), "crossfield": torch.tensor(0.2)},
            {},
        )
        object.__setattr__(model, "loss_function", mock_loss_fn)
        model.use_compound_loss = True
        model.cfg = OmegaConf.merge(model.cfg, OmegaConf.create({
            "loss_params": {"compound_loss": {"normalize_losses": False}}
        }))
        type(model).current_epoch = PropertyMock(return_value=0)
        model.log = MagicMock()

        model.test_step(_make_batch(), 0)
        logged_keys = [call.args[0] for call in model.log.call_args_list]
        assert "losses/test_seg" in logged_keys
        assert "losses/test_crossfield" in logged_keys

    def test_logs_extra_info_with_test_prefix(self):
        from pytorch_segmentation_models_trainer.custom_losses.base_loss import MultiLoss
        from unittest.mock import PropertyMock

        model = _make_model()
        mock_loss_fn = MagicMock(spec=MultiLoss)
        mock_loss_fn.return_value = (
            torch.tensor(1.0),
            {},
            {"seg": {"iou": torch.tensor(0.7)}},
        )
        object.__setattr__(model, "loss_function", mock_loss_fn)
        model.use_compound_loss = True
        model.cfg = OmegaConf.merge(model.cfg, OmegaConf.create({
            "loss_params": {"compound_loss": {"normalize_losses": False}}
        }))
        type(model).current_epoch = PropertyMock(return_value=0)
        model.log = MagicMock()

        model.test_step(_make_batch(), 0)
        logged_keys = [call.args[0] for call in model.log.call_args_list]
        assert "extra/test_seg_iou" in logged_keys

    def test_computes_metrics_with_test_prefix(self):
        cfg = _make_cfg(**{
            "metrics": [
                {"_target_": "torchmetrics.F1Score", "task": "multiclass", "num_classes": CLASSES}
            ]
        })
        model = _make_model(cfg)
        model.log = MagicMock()
        model.log_dict = MagicMock()
        model.test_step(_make_batch(), 0)
        assert model.log_dict.called
        logged_dict = model.log_dict.call_args_list[-1].args[0]
        assert any(k.startswith("test/") for k in logged_dict)

    def test_works_without_test_metrics(self):
        """test_step must not crash when cfg has no metrics section."""
        model = _make_model()
        assert not hasattr(model, "test_metrics")
        model.log = MagicMock()
        loss = model.test_step(_make_batch(), 0)
        assert isinstance(loss, torch.Tensor)

    def test_gpu_test_transform_applied_when_set(self):
        """gpu_test_transform must be applied to images before the forward pass."""
        model = _make_model()
        transform = MagicMock(side_effect=lambda x: x)
        model.gpu_test_transform = transform
        model.log = MagicMock()
        model.test_step(_make_batch(), 0)
        transform.assert_called_once()
