# -*- coding: utf-8 -*-
"""
Tests for new methods in pytorch_segmentation_models_trainer.model_loader.model.Model
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, PropertyMock
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.custom_losses.base_loss import MultiLoss


# ---------------------------------------------------------------------------
# Helper: build a minimal Model using inference_mode=True to avoid heavy setup
# ---------------------------------------------------------------------------

def _make_model(cfg_overrides=None):
    """
    Instantiate Model with inference_mode=True and mocked get_model,
    then attach attributes needed for the methods under test.
    """
    from pytorch_segmentation_models_trainer.model_loader.model import Model

    base_cfg = OmegaConf.create({
        "model": {
            "_target_": "segmentation_models_pytorch.Unet",
            "encoder_name": "resnet18",
            "in_channels": 3,
            "classes": 1,
        },
        "hyperparameters": {
            "batch_size": 4,
            "devices": 1,
            "accelerator": "cpu",
        },
    })

    if cfg_overrides:
        cfg = OmegaConf.merge(base_cfg, OmegaConf.create(cfg_overrides))
    else:
        cfg = base_cfg

    mock_model = MagicMock()

    with patch.object(Model, "get_model", return_value=mock_model):
        model = Model(cfg, inference_mode=True)

    # Attach attributes normally set in full __init__
    model.loss_function = nn.BCEWithLogitsLoss()
    model.use_compound_loss = False
    model.should_normalize = False
    return model


# ---------------------------------------------------------------------------
# _compute_device_count
# ---------------------------------------------------------------------------

class TestComputeDeviceCount:
    def test_single_int_device(self):
        model = _make_model({"hyperparameters": {"devices": 2, "accelerator": "cpu", "batch_size": 4}})
        assert model._compute_device_count() == 2

    def test_list_of_devices(self):
        model = _make_model({"hyperparameters": {"devices": [0, 1], "accelerator": "gpu", "batch_size": 4}})
        assert model._compute_device_count() == 2

    def test_auto_without_cuda(self):
        model = _make_model({"hyperparameters": {"devices": "auto", "accelerator": "gpu", "batch_size": 4}})
        with patch("torch.cuda.is_available", return_value=False):
            assert model._compute_device_count() == 1

    def test_auto_with_cuda(self):
        model = _make_model({"hyperparameters": {"devices": "auto", "accelerator": "gpu", "batch_size": 4}})
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=4):
            assert model._compute_device_count() == 4


# ---------------------------------------------------------------------------
# _compute_steps_from_config
# ---------------------------------------------------------------------------

class TestComputeStepsFromConfig:
    def test_no_train_dataset_returns_none(self):
        model = _make_model()
        result = model._compute_steps_from_config()
        assert result is None

    def test_csv_not_found_returns_none(self, tmp_path):
        model = _make_model()
        model.cfg = OmegaConf.merge(model.cfg, OmegaConf.create({
            "train_dataset": {
                "input_csv_path": str(tmp_path / "nonexistent.csv"),
                "data_loader": {"batch_size": 4},
            }
        }))
        result = model._compute_steps_from_config()
        assert result is None

    def test_valid_csv_returns_steps(self, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "train.csv"
        pd.DataFrame({"image": range(100), "mask": range(100)}).to_csv(csv_path, index=False)

        model = _make_model()
        model.cfg = OmegaConf.merge(model.cfg, OmegaConf.create({
            "train_dataset": {
                "input_csv_path": str(csv_path),
                "data_loader": {"batch_size": 4},
            },
        }))
        result = model._compute_steps_from_config()
        # 100 samples / 4 batch / 1 device / 1 grad_accum = 25
        assert result == 25


# ---------------------------------------------------------------------------
# get_loss_function
# ---------------------------------------------------------------------------

class TestGetLossFunction:
    def test_compound_loss_path(self):
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        cfg = OmegaConf.create({
            "model": {"_target_": "segmentation_models_pytorch.Unet"},
            "loss_params": {
                "compound_loss": {
                    "losses": [
                        {"loss": {"_target_": "torch.nn.BCEWithLogitsLoss"}, "weight": 1.0}
                    ]
                }
            },
            "hyperparameters": {"batch_size": 4, "devices": 1, "accelerator": "cpu"},
        })

        mock_multiloss = MagicMock(spec=MultiLoss)
        with patch.object(Model, "get_model", return_value=MagicMock()), \
             patch(
                "pytorch_segmentation_models_trainer.custom_losses.loss_builder.build_compound_loss_from_config",
                return_value=mock_multiloss,
             ):
            model = Model(cfg, inference_mode=True)
            result = model.get_loss_function()

        assert result is mock_multiloss

    def test_simple_loss_path(self):
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        cfg = OmegaConf.create({
            "model": {"_target_": "segmentation_models_pytorch.Unet"},
            "loss": {"_target_": "torch.nn.BCEWithLogitsLoss"},
            "hyperparameters": {"batch_size": 4, "devices": 1, "accelerator": "cpu"},
        })

        with patch.object(Model, "get_model", return_value=MagicMock()):
            model = Model(cfg, inference_mode=True)
            result = model.get_loss_function()

        assert isinstance(result, nn.BCEWithLogitsLoss)

    def test_no_config_raises_value_error(self):
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        cfg = OmegaConf.create({
            "model": {"_target_": "segmentation_models_pytorch.Unet"},
            "hyperparameters": {"batch_size": 4, "devices": 1, "accelerator": "cpu"},
        })

        with patch.object(Model, "get_model", return_value=MagicMock()):
            model = Model(cfg, inference_mode=True)

        with pytest.raises(ValueError):
            model.get_loss_function()


# ---------------------------------------------------------------------------
# _compute_loss
# ---------------------------------------------------------------------------

class TestComputeLoss:
    def test_simple_loss_returns_triple(self):
        model = _make_model()
        mock_loss = MagicMock(return_value=torch.tensor(0.5))
        # Bypass nn.Module's __setattr__ to set a non-Module mock
        object.__setattr__(model, 'loss_function', mock_loss)
        model.use_compound_loss = False

        pred = torch.randn(2, 1, 4, 4)
        gt = torch.randint(0, 2, (2, 1, 4, 4)).float()
        loss, ind, extra = model._compute_loss(pred, gt)
        assert isinstance(loss, torch.Tensor)
        assert ind == {}
        assert extra == {}

    def test_compound_loss_returns_triple(self):
        model = _make_model()
        mock_loss_fn = MagicMock(spec=MultiLoss)
        mock_loss_fn.return_value = (
            torch.tensor(1.0),
            {"seg": torch.tensor(0.8)},
            {"seg": {"iou": torch.tensor(0.6)}},
        )
        model.loss_function = mock_loss_fn
        model.use_compound_loss = True
        model.cfg = OmegaConf.merge(model.cfg, OmegaConf.create({
            "loss_params": {
                "compound_loss": {"normalize_losses": True}
            }
        }))
        type(model).current_epoch = PropertyMock(return_value=0)

        pred = torch.randn(2, 1, 4, 4)
        gt = torch.randint(0, 2, (2, 1, 4, 4)).float()
        loss, ind, extra = model._compute_loss(pred, gt)
        assert isinstance(loss, torch.Tensor)
        assert "seg" in ind


# ---------------------------------------------------------------------------
# check_if_should_normalize
# ---------------------------------------------------------------------------

class TestCheckIfShouldNormalize:
    def test_normalize_true_from_config(self):
        model = _make_model()
        model.cfg = OmegaConf.merge(model.cfg, OmegaConf.create({
            "loss_params": {"compound_loss": {"normalize_losses": True}}
        }))
        assert model.check_if_should_normalize() is True

    def test_normalize_false_from_config(self):
        model = _make_model()
        model.cfg = OmegaConf.merge(model.cfg, OmegaConf.create({
            "loss_params": {"compound_loss": {"normalize_losses": False}}
        }))
        assert model.check_if_should_normalize() is False

    def test_no_loss_params_returns_false(self):
        model = _make_model()
        assert model.check_if_should_normalize() is False
