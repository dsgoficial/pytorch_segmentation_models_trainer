# -*- coding: utf-8 -*-
"""
Tests for new methods in pytorch_segmentation_models_trainer.model_loader.model.Model
"""

import pandas as pd
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import WeightedRandomSampler
from unittest.mock import MagicMock, patch, PropertyMock

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

    base_cfg = OmegaConf.create(
        {
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
        }
    )

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
        model = _make_model(
            {"hyperparameters": {"devices": 2, "accelerator": "cpu", "batch_size": 4}}
        )
        assert model._compute_device_count() == 2

    def test_list_of_devices(self):
        model = _make_model(
            {
                "hyperparameters": {
                    "devices": [0, 1],
                    "accelerator": "gpu",
                    "batch_size": 4,
                }
            }
        )
        assert model._compute_device_count() == 2

    def test_auto_without_cuda(self):
        model = _make_model(
            {
                "hyperparameters": {
                    "devices": "auto",
                    "accelerator": "gpu",
                    "batch_size": 4,
                }
            }
        )
        with patch("torch.cuda.is_available", return_value=False):
            assert model._compute_device_count() == 1

    def test_auto_with_cuda(self):
        model = _make_model(
            {
                "hyperparameters": {
                    "devices": "auto",
                    "accelerator": "gpu",
                    "batch_size": 4,
                }
            }
        )
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=4),
        ):
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
        model.cfg = OmegaConf.merge(
            model.cfg,
            OmegaConf.create(
                {
                    "train_dataset": {
                        "input_csv_path": str(tmp_path / "nonexistent.csv"),
                        "data_loader": {"batch_size": 4},
                    }
                }
            ),
        )
        result = model._compute_steps_from_config()
        assert result is None

    def test_valid_csv_returns_steps(self, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "train.csv"
        pd.DataFrame({"image": range(100), "mask": range(100)}).to_csv(
            csv_path, index=False
        )

        model = _make_model()
        model.cfg = OmegaConf.merge(
            model.cfg,
            OmegaConf.create(
                {
                    "train_dataset": {
                        "input_csv_path": str(csv_path),
                        "data_loader": {"batch_size": 4},
                    },
                }
            ),
        )
        result = model._compute_steps_from_config()
        # 100 samples / 4 batch / 1 device / 1 grad_accum = 25
        assert result == 25

    def test_zero_effective_batch_size_raises_value_error(self, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "train.csv"
        pd.DataFrame({"image": range(100), "mask": range(100)}).to_csv(
            csv_path, index=False
        )

        model = _make_model()
        model.cfg = OmegaConf.merge(
            model.cfg,
            OmegaConf.create(
                {
                    "train_dataset": {
                        "input_csv_path": str(csv_path),
                        "data_loader": {"batch_size": 0},
                    },
                }
            ),
        )
        # The ValueError is caught internally and logged, so the method
        # returns None instead of propagating.
        result = model._compute_steps_from_config()
        assert result is None

    def test_small_dataset_clamps_steps_per_epoch_to_one(self, tmp_path):
        import pandas as pd

        csv_path = tmp_path / "train.csv"
        pd.DataFrame({"image": range(2), "mask": range(2)}).to_csv(
            csv_path, index=False
        )

        model = _make_model()
        model.cfg = OmegaConf.merge(
            model.cfg,
            OmegaConf.create(
                {
                    "train_dataset": {
                        "input_csv_path": str(csv_path),
                        "data_loader": {"batch_size": 4},
                    },
                }
            ),
        )
        # 2 samples / effective_batch_size 4 -> 0, clamped to 1
        result = model._compute_steps_from_config()
        assert result == 1


# ---------------------------------------------------------------------------
# get_loss_function
# ---------------------------------------------------------------------------


class TestGetLossFunction:
    def test_compound_loss_path(self):
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        cfg = OmegaConf.create(
            {
                "model": {"_target_": "segmentation_models_pytorch.Unet"},
                "loss_params": {
                    "compound_loss": {
                        "losses": [
                            {
                                "loss": {"_target_": "torch.nn.BCEWithLogitsLoss"},
                                "weight": 1.0,
                            }
                        ]
                    }
                },
                "hyperparameters": {
                    "batch_size": 4,
                    "devices": 1,
                    "accelerator": "cpu",
                },
            }
        )

        mock_multiloss = MagicMock(spec=MultiLoss)
        with (
            patch.object(Model, "get_model", return_value=MagicMock()),
            patch(
                "pytorch_segmentation_models_trainer.custom_losses.loss_builder.build_compound_loss_from_config",
                return_value=mock_multiloss,
            ),
        ):
            model = Model(cfg, inference_mode=True)
            result = model.get_loss_function()

        assert result is mock_multiloss

    def test_simple_loss_path(self):
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        cfg = OmegaConf.create(
            {
                "model": {"_target_": "segmentation_models_pytorch.Unet"},
                "loss": {"_target_": "torch.nn.BCEWithLogitsLoss"},
                "hyperparameters": {
                    "batch_size": 4,
                    "devices": 1,
                    "accelerator": "cpu",
                },
            }
        )

        with patch.object(Model, "get_model", return_value=MagicMock()):
            model = Model(cfg, inference_mode=True)
            result = model.get_loss_function()

        assert isinstance(result, nn.BCEWithLogitsLoss)

    def test_no_config_raises_value_error(self):
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        cfg = OmegaConf.create(
            {
                "model": {"_target_": "segmentation_models_pytorch.Unet"},
                "hyperparameters": {
                    "batch_size": 4,
                    "devices": 1,
                    "accelerator": "cpu",
                },
            }
        )

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
        object.__setattr__(model, "loss_function", mock_loss)
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
        model.cfg = OmegaConf.merge(
            model.cfg,
            OmegaConf.create(
                {"loss_params": {"compound_loss": {"normalize_losses": True}}}
            ),
        )
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
        model.cfg = OmegaConf.merge(
            model.cfg,
            OmegaConf.create(
                {"loss_params": {"compound_loss": {"normalize_losses": True}}}
            ),
        )
        assert model.check_if_should_normalize() is True

    def test_normalize_false_from_config(self):
        model = _make_model()
        model.cfg = OmegaConf.merge(
            model.cfg,
            OmegaConf.create(
                {"loss_params": {"compound_loss": {"normalize_losses": False}}}
            ),
        )
        assert model.check_if_should_normalize() is False

    def test_no_loss_params_returns_false(self):
        model = _make_model()
        assert model.check_if_should_normalize() is False


# ---------------------------------------------------------------------------
# _make_weighted_sampler / train_dataloader with WeightedRandomSampler
# ---------------------------------------------------------------------------


def _dl_cfg(weighted_sampler=False, num_samples=None, replacement=True):
    d = {
        "shuffle": True,
        "num_workers": 0,
        "weighted_sampler": weighted_sampler,
    }
    if num_samples is not None:
        d["weighted_sampler_num_samples"] = num_samples
    if not replacement:
        d["weighted_sampler_replacement"] = False
    return OmegaConf.create(d)


def _attach_train_ds(model, weights, extra_cols=None):
    """Attach a mock train_ds with a sampler_weight column."""
    df = pd.DataFrame({"sampler_weight": weights})
    if extra_cols:
        for k, v in extra_cols.items():
            df[k] = v
    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=len(weights))
    mock_ds.df = df
    model.train_ds = mock_ds
    return mock_ds


class TestMakeWeightedSampler:
    def test_returns_none_when_disabled(self):
        model = _make_model()
        dl_cfg = _dl_cfg(weighted_sampler=False)
        assert model._make_weighted_sampler(dl_cfg) is None

    def test_returns_sampler_when_enabled(self):
        model = _make_model()
        _attach_train_ds(model, [0.1, 0.5, 0.9, 0.3])
        dl_cfg = _dl_cfg(weighted_sampler=True)
        sampler = model._make_weighted_sampler(dl_cfg)
        assert isinstance(sampler, WeightedRandomSampler)

    def test_default_num_samples_equals_dataset_len(self):
        model = _make_model()
        _attach_train_ds(model, [0.2, 0.8, 0.5])
        dl_cfg = _dl_cfg(weighted_sampler=True)
        sampler = model._make_weighted_sampler(dl_cfg)
        assert sampler.num_samples == 3

    def test_custom_num_samples(self):
        model = _make_model()
        _attach_train_ds(model, [0.2, 0.8, 0.5, 0.4])
        dl_cfg = _dl_cfg(weighted_sampler=True, num_samples=10)
        sampler = model._make_weighted_sampler(dl_cfg)
        assert sampler.num_samples == 10

    def test_replacement_false(self):
        model = _make_model()
        _attach_train_ds(model, [0.2, 0.8, 0.5, 0.4])
        dl_cfg = _dl_cfg(weighted_sampler=True, replacement=False)
        sampler = model._make_weighted_sampler(dl_cfg)
        assert sampler.replacement is False

    def test_raises_when_column_missing(self):
        model = _make_model()
        mock_ds = MagicMock()
        mock_ds.df = pd.DataFrame({"image_path": ["/a.tif"]})
        model.train_ds = mock_ds
        dl_cfg = _dl_cfg(weighted_sampler=True)
        with pytest.raises(ValueError, match="sampler_weight"):
            model._make_weighted_sampler(dl_cfg)

    def test_raises_when_ds_has_no_df(self):
        model = _make_model()
        mock_ds = MagicMock(spec=[])  # no .df attribute
        model.train_ds = mock_ds
        dl_cfg = _dl_cfg(weighted_sampler=True)
        with pytest.raises(ValueError, match="sampler_weight"):
            model._make_weighted_sampler(dl_cfg)


class TestTrainDataloaderWeightedSampler:
    def _full_dl_cfg(self, weighted=True):
        return OmegaConf.create(
            {
                "shuffle": True,
                "num_workers": 0,
                "weighted_sampler": weighted,
                "pin_memory": False,
                "drop_last": False,
                "persistent_workers": False,
            }
        )

    def _setup_model(self, weights, weighted=True):
        model = _make_model()
        _attach_train_ds(model, weights)
        model.cfg = OmegaConf.merge(
            model.cfg,
            OmegaConf.create(
                {"train_dataset": {"data_loader": self._full_dl_cfg(weighted)}}
            ),
        )
        return model

    def test_dataloader_uses_sampler_when_enabled(self):
        model = self._setup_model([0.1, 0.5, 0.9, 0.3], weighted=True)
        dl = model.train_dataloader()
        assert dl.sampler is not None
        assert isinstance(dl.sampler, WeightedRandomSampler)

    def test_dataloader_shuffle_false_when_sampler_active(self):
        model = self._setup_model([0.1, 0.5, 0.9], weighted=True)
        dl = model.train_dataloader()
        # When sampler is set, DataLoader uses SequentialSampler internally
        # (the WeightedRandomSampler *is* the sampler); shuffle must not be True
        from torch.utils.data import SequentialSampler

        assert not isinstance(dl.sampler, SequentialSampler)
        assert isinstance(dl.sampler, WeightedRandomSampler)

    def test_dataloader_no_sampler_when_disabled(self):
        model = self._setup_model([0.1, 0.5, 0.9], weighted=False)
        dl = model.train_dataloader()
        assert not isinstance(dl.sampler, WeightedRandomSampler)


# ---------------------------------------------------------------------------
# _zero_init_extra_input_channels
# ---------------------------------------------------------------------------


class TestZeroInitExtraInputChannels:
    """Tests for Model._zero_init_extra_input_channels static method."""

    def _make_conv(self, in_channels, out_channels=64, kernel_size=7):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
        nn.init.ones_(conv.weight)
        return conv

    def test_zeros_channels_beyond_n_base(self):
        conv = self._make_conv(21)
        model = nn.Sequential(conv)
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        Model._zero_init_extra_input_channels(model, n_base=3)
        assert conv.weight[:, 3:, :, :].abs().max().item() == 0.0

    def test_preserves_base_channels(self):
        conv = self._make_conv(21)
        model = nn.Sequential(conv)
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        Model._zero_init_extra_input_channels(model, n_base=3)
        assert conv.weight[:, :3, :, :].abs().max().item() > 0.0

    def test_only_first_conv_is_modified(self):
        conv1 = self._make_conv(21)
        conv2 = self._make_conv(21)
        model = nn.Sequential(conv1, conv2)
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        Model._zero_init_extra_input_channels(model, n_base=3)
        # conv1 extra channels zeroed
        assert conv1.weight[:, 3:, :, :].abs().max().item() == 0.0
        # conv2 untouched
        assert conv2.weight[:, 3:, :, :].abs().max().item() > 0.0

    def test_no_op_when_in_channels_equals_n_base(self):
        conv = self._make_conv(3)
        model = nn.Sequential(conv)
        original = conv.weight.clone()
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        Model._zero_init_extra_input_channels(model, n_base=3)
        assert torch.equal(conv.weight, original)

    def test_get_model_applies_zero_init_when_flag_set(self):
        """get_model() with zero_init_extra_input_channels=true zeroes extra channels."""
        from omegaconf import OmegaConf
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        cfg = OmegaConf.create(
            {
                "model": {
                    "_target_": "segmentation_models_pytorch.Unet",
                    "encoder_name": "resnet18",
                    "encoder_weights": None,
                    "in_channels": 6,
                    "classes": 6,
                },
                "zero_init_extra_input_channels": True,
            }
        )
        with patch.object(Model, "__init__", lambda self, *a, **kw: None):
            instance = Model.__new__(Model)
            instance.cfg = cfg
            smp_model = instance.get_model()

        first_conv = next(
            m
            for m in smp_model.modules()
            if isinstance(m, nn.Conv2d) and m.in_channels == 6
        )
        assert first_conv.weight[:, 3:, :, :].abs().max().item() == 0.0
        assert first_conv.weight[:, :3, :, :].abs().max().item() > 0.0

    def test_get_model_skips_zero_init_when_flag_absent(self):
        """get_model() without the flag leaves all weights as initialized."""
        from omegaconf import OmegaConf
        from pytorch_segmentation_models_trainer.model_loader.model import Model

        cfg = OmegaConf.create(
            {
                "model": {
                    "_target_": "segmentation_models_pytorch.Unet",
                    "encoder_name": "resnet18",
                    "encoder_weights": None,
                    "in_channels": 6,
                    "classes": 6,
                },
            }
        )
        with patch.object(Model, "__init__", lambda self, *a, **kw: None):
            instance = Model.__new__(Model)
            instance.cfg = cfg
            smp_model = instance.get_model()

        first_conv = next(
            m
            for m in smp_model.modules()
            if isinstance(m, nn.Conv2d) and m.in_channels == 6
        )
        # Without zero-init, SMP averaging fills all channels — none are all-zero
        assert first_conv.weight[:, 3:, :, :].abs().max().item() > 0.0
