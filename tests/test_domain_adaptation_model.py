# -*- coding: utf-8 -*-
"""
Tests for DomainAdaptationModel.

Covers:
- Pretrained checkpoint loading (PL and pytorch formats, strict/non-strict)
- train_dataloader returns CombinedLoader with source+target keys
- configure_optimizers includes extra parameter groups from the method
- training_step runs without error and returns a scalar loss
- validation_step runs without error
- _get_val_domain_name mapping
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import Dataset, TensorDataset

from pytorch_segmentation_models_trainer.domain_adaptation.base_method import (
    BaseDomainAdaptationMethod,
    DomainAdaptationLossOutput,
)
from pytorch_segmentation_models_trainer.model_loader.domain_adaptation_model import (
    DomainAdaptationModel,
)

# ---------------------------------------------------------------------------
# Minimal helpers
# ---------------------------------------------------------------------------


class _DummySegModel(nn.Module):
    """Tiny 2-class segmentation model for CPU tests."""

    def __init__(self, in_channels=3, num_classes=2, height=8, width=8):
        super().__init__()
        self.encoder = nn.Conv2d(in_channels, 8, kernel_size=1)
        self.decoder = nn.Conv2d(8, num_classes, kernel_size=1)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class _DummyMethod(BaseDomainAdaptationMethod):
    requires_features = False

    def compute_da_loss(
        self,
        source_batch,
        target_batch,
        source_output,
        target_output,
        source_features,
        target_features,
        **kwargs,
    ):
        # Simple entropy loss on target
        probs = torch.softmax(target_output, dim=1)
        entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=1).mean()
        return DomainAdaptationLossOutput(
            loss=entropy, log_dict={"entropy": entropy.item()}
        )


class _DummyMethodWithExtras(BaseDomainAdaptationMethod):
    requires_features = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = nn.Linear(4, 1)

    def compute_da_loss(
        self,
        source_batch,
        target_batch,
        source_output,
        target_output,
        source_features,
        target_features,
        **kwargs,
    ):
        loss = torch.tensor(0.0, requires_grad=True)
        return DomainAdaptationLossOutput(loss=loss, log_dict={})

    def get_extra_parameter_groups(self):
        return [{"params": list(self.discriminator.parameters()), "lr": 1e-4}]


class _DummyFeatureMethod(_DummyMethod):
    """Dummy DA method that requests feature maps from the segmentation model."""

    requires_features = True


class _DummyDataset(Dataset):
    """Returns image + mask dicts of shape (3,8,8) and (8,8).

    Accepts ``data_loader`` and other extra kwargs so that Hydra can instantiate
    this class from a config that includes DataLoader settings.
    """

    def __init__(self, size=16, num_classes=2, data_loader=None, **kwargs):
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 8, 8),
            "mask": torch.randint(0, self.num_classes, (8, 8)),
        }


class _DummyCollateDataset(_DummyDataset):
    """Dataset with a custom collate function for dataloader branch coverage."""

    def collate_fn(self, batch):
        return {"count": len(batch), "items": batch}


class _LambdaSchedule:
    """Minimal lambda schedule used to exercise epoch-based DA weights."""

    def get_lambda(self, epoch, max_epochs):
        return float(epoch + max_epochs)


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def _make_cfg(
    with_source_val=False,
    with_target_val=False,
    with_pretrained_ckpt=None,
    method_target="_ConcreteMethod",
    feature_layers=None,
):
    """Build a minimal OmegaConf config that DomainAdaptationModel can consume."""
    base = {
        "model": {
            "_target_": ("tests.test_domain_adaptation_model._DummySegModel"),
        },
        "optimizer": {
            "_target_": "torch.optim.SGD",
            "lr": 1e-3,
            "momentum": 0.9,
        },
        "hyperparameters": {
            "batch_size": 4,
            "epochs": 2,
            "max_lr": 1e-3,
        },
        "loss": {
            "_target_": "torch.nn.CrossEntropyLoss",
        },
        "domain_adaptation": {
            "method": {
                "_target_": method_target,
                "lambda_da": 1.0,
            },
            "source_dataset": {
                "_target_": "tests.test_domain_adaptation_model._DummyDataset",
                "data_loader": {
                    "batch_size": 4,
                    "num_workers": 0,
                    "shuffle": False,
                    "pin_memory": False,
                    "drop_last": False,
                    "prefetch_factor": 2,
                    "persistent_workers": False,
                },
            },
            "target_dataset": {
                "_target_": "tests.test_domain_adaptation_model._DummyDataset",
                "data_loader": {
                    "batch_size": 4,
                    "num_workers": 0,
                    "shuffle": False,
                    "pin_memory": False,
                    "drop_last": False,
                    "prefetch_factor": 2,
                    "persistent_workers": False,
                },
            },
            "source_val_dataset": None,
            "target_val_dataset": None,
            "feature_layers": feature_layers or [],
            "pretrained_checkpoint": None,
        },
    }

    if with_source_val:
        base["domain_adaptation"]["source_val_dataset"] = {
            "_target_": "tests.test_domain_adaptation_model._DummyDataset",
            "data_loader": {
                "batch_size": 4,
                "num_workers": 0,
                "shuffle": False,
                "pin_memory": False,
                "drop_last": False,
                "prefetch_factor": 2,
                "persistent_workers": False,
            },
        }

    if with_target_val:
        base["domain_adaptation"]["target_val_dataset"] = {
            "_target_": "tests.test_domain_adaptation_model._DummyDataset",
            "data_loader": {
                "batch_size": 4,
                "num_workers": 0,
                "shuffle": False,
                "pin_memory": False,
                "drop_last": False,
                "prefetch_factor": 2,
                "persistent_workers": False,
            },
        }

    if with_pretrained_ckpt:
        base["domain_adaptation"]["pretrained_checkpoint"] = with_pretrained_ckpt

    return OmegaConf.create(base)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDomainAdaptationModelCheckpointLoading:
    def test_load_pytorch_lightning_format(self, tmp_path):
        """Weights stored under 'state_dict' with 'model.' prefix are loaded."""
        model = _DummySegModel()
        # Simulate PL checkpoint structure
        ckpt = {"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}
        ckpt_path = str(tmp_path / "model.ckpt")
        torch.save(ckpt, ckpt_path)

        target_model = _DummySegModel()
        ckpt_cfg = OmegaConf.create(
            {
                "path": ckpt_path,
                "source_format": "pytorch_lightning",
                "strict_loading": True,
            }
        )

        # Call the loading method directly
        cfg = _make_cfg()
        da_model = DomainAdaptationModel.__new__(DomainAdaptationModel)
        da_model.cfg = cfg
        da_model._load_pretrained_weights(target_model, ckpt_cfg)

        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), target_model.state_dict().items()
        ):
            assert torch.allclose(v1, v2), f"Mismatch on key {k1}"

    def test_load_pytorch_format(self, tmp_path):
        """Plain PyTorch state dict (no prefix) is loaded correctly."""
        model = _DummySegModel()
        ckpt_path = str(tmp_path / "model.pt")
        torch.save(model.state_dict(), ckpt_path)

        target_model = _DummySegModel()
        ckpt_cfg = OmegaConf.create(
            {
                "path": ckpt_path,
                "source_format": "pytorch",
                "strict_loading": True,
            }
        )

        cfg = _make_cfg()
        da_model = DomainAdaptationModel.__new__(DomainAdaptationModel)
        da_model.cfg = cfg
        da_model._load_pretrained_weights(target_model, ckpt_cfg)

        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), target_model.state_dict().items()
        ):
            assert torch.allclose(v1, v2)

    def test_strict_loading_false_allows_partial(self, tmp_path):
        """strict_loading=False tolerates missing keys."""
        model = _DummySegModel()
        # Save only the encoder weights
        partial_state = {k: v for k, v in model.state_dict().items() if "encoder" in k}
        ckpt_path = str(tmp_path / "partial.pt")
        torch.save(partial_state, ckpt_path)

        target_model = _DummySegModel()
        ckpt_cfg = OmegaConf.create(
            {
                "path": ckpt_path,
                "source_format": "pytorch",
                "strict_loading": False,
            }
        )

        cfg = _make_cfg()
        da_model = DomainAdaptationModel.__new__(DomainAdaptationModel)
        da_model.cfg = cfg
        # Should not raise
        da_model._load_pretrained_weights(target_model, ckpt_cfg)

    def test_non_strict_loading_logs_unexpected_keys(self, tmp_path):
        """Unexpected checkpoint keys are surfaced when strict loading is disabled."""
        ckpt_path = str(tmp_path / "unexpected.pt")
        torch.save({"unexpected.weight": torch.ones(1)}, ckpt_path)
        ckpt_cfg = OmegaConf.create(
            {
                "path": ckpt_path,
                "source_format": "pytorch",
                "strict_loading": False,
            }
        )

        da_model = DomainAdaptationModel.__new__(DomainAdaptationModel)
        da_model.cfg = _make_cfg()
        with patch(
            "pytorch_segmentation_models_trainer.model_loader.domain_adaptation_model.logger"
        ) as logger_mock:
            da_model._load_pretrained_weights(_DummySegModel(), ckpt_cfg)

        assert logger_mock.warning.call_count == 2

    def test_get_model_returns_parent_model_without_domain_adaptation_cfg(self):
        """get_model delegates to the parent and skips warm-start when DA config is absent."""
        da_model = DomainAdaptationModel.__new__(DomainAdaptationModel)
        da_model.cfg = OmegaConf.create({"model": {"_target_": "unused"}})
        base_model = _DummySegModel()

        with patch(
            "pytorch_segmentation_models_trainer.model_loader.model.Model.get_model",
            return_value=base_model,
        ):
            assert DomainAdaptationModel.get_model(da_model) is base_model

    def test_get_model_loads_pretrained_checkpoint_when_configured(self, tmp_path):
        """Configured warm-start checkpoint is loaded during model construction."""
        ckpt_path = str(tmp_path / "model.pt")
        torch.save(_DummySegModel().state_dict(), ckpt_path)
        cfg = _make_cfg(
            with_pretrained_ckpt={
                "path": ckpt_path,
                "source_format": "pytorch",
                "strict_loading": True,
            }
        )
        da_model = DomainAdaptationModel.__new__(DomainAdaptationModel)
        da_model.cfg = cfg
        base_model = _DummySegModel()

        with (
            patch(
                "pytorch_segmentation_models_trainer.model_loader.model.Model.get_model",
                return_value=base_model,
            ),
            patch.object(
                DomainAdaptationModel, "_load_pretrained_weights"
            ) as load_mock,
        ):
            assert DomainAdaptationModel.get_model(da_model) is base_model

        load_mock.assert_called_once()


class TestDomainAdaptationModelDataLoaders:
    def _build_model(self, **cfg_kwargs):
        cfg = _make_cfg(
            method_target="tests.test_domain_adaptation_model._DummyMethod",
            **cfg_kwargs,
        )
        return DomainAdaptationModel(cfg)

    def test_train_dataloader_returns_combined_loader(self):
        from pytorch_lightning.utilities.combined_loader import CombinedLoader

        model = self._build_model()
        loader = model.train_dataloader()
        assert isinstance(loader, CombinedLoader)

    def test_train_dataloader_has_source_and_target_keys(self):
        model = self._build_model()
        loader = model.train_dataloader()
        assert "source" in loader.iterables
        assert "target" in loader.iterables

    def test_val_dataloader_source_only(self):
        from torch.utils.data import DataLoader

        model = self._build_model(with_source_val=True)
        loader = model.val_dataloader()
        assert isinstance(loader, DataLoader)

    def test_val_dataloader_both_domains_returns_combined(self):
        from pytorch_lightning.utilities.combined_loader import CombinedLoader

        model = self._build_model(with_source_val=True, with_target_val=True)
        loader = model.val_dataloader()
        assert isinstance(loader, CombinedLoader)

    def test_val_dataloader_without_validation_sets_returns_empty_loader(self):
        from torch.utils.data import DataLoader

        model = self._build_model()
        loader = model.val_dataloader()
        assert isinstance(loader, DataLoader)
        assert list(loader) == []

    def test_make_dataloader_uses_safe_defaults_and_prefetch_for_workers(self):
        model = self._build_model()
        loader = model._make_dataloader(_DummyDataset(size=2), None)

        assert loader.batch_size == 8
        assert loader.num_workers == 4
        assert loader.prefetch_factor == 2
        assert loader.drop_last is False

    def test_make_dataloader_uses_custom_collate_function(self):
        model = self._build_model()
        loader = model._make_dataloader(
            _DummyCollateDataset(size=2),
            OmegaConf.create({"batch_size": 2, "num_workers": 0}),
        )

        batch = next(iter(loader))
        assert batch["count"] == 2


class TestDomainAdaptationModelOptimizers:
    def test_configure_optimizers_returns_one_optimizer_by_default(self):
        cfg = _make_cfg(method_target="tests.test_domain_adaptation_model._DummyMethod")
        model = DomainAdaptationModel(cfg)
        optimizers, schedulers = model.configure_optimizers()
        assert len(optimizers) == 1

    def test_configure_optimizers_includes_extra_param_groups(self):
        cfg = _make_cfg(
            method_target="tests.test_domain_adaptation_model._DummyMethodWithExtras"
        )
        model = DomainAdaptationModel(cfg)
        optimizers, _ = model.configure_optimizers()
        assert len(optimizers) == 1
        optimizer = optimizers[0]
        # Should have at least 2 param groups: main + discriminator
        assert len(optimizer.param_groups) >= 2
        # The extra group should have lr=1e-4
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        assert 1e-4 in lrs

    def test_build_scheduler_list_instantiates_configured_schedulers(self):
        cfg = _make_cfg(method_target="tests.test_domain_adaptation_model._DummyMethod")
        cfg.scheduler_list = [
            {
                "interval": "epoch",
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.StepLR",
                    "step_size": 1,
                    "gamma": 0.5,
                },
            }
        ]
        model = DomainAdaptationModel(cfg)
        optimizers, schedulers = model.configure_optimizers()

        assert len(optimizers) == 1
        assert len(schedulers) == 1
        assert schedulers[0]["interval"] == "epoch"


class TestDomainAdaptationModelTrainingStep:
    def _make_batch(self, batch_size=2, num_classes=2, h=8, w=8):
        return {
            "source": {
                "image": torch.randn(batch_size, 3, h, w),
                "mask": torch.randint(0, num_classes, (batch_size, h, w)),
            },
            "target": {
                "image": torch.randn(batch_size, 3, h, w),
            },
        }

    def test_training_step_returns_scalar(self):
        cfg = _make_cfg(method_target="tests.test_domain_adaptation_model._DummyMethod")
        model = DomainAdaptationModel(cfg)
        model.trainer = MagicMock()
        model.trainer.max_epochs = 10

        batch = self._make_batch()
        loss = model.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar

    def test_training_step_loss_is_finite(self):
        cfg = _make_cfg(method_target="tests.test_domain_adaptation_model._DummyMethod")
        model = DomainAdaptationModel(cfg)
        model.trainer = MagicMock()
        model.trainer.max_epochs = 10

        batch = self._make_batch()
        loss = model.training_step(batch, 0)
        assert torch.isfinite(loss)

    def test_training_step_logs_individual_losses_and_uses_current_lambda(self):
        cfg = _make_cfg(method_target="tests.test_domain_adaptation_model._DummyMethod")
        model = DomainAdaptationModel(cfg)
        model.method._current_lambda = 0.25
        model.trainer = MagicMock(max_epochs=10)
        model.log = MagicMock()
        seg_loss = torch.tensor(2.0, requires_grad=True)

        with patch.object(
            model,
            "_compute_loss",
            return_value=(seg_loss, {"ce": torch.tensor(2.0)}, {}),
        ):
            loss = model.training_step(self._make_batch(), 0)

        assert loss.shape == torch.Size([])
        logged_names = [call.args[0] for call in model.log.call_args_list]
        assert "losses/train_ce" in logged_names

    def test_get_lambda_da_uses_schedule_when_no_current_lambda(self):
        cfg = _make_cfg(method_target="tests.test_domain_adaptation_model._DummyMethod")
        model = DomainAdaptationModel(cfg)
        model.method.lambda_schedule = _LambdaSchedule()
        model.trainer = MagicMock(max_epochs=7)

        with patch.object(
            DomainAdaptationModel, "current_epoch", new_callable=PropertyMock
        ) as current_epoch:
            current_epoch.return_value = 3
            assert model._get_lambda_da() == 10.0

    def test_get_lambda_da_uses_schedule_default_max_epochs_without_trainer(self):
        cfg = _make_cfg(method_target="tests.test_domain_adaptation_model._DummyMethod")
        model = DomainAdaptationModel(cfg)
        model.method.lambda_schedule = _LambdaSchedule()

        with patch.object(
            DomainAdaptationModel, "trainer", new_callable=PropertyMock
        ) as trainer_mock:
            trainer_mock.return_value = None
            assert model._get_lambda_da() == 1.0


class TestDomainAdaptationModelFeatureHooks:
    def test_init_with_metrics_and_empty_feature_layers_warns(self):
        cfg = _make_cfg(
            method_target="tests.test_domain_adaptation_model._DummyFeatureMethod",
            feature_layers=[],
        )
        cfg.metrics = [{"_target_": "torchmetrics.MeanMetric"}]

        with patch(
            "pytorch_segmentation_models_trainer.model_loader.domain_adaptation_model.logger"
        ) as logger_mock:
            model = DomainAdaptationModel(cfg)

        assert hasattr(model, "train_metrics")
        assert model._feature_hook is None
        logger_mock.warning.assert_called()

    def test_forward_with_features_clears_and_returns_hook_features(self):
        cfg = _make_cfg(
            method_target="tests.test_domain_adaptation_model._DummyFeatureMethod",
            feature_layers=["encoder"],
        )
        model = DomainAdaptationModel(cfg)
        images = torch.randn(2, 3, 8, 8)

        output, features = model._forward_with_features(images)

        assert output.shape == (2, 2, 8, 8)
        assert "encoder" in features

    def test_on_fit_end_removes_feature_hook(self):
        cfg = _make_cfg(
            method_target="tests.test_domain_adaptation_model._DummyFeatureMethod",
            feature_layers=["encoder"],
        )
        model = DomainAdaptationModel(cfg)
        hook = model._feature_hook
        hook.remove = MagicMock()

        model.on_fit_end()

        hook.remove.assert_called_once()


class TestDomainAdaptationModelValDomainName:
    def test_only_source_val(self):
        cfg = _make_cfg(
            method_target="tests.test_domain_adaptation_model._DummyMethod",
            with_source_val=True,
        )
        model = DomainAdaptationModel(cfg)
        assert model._get_val_domain_name(0) == "source"

    def test_only_target_val(self):
        cfg = _make_cfg(
            method_target="tests.test_domain_adaptation_model._DummyMethod",
            with_target_val=True,
        )
        model = DomainAdaptationModel(cfg)
        assert model._get_val_domain_name(0) == "target"

    def test_both_val_sets_idx_0_is_source(self):
        cfg = _make_cfg(
            method_target="tests.test_domain_adaptation_model._DummyMethod",
            with_source_val=True,
            with_target_val=True,
        )
        model = DomainAdaptationModel(cfg)
        assert model._get_val_domain_name(0) == "source"
        assert model._get_val_domain_name(1) == "target"


class TestDomainAdaptationModelValidationStep:
    def test_validation_step_returns_none_without_mask(self):
        cfg = _make_cfg(method_target="tests.test_domain_adaptation_model._DummyMethod")
        model = DomainAdaptationModel(cfg)

        assert model.validation_step({"image": torch.randn(1, 3, 8, 8)}, 0) is None

    def test_validation_step_squeezes_channel_mask_and_logs_individual_losses(self):
        cfg = _make_cfg(
            method_target="tests.test_domain_adaptation_model._DummyMethod",
            with_source_val=True,
        )
        model = DomainAdaptationModel(cfg)
        model.log = MagicMock()
        batch = {
            "image": torch.randn(2, 3, 8, 8),
            "mask": torch.randint(0, 2, (2, 1, 8, 8)),
        }

        with patch.object(
            model,
            "_compute_loss",
            return_value=(torch.tensor(1.0), {"ce": torch.tensor(1.0)}, {}),
        ) as compute_loss:
            loss = model.validation_step(batch, 0)

        assert loss.shape == torch.Size([])
        assert compute_loss.call_args.args[1].shape == (2, 8, 8)
        logged_names = [call.args[0] for call in model.log.call_args_list]
        assert "loss/val_source" in logged_names
        assert "losses/val_source_ce" in logged_names


class TestDomainAdaptationModelLifecycle:
    def test_lifecycle_hooks_forward_to_method(self):
        cfg = _make_cfg(method_target="tests.test_domain_adaptation_model._DummyMethod")
        model = DomainAdaptationModel(cfg)
        model.method.on_fit_start = MagicMock()
        model.method.on_train_epoch_start = MagicMock()
        model.method.on_train_batch_start = MagicMock()
        model.method.on_train_epoch_end = MagicMock()
        model.trainer = MagicMock(num_training_batches=9)

        with patch.object(
            DomainAdaptationModel, "current_epoch", new_callable=PropertyMock
        ) as current_epoch:
            current_epoch.return_value = 4
            model.on_fit_start()
            model.on_train_epoch_start()
            model.on_train_batch_start({}, 3)
            model.on_train_epoch_end()

        model.method.on_fit_start.assert_called_once_with(model)
        model.method.on_train_epoch_start.assert_called_once_with(model, 4)
        model.method.on_train_batch_start.assert_called_once_with(model, 3, 9)
        model.method.on_train_epoch_end.assert_called_once_with(model, 4)

    def test_train_batch_start_uses_one_batch_without_trainer(self):
        cfg = _make_cfg(method_target="tests.test_domain_adaptation_model._DummyMethod")
        model = DomainAdaptationModel(cfg)
        model.method.on_train_batch_start = MagicMock()

        with patch.object(
            DomainAdaptationModel, "trainer", new_callable=PropertyMock
        ) as trainer_mock:
            trainer_mock.return_value = None
            model.on_train_batch_start({}, 2)

        model.method.on_train_batch_start.assert_called_once_with(model, 2, 1)
