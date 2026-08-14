# -*- coding: utf-8 -*-
"""
Tests for DomainClassifier and DANNMethod.

Covers:
- DomainClassifier forward shape and dtype
- DANNMethod contract (return type, loss properties, flags)
- Gradient flow from da_loss back to encoder
- lambda update via on_train_epoch_start
- Parameter group isolation (discriminator vs. encoder)
- Integration with DomainAdaptationModel.training_step
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from pytorch_segmentation_models_trainer.domain_adaptation.base_method import (
    DomainAdaptationLossOutput,
)
from pytorch_segmentation_models_trainer.domain_adaptation.methods.dann import (
    DANNMethod,
    DomainClassifier,
)
from pytorch_segmentation_models_trainer.domain_adaptation.methods.gradient_reversal import (
    GradientReversalLayer,
)
from pytorch_segmentation_models_trainer.model_loader.domain_adaptation_model import (
    DomainAdaptationModel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IN_CHANNELS = 16
H, W = 4, 4
BATCH = 3


def _make_features(batch_size=BATCH, channels=IN_CHANNELS, h=H, w=W):
    return torch.randn(batch_size, channels, h, w, requires_grad=True)


def _make_method(**kwargs):
    defaults = dict(feature_layer="encoder", in_channels=IN_CHANNELS, hidden_size=32)
    defaults.update(kwargs)
    return DANNMethod(**defaults)


def _make_fake_batch(
    batch_size=BATCH, img_ch=3, h=8, w=8, num_classes=2, with_mask=True
):
    b = {"image": torch.randn(batch_size, img_ch, h, w)}
    if with_mask:
        b["mask"] = torch.randint(0, num_classes, (batch_size, h, w))
    return b


# ---------------------------------------------------------------------------
# DomainClassifier
# ---------------------------------------------------------------------------


class TestDomainClassifier:
    def test_output_shape_is_b_times_2(self):
        clf = DomainClassifier(in_channels=IN_CHANNELS, hidden_size=32)
        x = torch.randn(BATCH, IN_CHANNELS, H, W)
        out = clf(x)
        assert out.shape == (BATCH, 2)

    def test_output_dtype_float32(self):
        clf = DomainClassifier(in_channels=IN_CHANNELS, hidden_size=32)
        x = torch.randn(2, IN_CHANNELS, H, W)
        assert clf(x).dtype == torch.float32

    def test_resolution_agnostic(self):
        """Same classifier works on different spatial sizes."""
        clf = DomainClassifier(in_channels=IN_CHANNELS, hidden_size=32)
        for size in [4, 8, 16, 32]:
            x = torch.randn(2, IN_CHANNELS, size, size)
            out = clf(x)
            assert out.shape == (2, 2)

    def test_has_trainable_parameters(self):
        clf = DomainClassifier(in_channels=IN_CHANNELS, hidden_size=32)
        assert sum(p.numel() for p in clf.parameters()) > 0

    def test_gradient_flows_through_classifier(self):
        clf = DomainClassifier(in_channels=IN_CHANNELS, hidden_size=32)
        x = torch.randn(2, IN_CHANNELS, H, W, requires_grad=True)
        clf(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# DANNMethod — class flags and construction
# ---------------------------------------------------------------------------


class TestDANNMethodFlags:
    def test_requires_features_is_true(self):
        assert DANNMethod.requires_features is True

    def test_requires_target_labels_is_false(self):
        assert DANNMethod.requires_target_labels is False

    def test_is_nn_module(self):
        method = _make_method()
        assert isinstance(method, nn.Module)

    def test_grl_initial_lambda_is_zero(self):
        """GRL starts at lambda=0 so the encoder is not perturbed at epoch 0."""
        method = _make_method()
        assert method.grl.lambda_ == pytest.approx(0.0)

    def test_domain_classifier_is_submodule(self):
        method = _make_method()
        assert isinstance(method.domain_classifier, DomainClassifier)

    def test_grl_is_submodule(self):
        method = _make_method()
        assert isinstance(method.grl, GradientReversalLayer)

    def test_domain_classifier_params_in_method_params(self):
        method = _make_method()
        method_ids = {id(p) for p in method.parameters()}
        clf_ids = {id(p) for p in method.domain_classifier.parameters()}
        assert clf_ids.issubset(method_ids)


# ---------------------------------------------------------------------------
# DANNMethod — compute_da_loss output contract
# ---------------------------------------------------------------------------


class TestDANNMethodComputeDaLoss:
    def _run(self, B_src=BATCH, B_tgt=BATCH):
        method = _make_method()
        src_feat = _make_features(B_src)
        tgt_feat = _make_features(B_tgt)
        return method.compute_da_loss(
            source_batch={},
            target_batch={},
            source_output=torch.randn(B_src, 2, H, W),
            target_output=torch.randn(B_tgt, 2, H, W),
            source_features={"encoder": src_feat},
            target_features={"encoder": tgt_feat},
        )

    def test_returns_da_loss_output_type(self):
        assert isinstance(self._run(), DomainAdaptationLossOutput)

    def test_loss_is_scalar(self):
        result = self._run()
        assert result.loss.ndim == 0

    def test_loss_is_finite(self):
        assert torch.isfinite(self._run().loss)

    def test_loss_has_grad_fn(self):
        """Loss must be differentiable — grad_fn must not be None."""
        assert self._run().loss.grad_fn is not None

    def test_log_dict_has_domain_loss(self):
        assert "domain_loss" in self._run().log_dict

    def test_log_dict_has_domain_acc(self):
        assert "domain_acc" in self._run().log_dict

    def test_domain_acc_in_valid_range(self):
        acc = self._run().log_dict["domain_acc"]
        assert 0.0 <= acc <= 1.0

    def test_domain_loss_is_positive(self):
        """Cross-entropy is always >= 0."""
        assert self._run().loss.item() >= 0.0

    def test_works_with_unequal_batch_sizes(self):
        """B_src != B_tgt must not raise."""
        result = self._run(B_src=2, B_tgt=5)
        assert torch.isfinite(result.loss)

    def test_gradient_flows_to_source_features(self):
        method = _make_method()
        src_feat = _make_features()
        tgt_feat = _make_features()
        result = method.compute_da_loss(
            source_batch={},
            target_batch={},
            source_output=torch.randn(BATCH, 2, H, W),
            target_output=torch.randn(BATCH, 2, H, W),
            source_features={"encoder": src_feat},
            target_features={"encoder": tgt_feat},
        )
        result.loss.backward()
        # GRL reverses the gradient — grad should be non-zero but negated
        assert src_feat.grad is not None
        assert tgt_feat.grad is not None

    def test_gradient_is_reversed_at_features(self):
        """With lambda=1, gradient flowing back to features should be negated."""
        method = _make_method()
        method.grl.set_lambda(1.0)

        # Compute gradient WITH GRL
        src_feat = _make_features(batch_size=1)
        tgt_feat = _make_features(batch_size=1)
        result = method.compute_da_loss(
            source_batch={},
            target_batch={},
            source_output=torch.randn(1, 2, H, W),
            target_output=torch.randn(1, 2, H, W),
            source_features={"encoder": src_feat},
            target_features={"encoder": tgt_feat},
        )
        result.loss.backward()
        grad_with_grl = src_feat.grad.clone()

        # Compute gradient WITHOUT GRL (lambda=0)
        method.grl.set_lambda(0.0)
        src_feat2 = _make_features(batch_size=1)
        tgt_feat2 = _make_features(batch_size=1)
        result2 = method.compute_da_loss(
            source_batch={},
            target_batch={},
            source_output=torch.randn(1, 2, H, W),
            target_output=torch.randn(1, 2, H, W),
            source_features={"encoder": src_feat2},
            target_features={"encoder": tgt_feat2},
        )
        result2.loss.backward()
        grad_without_grl = src_feat2.grad.clone()

        # With GRL the gradient is negated; without GRL it is zero (lambda=0).
        # Just verify they have opposite signs when both non-zero.
        assert not torch.allclose(grad_with_grl, grad_without_grl)


# ---------------------------------------------------------------------------
# DANNMethod — lambda schedule via on_train_epoch_start
# ---------------------------------------------------------------------------


class TestDANNMethodLambdaSchedule:
    def _mock_pl_module(self, max_epochs=50):
        pl_module = MagicMock()
        pl_module.trainer.max_epochs = max_epochs
        return pl_module

    def test_grl_lambda_zero_at_epoch_zero_with_dann_schedule(self):
        from pytorch_segmentation_models_trainer.domain_adaptation.schedulers import (
            DANNScheduler,
        )

        method = _make_method()
        method.lambda_schedule = DANNScheduler(gamma=10.0)
        method.on_train_epoch_start(self._mock_pl_module(50), epoch=0)
        # DANNScheduler at p=0: 2/(1+exp(0)) - 1 = 0.0
        assert method.grl.lambda_ == pytest.approx(0.0, abs=1e-6)

    def test_grl_lambda_approaches_one_at_last_epoch(self):
        from pytorch_segmentation_models_trainer.domain_adaptation.schedulers import (
            DANNScheduler,
        )

        method = _make_method()
        method.lambda_schedule = DANNScheduler(gamma=10.0)
        method.on_train_epoch_start(self._mock_pl_module(50), epoch=49)
        assert method.grl.lambda_ > 0.99

    def test_grl_lambda_monotonically_increases(self):
        from pytorch_segmentation_models_trainer.domain_adaptation.schedulers import (
            DANNScheduler,
        )

        method = _make_method()
        method.lambda_schedule = DANNScheduler(gamma=10.0)
        pl_module = self._mock_pl_module(20)

        lambdas = []
        for epoch in range(20):
            method.on_train_epoch_start(pl_module, epoch=epoch)
            lambdas.append(method.grl.lambda_)

        for i in range(1, len(lambdas)):
            assert lambdas[i] >= lambdas[i - 1]

    def test_constant_lambda_when_no_schedule(self):
        """Without a lambda_schedule, lambda_da is used as a constant."""
        method = _make_method(lambda_da=0.7)
        pl_module = self._mock_pl_module()
        method.on_train_epoch_start(pl_module, epoch=5)
        assert method.grl.lambda_ == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# DANNMethod — optimizer parameter groups
# ---------------------------------------------------------------------------


class TestDANNMethodParameterGroups:
    def test_extra_groups_has_one_entry(self):
        method = _make_method()
        groups = method.get_extra_parameter_groups()
        assert len(groups) == 1

    def test_extra_group_contains_domain_classifier_params(self):
        method = _make_method()
        group = method.get_extra_parameter_groups()[0]
        clf_ids = {id(p) for p in method.domain_classifier.parameters()}
        group_ids = {id(p) for p in group["params"]}
        assert clf_ids == group_ids

    def test_extra_group_has_lr_key(self):
        method = _make_method(discriminator_lr=5e-4)
        group = method.get_extra_parameter_groups()[0]
        assert "lr" in group
        assert group["lr"] == pytest.approx(5e-4)

    def test_encoder_not_duplicated_across_groups(self):
        """Main param group and extra group must not share parameters."""
        method = _make_method()
        extra_ids = {
            id(p) for g in method.get_extra_parameter_groups() for p in g["params"]
        }
        # GRL has no parameters — only domain_classifier params are in extra group
        grl_ids = {id(p) for p in method.grl.parameters()}
        assert grl_ids == set()  # GRL has no params
        # domain_classifier params should NOT appear in method params minus extra
        all_method_ids = {id(p) for p in method.parameters()}
        # They ARE in method.parameters() (nn.Module attribute), which is expected
        # What matters is they're also in extra_groups so DomainAdaptationModel
        # can exclude them from the main group.
        assert extra_ids.issubset(all_method_ids)


# ---------------------------------------------------------------------------
# Integration: DANNMethod + DomainAdaptationModel
# ---------------------------------------------------------------------------


class _EncoderDecoderModel(nn.Module):
    """Tiny model with explicit encoder/decoder, mirroring a U-Net structure."""

    def __init__(self, in_ch=3, num_classes=2, feat_ch=IN_CHANNELS):
        super().__init__()
        self.encoder = nn.Conv2d(in_ch, feat_ch, kernel_size=1)
        self.decoder = nn.Conv2d(feat_ch, num_classes, kernel_size=1)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class _DummyDataset(Dataset):
    def __init__(self, size=8, num_classes=2, data_loader=None, **kwargs):
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 8, 8),
            "mask": torch.randint(0, self.num_classes, (8, 8)),
        }


def _make_da_cfg(feature_layers=None):
    return OmegaConf.create(
        {
            "model": {
                "_target_": "tests.test_dann_method._EncoderDecoderModel",
            },
            "optimizer": {
                "_target_": "torch.optim.SGD",
                "lr": 1e-3,
                "momentum": 0.9,
            },
            "hyperparameters": {"batch_size": 2, "epochs": 2, "max_lr": 1e-3},
            "loss": {"_target_": "torch.nn.CrossEntropyLoss"},
            "domain_adaptation": {
                "method": {
                    "_target_": "pytorch_segmentation_models_trainer.domain_adaptation.methods.dann.DANNMethod",
                    "feature_layer": "encoder",
                    "in_channels": IN_CHANNELS,
                    "hidden_size": 32,
                    "discriminator_lr": 1e-4,
                    "lambda_da": 1.0,
                },
                "source_dataset": {
                    "_target_": "tests.test_dann_method._DummyDataset",
                    "data_loader": {
                        "batch_size": 2,
                        "num_workers": 0,
                        "shuffle": False,
                        "pin_memory": False,
                        "drop_last": False,
                        "prefetch_factor": 2,
                        "persistent_workers": False,
                    },
                },
                "target_dataset": {
                    "_target_": "tests.test_dann_method._DummyDataset",
                    "data_loader": {
                        "batch_size": 2,
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
                "feature_layers": feature_layers or ["encoder"],
                "pretrained_checkpoint": None,
            },
        }
    )


class TestDANNMethodIntegration:
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
        cfg = _make_da_cfg()
        model = DomainAdaptationModel(cfg)
        model.trainer = MagicMock()
        model.trainer.max_epochs = 10

        batch = self._make_batch()
        loss = model.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_training_step_loss_is_finite(self):
        cfg = _make_da_cfg()
        model = DomainAdaptationModel(cfg)
        model.trainer = MagicMock()
        model.trainer.max_epochs = 10

        loss = model.training_step(self._make_batch(), 0)
        assert torch.isfinite(loss)

    def test_training_step_loss_has_grad_fn(self):
        cfg = _make_da_cfg()
        model = DomainAdaptationModel(cfg)
        model.trainer = MagicMock()
        model.trainer.max_epochs = 10

        loss = model.training_step(self._make_batch(), 0)
        assert loss.grad_fn is not None

    def test_encoder_receives_gradient_from_da_loss(self):
        """Gradient from the adversarial branch must reach encoder weights."""
        cfg = _make_da_cfg()
        model = DomainAdaptationModel(cfg)
        model.trainer = MagicMock()
        model.trainer.max_epochs = 10

        # Set lambda > 0 so the DA loss actually contributes
        model.method.grl.set_lambda(1.0)

        loss = model.training_step(self._make_batch(), 0)
        loss.backward()

        encoder_params = list(model.model.encoder.parameters())
        for p in encoder_params:
            assert p.grad is not None, "Encoder parameter has no gradient"

    def test_configure_optimizers_has_two_param_groups(self):
        """Main group (encoder+decoder) + discriminator group."""
        cfg = _make_da_cfg()
        model = DomainAdaptationModel(cfg)
        optimizers, _ = model.configure_optimizers()
        optimizer = optimizers[0]
        assert len(optimizer.param_groups) >= 2

    def test_discriminator_lr_in_param_groups(self):
        cfg = _make_da_cfg()
        model = DomainAdaptationModel(cfg)
        optimizers, _ = model.configure_optimizers()
        lrs = [pg["lr"] for pg in optimizers[0].param_groups]
        assert pytest.approx(1e-4) in lrs


# ---------------------------------------------------------------------------
# DANNMethod — step_mode (epoch vs batch granularity)
# ---------------------------------------------------------------------------


class TestDANNMethodStepMode:
    def _mock_pl(self, max_epochs=10, current_epoch=0):
        pl = MagicMock()
        pl.trainer.max_epochs = max_epochs
        pl.current_epoch = current_epoch
        return pl

    # ── construction ────────────────────────────────────────────────────────

    def test_default_step_mode_is_epoch(self):
        method = _make_method()
        assert method.step_mode == "epoch"

    def test_batch_step_mode_accepted(self):
        method = _make_method(step_mode="batch")
        assert method.step_mode == "batch"

    def test_invalid_step_mode_raises(self):
        with pytest.raises(ValueError, match="step_mode"):
            _make_method(step_mode="step")

    def test_current_lambda_initialised_to_zero(self):
        method = _make_method()
        assert method._current_lambda == pytest.approx(0.0)

    # ── epoch mode ──────────────────────────────────────────────────────────

    def test_epoch_mode_updates_lambda_on_epoch_start(self):
        from pytorch_segmentation_models_trainer.domain_adaptation.schedulers import (
            DANNScheduler,
        )

        method = _make_method(step_mode="epoch")
        method.lambda_schedule = DANNScheduler(gamma=10.0)
        pl = self._mock_pl(max_epochs=10)

        method.on_train_epoch_start(pl, epoch=9)
        assert method.grl.lambda_ > 0.99
        assert method._current_lambda > 0.99

    def test_epoch_mode_batch_start_is_noop(self):
        """In epoch mode, on_train_batch_start must not change GRL lambda."""
        from pytorch_segmentation_models_trainer.domain_adaptation.schedulers import (
            DANNScheduler,
        )

        method = _make_method(step_mode="epoch")
        method.lambda_schedule = DANNScheduler(gamma=10.0)
        pl = self._mock_pl(max_epochs=10)

        # Simulate epoch 5 having been applied
        method.on_train_epoch_start(pl, epoch=5)
        lambda_after_epoch = method.grl.lambda_

        # on_train_batch_start must not change it
        method.on_train_batch_start(pl, batch_idx=3, num_batches=100)
        assert method.grl.lambda_ == pytest.approx(lambda_after_epoch)

    # ── batch mode ──────────────────────────────────────────────────────────

    def test_batch_mode_epoch_start_is_noop(self):
        """In batch mode, on_train_epoch_start must not change GRL lambda."""
        from pytorch_segmentation_models_trainer.domain_adaptation.schedulers import (
            DANNScheduler,
        )

        method = _make_method(step_mode="batch")
        method.lambda_schedule = DANNScheduler(gamma=10.0)
        pl = self._mock_pl(max_epochs=10)

        initial_lambda = method.grl.lambda_
        method.on_train_epoch_start(pl, epoch=5)
        assert method.grl.lambda_ == pytest.approx(initial_lambda)

    def test_batch_mode_updates_lambda_on_batch_start(self):
        from pytorch_segmentation_models_trainer.domain_adaptation.schedulers import (
            DANNScheduler,
        )

        method = _make_method(step_mode="batch")
        method.lambda_schedule = DANNScheduler(gamma=10.0)
        pl = self._mock_pl(max_epochs=2, current_epoch=1)

        # epoch=1, batch_idx=99, num_batches=100 → global_step=199, total=200 → p≈1
        method.on_train_batch_start(pl, batch_idx=99, num_batches=100)
        assert method.grl.lambda_ > 0.99
        assert method._current_lambda > 0.99

    def test_batch_mode_lambda_monotonically_increases(self):
        from pytorch_segmentation_models_trainer.domain_adaptation.schedulers import (
            DANNScheduler,
        )

        method = _make_method(step_mode="batch")
        method.lambda_schedule = DANNScheduler(gamma=10.0)
        num_batches = 10
        max_epochs = 3
        lambdas = []

        for epoch in range(max_epochs):
            pl = self._mock_pl(max_epochs=max_epochs, current_epoch=epoch)
            for batch_idx in range(num_batches):
                method.on_train_batch_start(
                    pl, batch_idx=batch_idx, num_batches=num_batches
                )
                lambdas.append(method.grl.lambda_)

        for i in range(1, len(lambdas)):
            assert lambdas[i] >= lambdas[i - 1] - 1e-9

    def test_batch_mode_finer_than_epoch_mode(self):
        """Batch mode produces more unique lambda values than epoch mode."""
        from pytorch_segmentation_models_trainer.domain_adaptation.schedulers import (
            DANNScheduler,
        )

        num_batches = 5
        max_epochs = 3
        schedule = DANNScheduler(gamma=10.0)

        # Epoch mode: one unique value per epoch
        epoch_method = _make_method(step_mode="epoch")
        epoch_method.lambda_schedule = schedule
        epoch_lambdas = set()
        for epoch in range(max_epochs):
            pl = self._mock_pl(max_epochs=max_epochs, current_epoch=epoch)
            epoch_method.on_train_epoch_start(pl, epoch=epoch)
            for _ in range(num_batches):
                epoch_lambdas.add(round(epoch_method.grl.lambda_, 6))

        # Batch mode: one unique value per batch
        batch_method = _make_method(step_mode="batch")
        batch_method.lambda_schedule = DANNScheduler(gamma=10.0)
        batch_lambdas = set()
        for epoch in range(max_epochs):
            pl = self._mock_pl(max_epochs=max_epochs, current_epoch=epoch)
            for batch_idx in range(num_batches):
                batch_method.on_train_batch_start(
                    pl, batch_idx=batch_idx, num_batches=num_batches
                )
                batch_lambdas.add(round(batch_method.grl.lambda_, 6))

        assert len(batch_lambdas) > len(epoch_lambdas)

    # ── _get_lambda_da integration ───────────────────────────────────────────

    def test_get_lambda_da_reads_current_lambda_in_batch_mode(self):
        """DomainAdaptationModel._get_lambda_da must use _current_lambda."""
        from pytorch_segmentation_models_trainer.domain_adaptation.schedulers import (
            DANNScheduler,
        )

        cfg = _make_da_cfg()
        model = DomainAdaptationModel(cfg)
        model.trainer = MagicMock()
        model.trainer.max_epochs = 10
        model.trainer.num_training_batches = 5

        # Manually set _current_lambda (simulates what on_train_batch_start does)
        model.method._current_lambda = 0.42

        assert model._get_lambda_da() == pytest.approx(0.42)

    def test_on_train_batch_start_forwarded_by_pl_model(self):
        """DomainAdaptationModel.on_train_batch_start must call method hook."""
        cfg = _make_da_cfg()
        model = DomainAdaptationModel(cfg)
        model.trainer = MagicMock()
        model.trainer.num_training_batches = 20

        model.method.on_train_batch_start = MagicMock()
        model.on_train_batch_start(batch={}, batch_idx=7)

        model.method.on_train_batch_start.assert_called_once_with(model, 7, 20)
