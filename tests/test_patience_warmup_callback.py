# -*- coding: utf-8 -*-
"""Tests for PatienceWarmupCallback."""

from unittest.mock import MagicMock, call

import pytest
import torch

from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import (
    PatienceWarmupCallback,
)
from pytorch_segmentation_models_trainer.config_definitions.autoencoder_config import (
    PatienceWarmupCallbackConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer(epoch=0, metrics=None):
    trainer = MagicMock()
    trainer.current_epoch = epoch
    trainer.callback_metrics = metrics or {}
    return trainer


def _make_pl_module():
    pl_module = MagicMock()
    return pl_module


def _run_epochs(cb, n_epochs, metric_values, monitor="val/reconstruction_loss"):
    """Simulate n_epochs of validation end events with given metric values."""
    trainer = _make_trainer()
    pl_module = _make_pl_module()
    cb.on_fit_start(trainer, pl_module)
    for epoch, value in enumerate(metric_values[:n_epochs]):
        trainer.current_epoch = epoch
        trainer.callback_metrics = {monitor: torch.tensor(value)}
        cb.on_validation_epoch_end(trainer, pl_module)
    return trainer, pl_module


# ===========================================================================
# Freeze on fit_start
# ===========================================================================


def test_freezes_encoder_on_fit_start():
    cb = PatienceWarmupCallback(monitor="val/reconstruction_loss", patience=3)
    trainer = _make_trainer()
    pl_module = _make_pl_module()
    cb.on_fit_start(trainer, pl_module)
    pl_module.set_encoder_trainable.assert_called_once_with(trainable=False)


def test_does_not_refreeze_if_already_warmed_up():
    cb = PatienceWarmupCallback(monitor="val/reconstruction_loss", patience=3)
    cb._warmed_up = True
    trainer = _make_trainer()
    pl_module = _make_pl_module()
    cb.on_fit_start(trainer, pl_module)
    pl_module.set_encoder_trainable.assert_not_called()


# ===========================================================================
# Patience logic
# ===========================================================================


def test_unfreezes_after_patience_exhausted():
    cb = PatienceWarmupCallback(monitor="val/reconstruction_loss", patience=3)
    # constant loss = no improvement → patience exhausted after 3 epochs
    values = [1.0, 1.0, 1.0, 1.0]
    trainer, pl_module = _run_epochs(cb, 4, values)
    pl_module.set_encoder_trainable.assert_called_with(trainable=True)
    assert cb._warmed_up


def test_does_not_unfreeze_while_improving():
    cb = PatienceWarmupCallback(monitor="val/reconstruction_loss", patience=3)
    # steadily improving → never exhausts patience
    values = [1.0, 0.9, 0.8, 0.7, 0.6]
    trainer, pl_module = _run_epochs(cb, 5, values)
    # should not have called with trainable=True
    calls = [
        c
        for c in pl_module.set_encoder_trainable.call_args_list
        if c == call(trainable=True)
    ]
    assert len(calls) == 0
    assert not cb._warmed_up


def test_resets_wait_counter_on_improvement():
    cb = PatienceWarmupCallback(monitor="val/reconstruction_loss", patience=3)
    # 2 stagnant, then improve, then 2 more stagnant → should NOT unfreeze at epoch 4
    values = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
    trainer, pl_module = _run_epochs(cb, 4, values)
    calls = [
        c
        for c in pl_module.set_encoder_trainable.call_args_list
        if c == call(trainable=True)
    ]
    assert len(calls) == 0


def test_unfreezes_after_patience_post_reset():
    cb = PatienceWarmupCallback(monitor="val/reconstruction_loss", patience=3)
    # improve once, then stagnate for 3 → should unfreeze
    values = [1.0, 0.5, 0.5, 0.5, 0.5]
    trainer, pl_module = _run_epochs(cb, 5, values)
    assert cb._warmed_up


def test_min_delta_filters_tiny_improvements():
    cb = PatienceWarmupCallback(
        monitor="val/reconstruction_loss", patience=3, min_delta=0.1
    )
    # tiny drops < min_delta → treated as no improvement
    values = [1.0, 0.99, 0.98, 0.97]
    trainer, pl_module = _run_epochs(cb, 4, values)
    assert cb._warmed_up


# ===========================================================================
# min_epochs guard
# ===========================================================================


def test_respects_min_epochs():
    cb = PatienceWarmupCallback(
        monitor="val/reconstruction_loss", patience=1, min_epochs=5
    )
    # patience=1 would normally unfreeze at epoch 1, but min_epochs=5 blocks it
    values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    trainer = _make_trainer()
    pl_module = _make_pl_module()
    cb.on_fit_start(trainer, pl_module)
    for epoch, value in enumerate(values):
        trainer.current_epoch = epoch
        trainer.callback_metrics = {"val/reconstruction_loss": torch.tensor(value)}
        cb.on_validation_epoch_end(trainer, pl_module)
        if epoch < 5:
            unfreeze_calls = [
                c
                for c in pl_module.set_encoder_trainable.call_args_list
                if c == call(trainable=True)
            ]
            assert len(unfreeze_calls) == 0, f"unfroze too early at epoch {epoch}"


# ===========================================================================
# mode=max
# ===========================================================================


def test_mode_max_unfreezes_when_metric_stops_rising():
    cb = PatienceWarmupCallback(monitor="val/silhouette", patience=3, mode="max")
    # silhouette stagnates at 0.5
    values = [0.5, 0.5, 0.5, 0.5]
    trainer, pl_module = _run_epochs(cb, 4, values, monitor="val/silhouette")
    assert cb._warmed_up


def test_mode_max_does_not_unfreeze_while_rising():
    cb = PatienceWarmupCallback(monitor="val/silhouette", patience=3, mode="max")
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    trainer, pl_module = _run_epochs(cb, 5, values, monitor="val/silhouette")
    assert not cb._warmed_up


# ===========================================================================
# Missing metric / already warmed up
# ===========================================================================


def test_skips_epoch_when_monitor_not_in_metrics():
    cb = PatienceWarmupCallback(monitor="val/reconstruction_loss", patience=2)
    trainer = _make_trainer(metrics={})  # metric absent
    pl_module = _make_pl_module()
    cb.on_fit_start(trainer, pl_module)
    cb.on_validation_epoch_end(trainer, pl_module)
    # wait counter should not increment
    assert cb._wait == 0
    assert not cb._warmed_up


def test_skips_validation_epoch_when_already_warmed_up():
    cb = PatienceWarmupCallback(monitor="val/reconstruction_loss", patience=2)
    cb._warmed_up = True
    trainer = _make_trainer(metrics={"val/reconstruction_loss": torch.tensor(1.0)})
    pl_module = _make_pl_module()
    cb.on_validation_epoch_end(trainer, pl_module)
    pl_module.set_encoder_trainable.assert_not_called()


# ===========================================================================
# Logging
# ===========================================================================


def test_logs_unfreeze_epoch():
    cb = PatienceWarmupCallback(monitor="val/reconstruction_loss", patience=2)
    values = [1.0, 1.0, 1.0]
    trainer, pl_module = _run_epochs(cb, 3, values)
    pl_module.log.assert_called()


# ===========================================================================
# Validation
# ===========================================================================


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        PatienceWarmupCallback(monitor="val/loss", patience=3, mode="invalid")


def test_invalid_patience_raises():
    with pytest.raises(ValueError, match="patience"):
        PatienceWarmupCallback(monitor="val/loss", patience=0)


def test_invalid_min_delta_raises():
    with pytest.raises(ValueError, match="min_delta"):
        PatienceWarmupCallback(monitor="val/loss", patience=3, min_delta=-0.1)


# ===========================================================================
# Config dataclass
# ===========================================================================


def test_config_dataclass_target():
    cfg = PatienceWarmupCallbackConfig()
    assert "PatienceWarmupCallback" in cfg._target_


def test_config_dataclass_defaults():
    cfg = PatienceWarmupCallbackConfig()
    assert cfg.patience == 5
    assert cfg.mode == "min"
    assert cfg.min_epochs == 0
    assert cfg.min_delta == 1e-4
