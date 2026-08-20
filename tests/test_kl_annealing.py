# -*- coding: utf-8 -*-
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from pytorch_segmentation_models_trainer.custom_callbacks.kl_annealing_callback import (
    KLAnnealingCallback,
)
from pytorch_segmentation_models_trainer.config_definitions.autoencoder_config import (
    KLAnnealingCallbackConfig,
)

# ──────────────────────────────────────────────────────────────────────────────
# Schedule shape tests
# ──────────────────────────────────────────────────────────────────────────────


def test_linear_schedule_increases_monotonically():
    cb = KLAnnealingCallback(
        min_beta=0.0, max_beta=1.0, annealing_steps=100, schedule="linear"
    )
    betas = [cb._compute_beta(t) for t in range(110)]
    for i in range(len(betas) - 1):
        assert betas[i] <= betas[i + 1] + 1e-9
    assert betas[0] == pytest.approx(0.0)
    assert betas[100] == pytest.approx(1.0)
    assert betas[109] == pytest.approx(1.0)


def test_linear_schedule_midpoint():
    cb = KLAnnealingCallback(
        min_beta=0.0, max_beta=1.0, annealing_steps=100, schedule="linear"
    )
    assert cb._compute_beta(50) == pytest.approx(0.5)


def test_cosine_schedule_endpoints():
    cb = KLAnnealingCallback(
        min_beta=0.0, max_beta=1.0, annealing_steps=100, schedule="cosine"
    )
    assert cb._compute_beta(0) == pytest.approx(0.0, abs=1e-6)
    assert cb._compute_beta(100) == pytest.approx(1.0, abs=1e-6)


def test_cosine_schedule_stays_in_bounds():
    cb = KLAnnealingCallback(
        min_beta=0.0, max_beta=1.0, annealing_steps=100, schedule="cosine"
    )
    for t in range(110):
        assert -1e-9 <= cb._compute_beta(t) <= 1.0 + 1e-9


def test_cosine_schedule_monotonically_non_decreasing():
    cb = KLAnnealingCallback(
        min_beta=0.0, max_beta=1.0, annealing_steps=100, schedule="cosine"
    )
    betas = [cb._compute_beta(t) for t in range(101)]
    for i in range(len(betas) - 1):
        assert betas[i] <= betas[i + 1] + 1e-9


def test_cyclical_schedule_resets_each_cycle():
    cb = KLAnnealingCallback(
        min_beta=0.0,
        max_beta=1.0,
        schedule="cyclical",
        cycle_length=20,
        cycle_ratio=0.5,
    )
    b0 = cb._compute_beta(0)
    b20 = cb._compute_beta(20)
    b40 = cb._compute_beta(40)
    assert b0 == pytest.approx(b20)
    assert b0 == pytest.approx(b40)


def test_cyclical_schedule_reaches_max_within_cycle():
    cb = KLAnnealingCallback(
        min_beta=0.0,
        max_beta=1.0,
        schedule="cyclical",
        cycle_length=20,
        cycle_ratio=0.5,
    )
    # ratio * cycle_length = 10 steps to reach max
    assert cb._compute_beta(10) == pytest.approx(1.0)
    assert cb._compute_beta(15) == pytest.approx(1.0)


def test_beta_clamped_to_max_after_annealing():
    cb = KLAnnealingCallback(
        min_beta=0.0, max_beta=0.5, annealing_steps=10, schedule="linear"
    )
    assert cb._compute_beta(10) == pytest.approx(0.5)
    assert cb._compute_beta(999) == pytest.approx(0.5)


def test_min_beta_is_starting_point():
    cb = KLAnnealingCallback(
        min_beta=0.2, max_beta=1.0, annealing_steps=100, schedule="linear"
    )
    assert cb._compute_beta(0) == pytest.approx(0.2)


# ──────────────────────────────────────────────────────────────────────────────
# Callback hook integration
# ──────────────────────────────────────────────────────────────────────────────


def test_step_based_updates_beta_on_batch_start():
    cb = KLAnnealingCallback(
        min_beta=0.0, max_beta=1.0, annealing_steps=10, schedule="linear"
    )
    trainer = MagicMock()
    trainer.global_step = 5
    pl_module = MagicMock()
    cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
    assert pl_module.loss_function.beta == pytest.approx(0.5)


def test_epoch_based_updates_beta_on_epoch_start():
    cb = KLAnnealingCallback(
        min_beta=0.0,
        max_beta=1.0,
        annealing_steps=5,
        schedule="linear",
        use_epochs=True,
    )
    trainer = MagicMock()
    trainer.current_epoch = 2
    pl_module = MagicMock()
    cb.on_train_epoch_start(trainer, pl_module)
    assert pl_module.loss_function.beta == pytest.approx(2 / 5)


def test_step_based_does_not_trigger_on_epoch_start():
    cb = KLAnnealingCallback(
        min_beta=0.0,
        max_beta=1.0,
        annealing_steps=10,
        schedule="linear",
        use_epochs=False,
    )
    trainer = MagicMock()
    trainer.current_epoch = 5
    pl_module = MagicMock()
    initial_beta = 0.0
    pl_module.loss_function.beta = initial_beta
    cb.on_train_epoch_start(trainer, pl_module)
    assert pl_module.loss_function.beta == initial_beta


def test_epoch_based_does_not_trigger_on_batch_start():
    cb = KLAnnealingCallback(
        min_beta=0.0,
        max_beta=1.0,
        annealing_steps=10,
        schedule="linear",
        use_epochs=True,
    )
    trainer = MagicMock()
    trainer.global_step = 5
    pl_module = MagicMock()
    initial_beta = 0.0
    pl_module.loss_function.beta = initial_beta
    cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
    assert pl_module.loss_function.beta == initial_beta


def test_callback_skips_if_loss_has_no_beta():
    cb = KLAnnealingCallback(min_beta=0.0, max_beta=1.0, annealing_steps=10)
    trainer = MagicMock()
    trainer.global_step = 5
    pl_module = MagicMock()
    pl_module.loss_function = nn.MSELoss()  # no beta attribute
    cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
    # Must not raise; logging still happens
    pl_module.log.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────────
# TensorBoard logging
# ──────────────────────────────────────────────────────────────────────────────


def test_logs_beta_step_based():
    cb = KLAnnealingCallback(
        min_beta=0.0, max_beta=1.0, annealing_steps=10, schedule="linear"
    )
    trainer = MagicMock()
    trainer.global_step = 5
    pl_module = MagicMock()
    cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
    args, kwargs = pl_module.log.call_args
    assert args[0] == "scheduler/kl_beta"
    assert args[1] == pytest.approx(0.5)
    assert kwargs["on_step"] is True
    assert kwargs["on_epoch"] is False
    assert kwargs["prog_bar"] is False
    assert kwargs["sync_dist"] is True


def test_logs_beta_epoch_based():
    cb = KLAnnealingCallback(
        min_beta=0.0,
        max_beta=1.0,
        annealing_steps=10,
        schedule="linear",
        use_epochs=True,
    )
    trainer = MagicMock()
    trainer.current_epoch = 3
    pl_module = MagicMock()
    cb.on_train_epoch_start(trainer, pl_module)
    args, kwargs = pl_module.log.call_args
    assert args[0] == "scheduler/kl_beta"
    assert args[1] == pytest.approx(0.3)
    assert kwargs["on_step"] is False
    assert kwargs["on_epoch"] is True


# ──────────────────────────────────────────────────────────────────────────────
# start_after delay
# ──────────────────────────────────────────────────────────────────────────────


def test_start_after_holds_min_beta_before_delay():
    cb = KLAnnealingCallback(
        min_beta=0.0,
        max_beta=1.0,
        annealing_steps=100,
        schedule="linear",
        start_after=50,
    )
    for t in range(50):
        assert cb._apply_beta.__func__ or True  # access via public hook below
    trainer = MagicMock()
    pl_module = MagicMock()
    for step in range(50):
        trainer.global_step = step
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
        assert pl_module.loss_function.beta == pytest.approx(0.0)


def test_start_after_annealing_begins_at_delay_boundary():
    cb = KLAnnealingCallback(
        min_beta=0.0,
        max_beta=1.0,
        annealing_steps=100,
        schedule="linear",
        start_after=50,
    )
    trainer = MagicMock()
    pl_module = MagicMock()
    trainer.global_step = 50
    cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
    assert pl_module.loss_function.beta == pytest.approx(0.0)


def test_start_after_annealing_midpoint_offset():
    cb = KLAnnealingCallback(
        min_beta=0.0,
        max_beta=1.0,
        annealing_steps=100,
        schedule="linear",
        start_after=50,
    )
    trainer = MagicMock()
    pl_module = MagicMock()
    trainer.global_step = 100  # 50 steps into the annealing ramp
    cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
    assert pl_module.loss_function.beta == pytest.approx(0.5)


def test_start_after_annealing_completes_after_delay_plus_steps():
    cb = KLAnnealingCallback(
        min_beta=0.0,
        max_beta=1.0,
        annealing_steps=100,
        schedule="linear",
        start_after=50,
    )
    trainer = MagicMock()
    pl_module = MagicMock()
    trainer.global_step = 150  # start_after + annealing_steps
    cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
    assert pl_module.loss_function.beta == pytest.approx(1.0)


def test_start_after_zero_is_identical_to_no_delay():
    cb_with = KLAnnealingCallback(
        min_beta=0.0,
        max_beta=1.0,
        annealing_steps=100,
        schedule="cosine",
        start_after=0,
    )
    cb_without = KLAnnealingCallback(
        min_beta=0.0,
        max_beta=1.0,
        annealing_steps=100,
        schedule="cosine",
    )
    for t in range(110):
        trainer = MagicMock()
        trainer.global_step = t
        pm_with = MagicMock()
        pm_without = MagicMock()
        cb_with.on_train_batch_start(trainer, pm_with, batch=None, batch_idx=0)
        cb_without.on_train_batch_start(trainer, pm_without, batch=None, batch_idx=0)
        assert pm_with.loss_function.beta == pytest.approx(
            pm_without.loss_function.beta
        )


def test_start_after_epoch_based_holds_min_beta():
    cb = KLAnnealingCallback(
        min_beta=0.0,
        max_beta=1.0,
        annealing_steps=10,
        schedule="linear",
        use_epochs=True,
        start_after=5,
    )
    trainer = MagicMock()
    pl_module = MagicMock()
    for epoch in range(5):
        trainer.current_epoch = epoch
        cb.on_train_epoch_start(trainer, pl_module)
        assert pl_module.loss_function.beta == pytest.approx(0.0)


def test_start_after_epoch_based_midpoint():
    cb = KLAnnealingCallback(
        min_beta=0.0,
        max_beta=1.0,
        annealing_steps=10,
        schedule="linear",
        use_epochs=True,
        start_after=5,
    )
    trainer = MagicMock()
    pl_module = MagicMock()
    trainer.current_epoch = 10  # 5 epochs into the ramp
    cb.on_train_epoch_start(trainer, pl_module)
    assert pl_module.loss_function.beta == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────


def test_invalid_schedule_raises():
    with pytest.raises(ValueError, match="schedule"):
        KLAnnealingCallback(schedule="warmup_cosine_restart")


# ──────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ──────────────────────────────────────────────────────────────────────────────


def test_config_defaults():
    cfg = KLAnnealingCallbackConfig()
    assert cfg.max_beta == 1.0
    assert cfg.min_beta == 0.0
    assert cfg.schedule == "linear"
    assert cfg.use_epochs is False
    assert cfg.cycle_length == 100
    assert cfg.cycle_ratio == 0.5
    assert cfg.start_after == 0


def test_config_annealing_steps_is_required():
    from omegaconf import MissingMandatoryValue
    import omegaconf

    cfg = omegaconf.OmegaConf.structured(KLAnnealingCallbackConfig)
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.annealing_steps


def test_config_target_points_to_callback():
    cfg = KLAnnealingCallbackConfig()
    assert "KLAnnealingCallback" in cfg._target_
