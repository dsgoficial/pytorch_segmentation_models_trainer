# -*- coding: utf-8 -*-
"""KL annealing callback for variational autoencoders."""

import math

import pytorch_lightning as pl

_VALID_SCHEDULES = {"linear", "cosine", "cyclical"}


class KLAnnealingCallback(pl.Callback):
    """Gradually increases the KL weight (beta) in a VAE loss during training.

    Supports linear, cosine, and cyclical annealing schedules. On each
    update the current beta is written to ``pl_module.loss_function.beta``
    and logged as ``scheduler/kl_beta`` for TensorBoard.

    Args:
        max_beta: Target KL weight reached at the end of the annealing period.
        min_beta: Starting KL weight (default 0 for full free-bits warmup).
        annealing_steps: Number of steps (or epochs when ``use_epochs=True``)
            to ramp from ``min_beta`` to ``max_beta``.
        schedule: Annealing curve — ``"linear"``, ``"cosine"``, or
            ``"cyclical"``.
        use_epochs: When ``True``, annealing is driven by
            ``trainer.current_epoch``; when ``False`` (default) by
            ``trainer.global_step``.
        cycle_length: Total length of one cycle (``"cyclical"`` only).
        cycle_ratio: Fraction of each cycle spent ramping up
            (``"cyclical"`` only).
        **kwargs: Reserved for Hydra compatibility.

    Example YAML:
        callbacks:
          - _target_: pytorch_segmentation_models_trainer.custom_callbacks.kl_annealing_callback.KLAnnealingCallback
            max_beta: 1.0
            min_beta: 0.0
            annealing_steps: 5000
            schedule: cosine
    """

    def __init__(
        self,
        max_beta: float = 1.0,
        min_beta: float = 0.0,
        annealing_steps: int = 1000,
        schedule: str = "linear",
        use_epochs: bool = False,
        cycle_length: int = 100,
        cycle_ratio: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        if schedule not in _VALID_SCHEDULES:
            raise ValueError(
                f"schedule must be one of {sorted(_VALID_SCHEDULES)!r}; got {schedule!r}"
            )
        self.max_beta = max_beta
        self.min_beta = min_beta
        self.annealing_steps = annealing_steps
        self.schedule = schedule
        self.use_epochs = use_epochs
        self.cycle_length = cycle_length
        self.cycle_ratio = cycle_ratio

    def _compute_beta(self, t: int) -> float:
        """Return the annealed beta value at position ``t``.

        Args:
            t: Current step or epoch index.

        Returns:
            Scalar beta in ``[min_beta, max_beta]``.
        """
        span = self.max_beta - self.min_beta

        if self.schedule == "linear":
            ratio = min(t / self.annealing_steps, 1.0)
            return self.min_beta + span * ratio

        if self.schedule == "cosine":
            ratio = min(t / self.annealing_steps, 1.0)
            return self.min_beta + span * 0.5 * (1.0 - math.cos(math.pi * ratio))

        # cyclical
        pos_in_cycle = t % self.cycle_length
        ramp_steps = self.cycle_length * self.cycle_ratio
        ratio = min(pos_in_cycle / ramp_steps, 1.0)
        return self.min_beta + span * ratio

    def _apply_beta(self, trainer, pl_module, t: int, on_step: bool) -> None:
        beta = self._compute_beta(t)
        if hasattr(pl_module, "loss_function") and hasattr(
            pl_module.loss_function, "beta"
        ):
            pl_module.loss_function.beta = beta
        pl_module.log(
            "scheduler/kl_beta",
            beta,
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=False,
            sync_dist=True,
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if self.use_epochs:
            return
        self._apply_beta(trainer, pl_module, trainer.global_step, on_step=True)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if not self.use_epochs:
            return
        self._apply_beta(trainer, pl_module, trainer.current_epoch, on_step=False)
