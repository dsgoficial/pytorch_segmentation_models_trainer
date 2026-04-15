# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-04-14
        git sha              : $Format:%H$
        copyright            : (C) 2026 by Philipe Borba - Cartographic Engineer
                                                            @ Brazilian Army
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ****
"""
from __future__ import annotations

import math

import torch.nn as nn


class BaseLambdaScheduler(nn.Module):
    """Abstract base class for DA loss weight (lambda) schedulers.

    A lambda scheduler controls how the DA loss weight evolves over training.
    Schedulers are ``nn.Module`` subclasses so they can be instantiated via
    Hydra and owned by a ``BaseDomainAdaptationMethod``.

    Usage inside a method::

        class MyMethod(BaseDomainAdaptationMethod):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.lambda_schedule = instantiate(cfg.lambda_schedule)

            def compute_da_loss(self, ...):
                # get current lambda from DomainAdaptationModel
                lam = self.lambda_schedule.get_lambda(epoch, total_epochs)
                ...
    """

    def get_lambda(self, epoch: int, total_epochs: int) -> float:
        """Return the lambda value for the given training progress.

        Args:
            epoch: Current epoch index (0-based).
            total_epochs: Total number of training epochs (``trainer.max_epochs``).

        Returns:
            Lambda scalar in the range appropriate for the scheduler.

        Raises:
            NotImplementedError: If not overridden by the subclass.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement get_lambda()")

    def forward(self, epoch: int, total_epochs: int) -> float:
        """Alias for ``get_lambda`` so the scheduler can be called directly."""
        return self.get_lambda(epoch, total_epochs)


class ConstantScheduler(BaseLambdaScheduler):
    """Returns a fixed lambda value throughout training.

    Args:
        value: The constant lambda value to return. Default: ``1.0``.

    Example YAML::

        lambda_schedule:
          _target_: ...schedulers.ConstantScheduler
          value: 1.0
    """

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.value = value

    def get_lambda(self, epoch: int, total_epochs: int) -> float:
        return self.value


class LinearScheduler(BaseLambdaScheduler):
    """Linearly interpolates lambda from ``start_value`` to ``end_value``.

    Reaches ``end_value`` at the last epoch (``total_epochs - 1``).

    Args:
        start_value: Lambda at epoch 0. Default: ``0.0``.
        end_value: Lambda at the final epoch. Default: ``1.0``.

    Example YAML::

        lambda_schedule:
          _target_: ...schedulers.LinearScheduler
          start_value: 0.0
          end_value: 1.0
    """

    def __init__(self, start_value: float = 0.0, end_value: float = 1.0):
        super().__init__()
        self.start_value = start_value
        self.end_value = end_value

    def get_lambda(self, epoch: int, total_epochs: int) -> float:
        if total_epochs <= 1:
            return self.end_value
        progress = epoch / (total_epochs - 1)
        return self.start_value + progress * (self.end_value - self.start_value)


class DANNScheduler(BaseLambdaScheduler):
    """DANN annealing schedule from Ganin et al. (2016).

    Computes:

    .. math::

        \\lambda = \\frac{2}{1 + \\exp(-\\gamma \\cdot p)} - 1

    where :math:`p = \\text{epoch} / (\\text{total\_epochs} - 1)` is the
    training progress in ``[0, 1]``.

    This schedule starts near 0, grows slowly, then accelerates toward 1,
    allowing the feature extractor to stabilize before the adversarial
    pressure is fully applied.

    Args:
        gamma: Controls the steepness of the growth curve. Higher values
            produce a sharper transition. Default: ``10.0``.

    Reference:
        Ganin, Y. et al. "Domain-adversarial training of neural networks."
        JMLR 2016.

    Example YAML::

        lambda_schedule:
          _target_: ...schedulers.DANNScheduler
          gamma: 10.0
    """

    def __init__(self, gamma: float = 10.0):
        super().__init__()
        self.gamma = gamma

    def get_lambda(self, epoch: int, total_epochs: int) -> float:
        if total_epochs <= 1:
            return 1.0
        progress = epoch / (total_epochs - 1)
        return 2.0 / (1.0 + math.exp(-self.gamma * progress)) - 1.0
