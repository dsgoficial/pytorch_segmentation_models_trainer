# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-04-16
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

import torch
import torch.nn as nn
from torch import Tensor


class GradientReversalFunction(torch.autograd.Function):
    """Custom autograd function that reverses and scales the gradient on the backward pass.

    During the forward pass this function acts as an identity mapping.
    During the backward pass the incoming gradient is multiplied by ``-lambda_``,
    effectively reversing its sign and scaling its magnitude.

    This is the canonical implementation from:
        Ganin, Y. & Lempitsky, V. "Unsupervised Domain Adaptation by
        Backpropagation." ICML 2015.

    ``torch.autograd.Function`` is used instead of ``Tensor.register_hook``
    because:
    - It integrates cleanly with ``torch.compile`` and ``torch.jit``.
    - It does not hold a reference to the computation graph beyond the
      backward call, avoiding memory leaks.
    - ``lambda_`` is passed as a plain scalar (not a tensor parameter), so it
      does not appear in ``model.parameters()`` and does not consume device
      memory.
    """

    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float) -> Tensor:
        ctx.save_for_backward(torch.as_tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (lambda_,) = ctx.saved_tensors
        # Return gradient for x (reversed + scaled) and None for lambda_
        # (lambda_ is not a tensor that requires grad).
        return -lambda_.item() * grad_output, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer (GRL) for adversarial domain adaptation.

    Wraps :class:`GradientReversalFunction` as a standard ``nn.Module`` so it
    can be placed in a ``nn.Sequential`` or called directly inside
    ``compute_da_loss``.

    During the forward pass the layer is a transparent identity mapping.
    During backpropagation the gradient is multiplied by ``-lambda_``:

    * ``lambda_ = 0``: gradient is completely blocked (useful at epoch 0 of
      DANN training, before the domain classifier has learned anything).
    * ``lambda_ = 1``: gradient is fully reversed (standard GRL behaviour).
    * Values between 0 and 1 can be set by a :class:`DANNScheduler
      <pytorch_segmentation_models_trainer.domain_adaptation.schedulers.DANNScheduler>`
      that gradually ramps up the adversarial pressure over training.

    Args:
        lambda_: Initial reversal coefficient. Updated each epoch via
            :meth:`set_lambda`. Default: ``1.0``.

    Example::

        grl = GradientReversalLayer(lambda_=0.5)
        out = grl(features)          # forward: identity
        out.sum().backward()         # backward: gradient * -0.5 reaches features

        # Update coefficient each epoch
        grl.set_lambda(new_lambda)
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_: float = lambda_

    def forward(self, x: Tensor) -> Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, value: float) -> None:
        """Update the reversal coefficient.

        Args:
            value: New lambda value. Typically called at the start of each
                training epoch by :meth:`DANNMethod.on_train_epoch_start`.
        """
        self.lambda_ = value

    def extra_repr(self) -> str:
        return f"lambda_={self.lambda_:.4f}"
