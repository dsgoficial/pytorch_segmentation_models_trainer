# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-04-24
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

Utility functions for Monte Carlo Dropout at test time.

MC Dropout (Gal & Ghahramani, 2016) approximates Bayesian inference by
keeping Dropout layers active during inference and running T stochastic
forward passes.  The mean of the T predictions is the final output; their
disagreement is a proxy for epistemic uncertainty.

Training is completely unchanged — this module is only used at inference time.

Reference:
    Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning. ICML 2016.
    https://arxiv.org/abs/1506.02142
"""

import warnings

import torch
import torch.nn as nn


def enable_mc_dropout(model: nn.Module) -> None:
    """Set all Dropout* layers to training mode, leaving the rest in eval mode.

    PyTorch's ``model.eval()`` disables dropout by setting every module to
    eval mode.  This function reverses that only for dropout layers, so the
    rest of the network (BatchNorm statistics, etc.) stays deterministic.

    Args:
        model: Any ``nn.Module``.  Only ``Dropout``, ``Dropout2d``, and
            ``Dropout3d`` layers are affected.

    Example::

        model.eval()
        enable_mc_dropout(model)
        with torch.no_grad():
            logits = model(x)   # Dropout is active; BatchNorm uses running stats
    """
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def warn_if_no_dropout(model: nn.Module) -> bool:
    """Emit a ``UserWarning`` if the model has no Dropout layers.

    Without Dropout layers, all T MC samples will be identical and the
    uncertainty map will be uniformly zero — MC Dropout degenerates to
    a standard (deterministic) single-pass inference.

    Args:
        model: ``nn.Module`` to inspect.

    Returns:
        ``True`` if at least one Dropout layer was found, ``False`` otherwise.
    """
    has_dropout = any(
        isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d))
        for m in model.modules()
    )
    if not has_dropout:
        warnings.warn(
            "MCDropout: no Dropout / Dropout2d / Dropout3d layers found in the "
            "model.  All MC samples will be identical and uncertainty will be "
            "zero.  Configure dropout in the model before using MC Dropout "
            "(e.g. set decoder_dropout for SMP models).",
            UserWarning,
            stacklevel=2,
        )
    return has_dropout


def compute_uncertainty(
    samples: torch.Tensor,
    mode: str = "entropy",
) -> torch.Tensor:
    """Compute per-pixel uncertainty from a stack of softmax probability samples.

    Supports two modes:

    **``"entropy"``** — predictive entropy of the mean distribution:

    .. code-block:: text

        p̄(c|x)  = (1/T) Σ_t p_t(c|x)
        H        = -Σ_c p̄_c · log(p̄_c)

    Captures total uncertainty (epistemic + aleatoric).  Range: [0, log C].

    **``"mutual_information"``** — BALD (Bayesian Active Learning by
    Disagreement):

    .. code-block:: text

        MI = H[p̄] - (1/T) Σ_t H[p_t]

    Isolates *epistemic* uncertainty: high only when the T samples *disagree*
    with each other, regardless of how confident each individual sample is.
    Range: [0, log C].

    Args:
        samples: Tensor ``[T, B, C, H, W]`` — softmax probabilities from T
            stochastic forward passes.  Values must be in ``(0, 1]`` and sum
            to 1 along the ``C`` dimension.
        mode: ``"entropy"`` (default) or ``"mutual_information"``.

    Returns:
        Tensor ``[B, 1, H, W]`` — per-pixel uncertainty values.

    Raises:
        ValueError: if ``mode`` is not ``"entropy"`` or ``"mutual_information"``.

    Example YAML (inference processor)::

        inference_processor:
          _target_: ...MCDropoutInferenceProcessor
          n_samples: 10
          uncertainty_mode: entropy   # or mutual_information
          export_uncertainty_map: true
    """
    if mode not in ("entropy", "mutual_information"):
        raise ValueError(
            f"uncertainty_mode must be 'entropy' or 'mutual_information', "
            f"got '{mode}'"
        )

    eps = 1e-10
    p_mean = samples.mean(dim=0)  # [B, C, H, W]

    # Predictive entropy H[p̄(y|x)]
    entropy_mean = -(p_mean * (p_mean + eps).log()).sum(dim=1, keepdim=True)  # [B, 1, H, W]

    if mode == "entropy":
        return entropy_mean

    # Mutual Information: MI = H[p̄] - E_w[H[p(y|x,w)]]
    mean_entropy = (
        -(samples * (samples + eps).log()).sum(dim=2, keepdim=True).mean(dim=0)
    )  # [B, 1, H, W]
    return (entropy_mean - mean_entropy).clamp(min=0.0)
