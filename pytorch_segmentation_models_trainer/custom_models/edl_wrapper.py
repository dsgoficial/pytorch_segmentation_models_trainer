# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-04-15
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

Generic Evidential Deep Learning wrapper for segmentation models.

Wraps any model whose forward() returns a logit tensor [B, K, H, W] and
adds the Dirichlet parameterisation:

    evidence  = Softplus(logits)        # e >= 0
    alpha     = evidence + 1            # α_k >= 1
    S         = sum_k(alpha_k)          # total evidence strength
    probs     = alpha / S               # expected class probabilities
    uncertainty = K / S                 # epistemic uncertainty ∈ (0, 1]

The wrapper is architecture-agnostic: it works with any SMP model, HuggingFace
model, timm model, or custom model that produces [B, K, H, W] tensors.

Usage (Hydra config)::

    model:
      _target_: pytorch_segmentation_models_trainer.custom_models.edl_wrapper.EvidentialWrapper
      freeze_encoder: true   # set false when training from scratch
      model:
        _target_: segmentation_models_pytorch.Unet
        encoder_name: resnet50
        encoder_weights: imagenet
        in_channels: 3
        classes: 5
"""

import logging
from typing import Dict

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def _instantiate_if_config(model) -> nn.Module:
    """Instantiate model from a Hydra DictConfig if needed.

    When ``Model.get_model()`` calls ``instantiate(cfg.model, _recursive_=False)``,
    nested sub-configs (like the inner ``model`` field of ``EvidentialWrapper``)
    arrive as ``DictConfig`` objects rather than instantiated modules.
    This helper detects that case and performs the instantiation.
    """
    if isinstance(model, nn.Module):
        return model
    try:
        from hydra.utils import instantiate as hydra_instantiate

        return hydra_instantiate(model, _recursive_=True)
    except Exception as exc:
        raise TypeError(
            f"EvidentialWrapper.model must be a nn.Module or a Hydra-compatible "
            f"config dict; got {type(model)}. "
            f"Inner error: {exc}"
        ) from exc


class EvidentialWrapper(nn.Module):
    """Wraps any segmentation model to produce Dirichlet evidence outputs.

    The wrapper does **not** modify the wrapped model's weights — it only
    changes the interpretation of its output by applying Softplus instead of
    Softmax.  Pre-trained weights remain usable without re-initialisation.

    Args:
        model: A ``torch.nn.Module`` that returns logits ``[B, K, H, W]``.
        freeze_encoder: When ``True`` the encoder/backbone parameters are
            frozen at construction time.  Intended for fine-tuning scenarios;
            set to ``False`` when training from scratch.

    Forward returns a dict with keys:
        ``"logits"``      — raw model output ``[B, K, H, W]``
        ``"evidence"``    — ``Softplus(logits)`` ``[B, K, H, W]``
        ``"alpha"``       — ``evidence + 1`` ``[B, K, H, W]``
        ``"probs"``       — expected probabilities ``[B, K, H, W]``
        ``"uncertainty"`` — epistemic uncertainty ``K/S`` ``[B, 1, H, W]``
    """

    def __init__(self, model, freeze_encoder: bool = False):
        super().__init__()
        self.model = _instantiate_if_config(model)
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            self._freeze_encoder()

    # ------------------------------------------------------------------
    # Encoder freeze helpers
    # ------------------------------------------------------------------

    def _freeze_encoder(self):
        """Freeze the encoder/backbone parameters."""
        encoder = self._find_encoder()
        if encoder is not None:
            for param in encoder.parameters():
                param.requires_grad = False
            logger.info(
                "EvidentialWrapper: froze encoder (%s parameters frozen).",
                sum(p.numel() for p in encoder.parameters()),
            )
        else:
            logger.warning(
                "EvidentialWrapper: freeze_encoder=True but could not locate "
                "encoder attribute. No parameters were frozen."
            )

    def unfreeze_encoder(self):
        """Unfreeze the encoder/backbone parameters (called by warm-up callback)."""
        encoder = self._find_encoder()
        if encoder is not None:
            for param in encoder.parameters():
                param.requires_grad = True
            logger.info("EvidentialWrapper: encoder unfrozen.")

    def _find_encoder(self) -> nn.Module | None:
        """Return the encoder/backbone sub-module, or None if not found."""
        backbone_attrs = ("encoder", "backbone")
        # Check direct attributes and one level of nesting
        candidates = [self.model]
        inner = getattr(self.model, "model", None)
        if inner is not None:
            candidates.append(inner)

        for obj in candidates:
            for attr in backbone_attrs:
                enc = getattr(obj, attr, None)
                if enc is not None:
                    return enc
        return None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Run the wrapped model and compute Dirichlet parameterisation.

        Args:
            x: Input image tensor ``[B, C, H, W]``.

        Returns:
            Dict with ``logits``, ``evidence``, ``alpha``, ``probs``,
            ``uncertainty`` tensors.
        """
        raw = self.model(x)

        # Handle models that return tuples/lists (take first element)
        if isinstance(raw, (tuple, list)):
            logits = raw[0]
        elif isinstance(raw, dict):
            logits = raw.get("out", raw.get("seg", next(iter(raw.values()))))
        else:
            logits = raw

        evidence = F.softplus(logits)  # [B, K, H, W], e >= 0
        alpha = evidence + 1.0  # [B, K, H, W], α >= 1
        S = alpha.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        probs = alpha / S  # [B, K, H, W]
        uncertainty = alpha.shape[1] / S  # [B, 1, H, W], u ∈ (0,1]

        return {
            "logits": logits,
            "evidence": evidence,
            "alpha": alpha,
            "probs": probs,
            "uncertainty": uncertainty,
        }

    # ------------------------------------------------------------------
    # Compatibility helpers (used by Model.set_encoder_trainable)
    # ------------------------------------------------------------------

    @property
    def encoder(self):
        """Expose encoder for compatibility with Model.set_encoder_trainable."""
        return self._find_encoder()

    def __repr__(self):
        return (
            f"EvidentialWrapper(\n"
            f"  freeze_encoder={self.freeze_encoder},\n"
            f"  model={self.model.__class__.__name__}\n"
            f")"
        )


def is_evidential_model(model_or_cfg) -> bool:
    """Return True if a model instance or Hydra config targets EvidentialWrapper.

    Args:
        model_or_cfg: Either a ``torch.nn.Module`` instance or a Hydra
            DictConfig / OmegaConf object with a ``_target_`` field.

    Returns:
        ``True`` when the model is (or will be) an EvidentialWrapper.
    """
    if isinstance(model_or_cfg, EvidentialWrapper):
        return True
    # Handle Hydra DictConfig
    target = getattr(model_or_cfg, "_target_", None)
    if target is None and isinstance(model_or_cfg, dict):
        target = model_or_cfg.get("_target_", None)
    if target is not None:
        return "EvidentialWrapper" in str(target)
    return False
