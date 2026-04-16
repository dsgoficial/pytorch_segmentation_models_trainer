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

import logging
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from pytorch_segmentation_models_trainer.domain_adaptation.base_method import (
    BaseDomainAdaptationMethod,
    DomainAdaptationLossOutput,
)
from pytorch_segmentation_models_trainer.domain_adaptation.methods.gradient_reversal import (
    GradientReversalLayer,
)

logger = logging.getLogger(__name__)


class DomainClassifier(nn.Module):
    """MLP domain classifier that receives spatial feature maps and predicts domain.

    Applies global average pooling to handle arbitrary spatial dimensions, then
    passes through a two-hidden-layer MLP to produce a two-class logit vector
    (source=0, target=1).

    The ``AdaptiveAvgPool2d(1)`` makes the module resolution-agnostic: the same
    classifier works regardless of input image size or encoder architecture.

    To find ``in_channels`` for common SMP encoders:

    * ResNet-18 / ResNet-34: 512
    * ResNet-50 / ResNet-101: 2048
    * EfficientNet-B0: 1280
    * MiT-B2: 512

    You can also inspect the encoder output at runtime::

        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name="resnet34", ...)
        # encoder output channels are model.encoder.out_channels[-1]
        print(model.encoder.out_channels)  # e.g. (3, 64, 64, 128, 256, 512)

    Args:
        in_channels: Number of channels in the feature map produced by the
            hooked encoder layer.
        hidden_size: Width of each hidden fully-connected layer. Default: 1024.
    """

    def __init__(self, in_channels: int, hidden_size: int = 1024) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: Feature map of shape ``(B, C, H, W)``.

        Returns:
            Domain logits of shape ``(B, 2)``.
        """
        return self.classifier(self.pool(x))


class DANNMethod(BaseDomainAdaptationMethod):
    """Domain Adversarial Neural Network (DANN) via Gradient Reversal Layer.

    Implements the unsupervised domain adaptation strategy from:

        Ganin, Y. & Lempitsky, V. "Unsupervised Domain Adaptation by
        Backpropagation." ICML 2015.

    and applied to semantic segmentation as discussed in:

        Vega, P.J.S. et al. "Weakly Supervised Domain Adversarial Neural
        Network for Deforestation Detection in Tropical Forests."
        IEEE JSTARS 2023.

    **Training loop (per step)**:

    1. Source and target features are captured at ``feature_layer`` by
       ``FeatureExtractorHook`` (activated because ``requires_features=True``).
    2. Features from both domains are concatenated and passed through
       ``GradientReversalLayer(lambda_) → DomainClassifier``.
    3. ``CrossEntropyLoss`` between domain logits and domain labels
       (source=0, target=1) produces ``domain_loss``.
    4. ``total_loss = seg_loss + lambda_da * domain_loss``.

    **Gradient flow**:

    * ``seg_loss`` → decoder + encoder (source path only).
    * ``domain_loss`` → ``DomainClassifier`` (normal gradient, gets better at
      classifying domains) AND → ``GRL`` → encoder (reversed gradient, forces
      encoder to produce domain-invariant features).

    **Which parts of the U-Net are updated**:

    * **Encoder** (up to the hooked layer): gradients from *both* losses.
    * **Decoder**: gradients from ``seg_loss`` only (source domain only).
    * **DomainClassifier**: gradients from ``domain_loss`` only.

    Args:
        feature_layer: Dot-separated name of the encoder layer whose output is
            used for domain alignment. Must match an entry in
            ``cfg.domain_adaptation.feature_layers``. Example: ``"encoder"``.
        in_channels: Number of output channels of ``feature_layer``. Used to
            build the :class:`DomainClassifier`. See its docstring for
            per-architecture values.
        hidden_size: Width of each hidden layer in the domain classifier.
            Default: 1024.
        discriminator_lr: Learning rate for the domain classifier parameter
            group. The classifier benefits from a dedicated LR because it
            starts from random weights while the encoder is pretrained.
            Default: ``1e-4``.
        step_mode: Granularity at which the ``lambda_schedule`` is queried to
            update the GRL coefficient. ``"epoch"`` updates once per epoch at
            ``on_train_epoch_start``; ``"batch"`` updates every batch at
            ``on_train_batch_start``, using global batch index as the step
            counter. ``"batch"`` mode produces smoother lambda growth, which
            is closer to the original Ganin et al. implementation.
            Default: ``"epoch"``.
        **kwargs: Forwarded to :class:`BaseDomainAdaptationMethod`
            (e.g. ``lambda_da``, ``lambda_schedule``).

    Example YAML config::

        domain_adaptation:
          feature_layers:
            - encoder          # must match feature_layer below
          method:
            _target_: pytorch_segmentation_models_trainer.domain_adaptation.methods.dann.DANNMethod
            feature_layer: encoder
            in_channels: 512   # resnet34 encoder output channels
            hidden_size: 1024
            discriminator_lr: 1.0e-4
            lambda_da: 1.0
            lambda_schedule:
              _target_: pytorch_segmentation_models_trainer.domain_adaptation.schedulers.DANNScheduler
              gamma: 10.0
    """

    requires_features: bool = True

    def __init__(
        self,
        feature_layer: str,
        in_channels: int,
        hidden_size: int = 1024,
        discriminator_lr: float = 1e-4,
        step_mode: Literal["epoch", "batch"] = "epoch",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if step_mode not in ("epoch", "batch"):
            raise ValueError(f"step_mode must be 'epoch' or 'batch', got {step_mode!r}")
        self.feature_layer = feature_layer
        self.step_mode = step_mode
        self.grl = GradientReversalLayer(lambda_=0.0)
        self.domain_classifier = DomainClassifier(in_channels, hidden_size)
        self._discriminator_lr = discriminator_lr
        self._domain_loss_fn = nn.CrossEntropyLoss()
        # Cached lambda — updated by on_train_epoch_start or on_train_batch_start
        # and read by DomainAdaptationModel._get_lambda_da so both the outer
        # loss weight and the GRL coefficient stay in sync.
        self._current_lambda: float = 0.0

    # ------------------------------------------------------------------
    # Core DA logic
    # ------------------------------------------------------------------

    def compute_da_loss(
        self,
        source_batch: Dict[str, Any],
        target_batch: Dict[str, Any],
        source_output: Tensor,
        target_output: Tensor,
        source_features: Dict[str, Tensor],
        target_features: Dict[str, Tensor],
        **kwargs,
    ) -> DomainAdaptationLossOutput:
        """Compute DANN domain alignment loss.

        Args:
            source_batch: Source-domain batch dict (unused here; present for
                interface compatibility).
            target_batch: Target-domain batch dict (unused here).
            source_output: Segmentation logits for the source batch.
            target_output: Segmentation logits for the target batch.
            source_features: Dict mapping layer name → feature tensor for
                source images. Must contain ``self.feature_layer``.
            target_features: Dict mapping layer name → feature tensor for
                target images. Must contain ``self.feature_layer``.

        Returns:
            :class:`DomainAdaptationLossOutput` with:
            - ``loss``: scalar domain alignment loss (differentiable).
            - ``log_dict``: ``{"domain_loss": float, "domain_acc": float}``.
        """
        src_feat = source_features[self.feature_layer]  # (B_s, C, H, W)
        tgt_feat = target_features[self.feature_layer]  # (B_t, C, H, W)

        B_src = src_feat.shape[0]
        B_tgt = tgt_feat.shape[0]

        features = torch.cat([src_feat, tgt_feat], dim=0)  # (B_s+B_t, C, H, W)
        domain_labels = torch.cat(
            [
                torch.zeros(B_src, dtype=torch.long, device=src_feat.device),
                torch.ones(B_tgt, dtype=torch.long, device=tgt_feat.device),
            ]
        )

        reversed_features = self.grl(features)
        domain_logits = self.domain_classifier(reversed_features)
        domain_loss = self._domain_loss_fn(domain_logits, domain_labels)

        with torch.no_grad():
            domain_acc = (domain_logits.argmax(dim=1) == domain_labels).float().mean()

        return DomainAdaptationLossOutput(
            loss=domain_loss,
            log_dict={
                "domain_loss": domain_loss.item(),
                "domain_acc": domain_acc.item(),
            },
        )

    # ------------------------------------------------------------------
    # Internal lambda update
    # ------------------------------------------------------------------

    def _update_lambda(self, step: int, total_steps: int) -> None:
        """Compute and apply the current lambda from the schedule.

        Updates both ``self._current_lambda`` (used by
        ``DomainAdaptationModel._get_lambda_da``) and ``self.grl.lambda_``
        (used during the forward pass for gradient reversal).

        Args:
            step: Current step index (epoch index or global batch index).
            total_steps: Total steps (``max_epochs`` or total batches).
        """
        if hasattr(self, "lambda_schedule"):
            lam = self.lambda_schedule.get_lambda(step, total_steps)
        else:
            lam = self.lambda_da
        self._current_lambda = lam
        self.grl.set_lambda(lam)
        logger.debug(
            "DANNMethod: step %d/%d — GRL lambda set to %.4f", step, total_steps, lam
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_epoch_start(
        self, pl_module: "DomainAdaptationModel", epoch: int
    ) -> None:
        """Update the GRL lambda at the start of each epoch (epoch mode only).

        In ``step_mode="batch"`` this hook is a no-op; the update happens in
        :meth:`on_train_batch_start` instead.

        Args:
            pl_module: The owning ``DomainAdaptationModel`` instance.
            epoch: Current epoch index (0-based).
        """
        if self.step_mode != "epoch":
            return
        max_epochs = (
            getattr(pl_module.trainer, "max_epochs", 1) if pl_module.trainer else 1
        )
        self._update_lambda(epoch, max_epochs)

    def on_train_batch_start(
        self,
        pl_module: "DomainAdaptationModel",
        batch_idx: int,
        num_batches: int,
    ) -> None:
        """Update the GRL lambda at the start of each batch (batch mode only).

        In ``step_mode="epoch"`` this hook is a no-op.

        The global batch index is computed as
        ``epoch * num_batches + batch_idx``, and the total steps as
        ``max_epochs * num_batches``, so the scheduler sees a uniformly
        increasing progress counter across the full training run.

        Args:
            pl_module: The owning ``DomainAdaptationModel`` instance.
            batch_idx: Index of the current batch within the epoch (0-based).
            num_batches: Total batches per epoch.
        """
        if self.step_mode != "batch":
            return
        max_epochs = (
            getattr(pl_module.trainer, "max_epochs", 1) if pl_module.trainer else 1
        )
        epoch = pl_module.current_epoch
        global_step = epoch * num_batches + batch_idx
        total_steps = max_epochs * num_batches
        self._update_lambda(global_step, total_steps)

    # ------------------------------------------------------------------
    # Optimizer integration
    # ------------------------------------------------------------------

    def get_extra_parameter_groups(self) -> List[Dict[str, Any]]:
        """Return a dedicated parameter group for the domain classifier.

        The domain classifier starts from random weights while the encoder is
        pretrained, so a separate (typically higher) learning rate is useful.

        Returns:
            List with one parameter group dict for
            ``self.domain_classifier.parameters()``.
        """
        return [
            {
                "params": list(self.domain_classifier.parameters()),
                "lr": self._discriminator_lr,
            }
        ]
