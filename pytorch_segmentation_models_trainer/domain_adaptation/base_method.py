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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from pytorch_segmentation_models_trainer.model_loader.domain_adaptation_model import DomainAdaptationModel

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class DomainAdaptationLossOutput:
    """Output contract for every domain adaptation method.

    Every ``BaseDomainAdaptationMethod.compute_da_loss`` implementation must
    return one of these. ``DomainAdaptationModel`` uses only ``loss`` and
    ``log_dict``; ``extra`` is available for callbacks or custom logging.

    Args:
        loss: The DA loss scalar to be weighted and added to the segmentation
            loss: ``total = seg_loss + lambda_da * loss``.
        log_dict: Flat dict of ``{metric_name: value}`` pairs that
            ``DomainAdaptationModel`` will log under the ``"da/"`` prefix.
        extra: Optional dict for debugging or callback communication (e.g.
            pseudo-label masks, domain predictions). Not logged automatically.
    """

    loss: Tensor
    log_dict: Dict[str, Any]
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseDomainAdaptationMethod(nn.Module):
    """Abstract base class for all domain adaptation methods.

    This class is a plain ``nn.Module`` (not a ``LightningModule``). It is
    instantiated and owned by ``DomainAdaptationModel``, which is the
    ``LightningModule`` responsible for the training loop. This mirrors the
    pattern used by ``FrameFieldModel`` inside ``FrameFieldSegmentationPLModel``.

    Class-level flags
    -----------------
    requires_features : bool
        Set to ``True`` if ``compute_da_loss`` needs intermediate feature maps
        (e.g. DANN). When ``True``, ``DomainAdaptationModel`` activates a
        ``FeatureExtractorHook`` on the layers listed in
        ``cfg.domain_adaptation.feature_layers`` and passes the captured
        tensors as ``source_features`` / ``target_features``.
        Default: ``False``.

    requires_target_labels : bool
        Set to ``True`` for semi-supervised DA (SSDA) methods that use target
        ground-truth labels during training. For fully unsupervised DA (UDA)
        this must remain ``False``. Default: ``False``.

    Implementing a new method
    -------------------------
    1. Subclass ``BaseDomainAdaptationMethod``.
    2. Override ``compute_da_loss`` — this is the only required method.
    3. Store auxiliary networks (discriminators, etc.) as ``nn.Module``
       attributes so their parameters are included in ``self.parameters()``.
    4. Override lifecycle hooks (``on_fit_start``, ``on_train_epoch_start``,
       ``on_train_epoch_end``) only when needed.
    5. Override ``get_extra_parameter_groups`` if auxiliary networks require
       a different learning rate from the main model.

    Example
    -------
    >>> class MyMethod(BaseDomainAdaptationMethod):
    ...     requires_features = False
    ...
    ...     def compute_da_loss(self, source_batch, target_batch,
    ...                         source_output, target_output,
    ...                         source_features, target_features, **kwargs):
    ...         loss = ...  # your logic
    ...         return DomainAdaptationLossOutput(
    ...             loss=loss,
    ...             log_dict={"my_loss": loss.item()},
    ...         )
    """

    requires_features: bool = False
    requires_target_labels: bool = False

    def __init__(self, **kwargs):
        """Initialize with keyword arguments as provided by Hydra's ``instantiate``.

        When Hydra instantiates a method from a config such as::

            method:
              _target_: my.package.MyMethod
              lambda_da: 1.0
              my_param: 42

        it calls ``MyMethod(lambda_da=1.0, my_param=42)``. All kwargs are
        stored in ``self.cfg`` as an OmegaConf dict for uniform access.

        Subclasses should follow the same pattern::

            class MyMethod(BaseDomainAdaptationMethod):
                def __init__(self, my_param=0.5, **kwargs):
                    super().__init__(**kwargs)
                    self.my_param = my_param
        """
        from omegaconf import OmegaConf

        super().__init__()
        self.cfg = OmegaConf.create(kwargs)
        self.lambda_da: float = float(kwargs.get("lambda_da", 1.0))

    # ------------------------------------------------------------------
    # REQUIRED — must be overridden by every concrete implementation
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
        """Compute the domain adaptation loss for one training step.

        Args:
            source_batch: Full source-domain batch dict (contains ``"image"``
                and ``"mask"`` keys at minimum).
            target_batch: Full target-domain batch dict (contains ``"image"``
                key; ``"mask"`` is only present when
                ``requires_target_labels = True``).
            source_output: Model logits for the source batch,
                shape ``(B, C, H, W)``.
            target_output: Model logits for the target batch,
                shape ``(B, C, H, W)``.
            source_features: Dict mapping layer name → feature tensor for the
                source batch. Empty dict when ``requires_features = False``.
            target_features: Dict mapping layer name → feature tensor for the
                target batch. Empty dict when ``requires_features = False``.
            **kwargs: Reserved for future use.

        Returns:
            ``DomainAdaptationLossOutput`` with at minimum ``loss`` and
            ``log_dict`` populated.

        Raises:
            NotImplementedError: If not overridden by the subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute_da_loss()"
        )

    # ------------------------------------------------------------------
    # OPTIONAL lifecycle hooks — default to no-op
    # ------------------------------------------------------------------

    def on_fit_start(self, pl_module: "DomainAdaptationModel") -> None:
        """Called once when ``trainer.fit()`` begins.

        Args:
            pl_module: The owning ``DomainAdaptationModel`` instance.
        """

    def on_train_epoch_start(
        self, pl_module: "DomainAdaptationModel", epoch: int
    ) -> None:
        """Called at the start of each training epoch.

        Args:
            pl_module: The owning ``DomainAdaptationModel`` instance.
            epoch: Current epoch index (0-based).
        """

    def on_train_batch_start(
        self,
        pl_module: "DomainAdaptationModel",
        batch_idx: int,
        num_batches: int,
    ) -> None:
        """Called at the start of each training batch.

        Override this hook to update per-batch state, such as a lambda
        scheduler running at batch granularity instead of epoch granularity.

        Args:
            pl_module: The owning ``DomainAdaptationModel`` instance.
            batch_idx: Index of the current batch within the epoch (0-based).
            num_batches: Total number of batches per epoch
                (``trainer.num_training_batches``).
        """

    def on_train_epoch_end(
        self, pl_module: "DomainAdaptationModel", epoch: int
    ) -> None:
        """Called at the end of each training epoch.

        Useful for updating pseudo-labels, thresholds, or any epoch-level
        state the method needs to maintain.

        Args:
            pl_module: The owning ``DomainAdaptationModel`` instance.
            epoch: Current epoch index (0-based).
        """

    # ------------------------------------------------------------------
    # OPTIONAL — override for methods with auxiliary networks at a
    # different LR from the main model
    # ------------------------------------------------------------------

    def get_extra_parameter_groups(self) -> List[Dict[str, Any]]:
        """Return extra optimizer parameter groups for auxiliary networks.

        Use this when auxiliary components (e.g. a domain discriminator) need
        a different learning rate from the backbone/decoder. The returned
        groups are appended to the main optimizer's ``param_groups``.

        Returns:
            List of parameter group dicts accepted by ``torch.optim.Optimizer``.
            Each dict must contain at least a ``"params"`` key.

        Example
        -------
        >>> def get_extra_parameter_groups(self):
        ...     return [{"params": self.discriminator.parameters(), "lr": 1e-4}]
        """
        return []
