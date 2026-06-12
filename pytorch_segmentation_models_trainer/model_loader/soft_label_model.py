# -*- coding: utf-8 -*-
"""LightningModule subclass that supports dict-mask batches for soft-label training."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from pytorch_segmentation_models_trainer.model_loader.model import Model

logger = logging.getLogger(__name__)


class SoftLabelModel(Model):
    """Model variant that natively handles ``SoftLabelDataset`` batches.

    ``SoftLabelDataset`` returns ``batch["mask"]`` as a dict::

        {"mask": p_soft_tensor (B, C, H, W),
         "w_conf": w_conf_tensor (B, 1, H, W)}

    The base ``Model._shared_step`` expects ``masks`` to be a plain tensor and
    calls ``masks.is_floating_point()`` on it, which would crash for a dict.

    This subclass intercepts the batch in ``_shared_step``, unwraps the dict
    (so the parent's soft-label detection works unchanged), stores ``w_conf``
    as ``self._w_conf``, and then re-assembles the dict in ``_compute_loss``
    so ``SoftLabelWeightedCELoss`` receives both tensors via the existing
    FrameField compound-loss path (``isinstance(masks, dict)``).

    All other Model behaviour — metrics, logging, optimisers, LR schedulers,
    dual-head, OHEM, EDL, MoE — is inherited without modification.

    Args:
        cfg: Hydra DictConfig (same as ``Model``).
        inference_mode: When ``True``, skips dataset and loss instantiation.

    Example YAML:
        _target_: pytorch_segmentation_models_trainer.model_loader.soft_label_model.SoftLabelModel
    """

    def _shared_step(self, batch: Dict[str, Any], prefix: str):
        """Unwrap dict-mask before delegating to the parent shared step.

        When ``batch["mask"]`` is a dict (``SoftLabelDataset`` format), this
        method extracts ``w_conf`` into ``self._w_conf`` and replaces
        ``batch["mask"]`` with the flat ``p_soft`` tensor so the parent's
        ``masks.is_floating_point()`` check works unchanged.
        """
        self._w_conf: Optional[torch.Tensor] = None

        mask_val = batch.get(self.cfg.get("mask_key", "mask"))
        if isinstance(mask_val, dict):
            self._w_conf = mask_val.get("w_conf")
            batch = {**batch, self.cfg.get("mask_key", "mask"): mask_val.get("mask")}

        return super()._shared_step(batch, prefix)

    def test_step(self, batch, batch_idx):
        """Unwrap dict-mask before delegating to the parent test_step."""
        self._w_conf: Optional[torch.Tensor] = None

        mask_val = batch.get(self.cfg.get("mask_key", "mask"))
        if isinstance(mask_val, dict):
            self._w_conf = mask_val.get("w_conf")
            batch = {**batch, self.cfg.get("mask_key", "mask"): mask_val.get("mask")}

        return super().test_step(batch, batch_idx)

    def _compute_loss(
        self,
        predicted_masks: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        """Re-wrap masks as a dict when w_conf is available.

        When ``self._w_conf`` was set by ``_shared_step``, the flat ``masks``
        tensor (P_soft) is wrapped back into a dict so that the parent's
        FrameField compound-loss path (``isinstance(masks, dict)``) forwards
        both ``mask`` and ``w_conf`` to ``SoftLabelWeightedCELoss.compute``.
        """
        if self._w_conf is not None:
            masks = {"mask": masks, "w_conf": self._w_conf}
        return super()._compute_loss(predicted_masks, masks)
