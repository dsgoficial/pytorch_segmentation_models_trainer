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

import logging
from typing import List, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DomainAdaptationMonitorCallback(pl.callbacks.Callback):
    """Monitors two key risks of domain adaptation training.

    **Risk 1 — Catastrophic forgetting on source domain**:
    Measures segmentation IoU on the source validation set every
    ``log_every_n_epochs`` epochs and emits a warning when the drop from
    the epoch-0 baseline exceeds ``forgetting_threshold``.

    **Risk 2 — Adaptation progress on target domain**:
    Measures segmentation IoU on the target validation set so you can see
    whether the model is actually learning in the target domain.

    The following scalars are logged to TensorBoard / W&B:

    * ``iou/source_val`` — source domain mean IoU (should stay stable)
    * ``iou/target_val`` — target domain mean IoU (should rise)
    * ``iou/gap_source_minus_target`` — convergence indicator (should shrink)
    * ``iou/source_drop_from_baseline`` — forgetting indicator (should stay ~0)

    This callback requires that ``DomainAdaptationModel`` exposes:

    * ``pl_module.source_val_ds`` (optional ``Dataset``)
    * ``pl_module.target_val_ds`` (optional ``Dataset``)

    If a dataset is ``None`` the corresponding metric is silently skipped.

    Args:
        num_classes: Number of segmentation classes.
        class_names: Optional list of class name strings for logging.
            Defaults to ``["Class 0", "Class 1", ...]``.
        log_every_n_epochs: Evaluation frequency. Default: ``1`` (every epoch).
        forgetting_threshold: Fraction-of-IoU drop from baseline that triggers
            a warning. Default: ``0.05`` (5 percentage points).
        eval_batch_size: Batch size used for evaluation forward passes.
            Defaults to ``8``.
        eval_num_workers: Number of DataLoader workers for evaluation.
            Default: ``0`` (main process, avoids spawning issues).
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        log_every_n_epochs: int = 1,
        forgetting_threshold: float = 0.05,
        eval_batch_size: int = 8,
        eval_num_workers: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        self.log_every_n_epochs = log_every_n_epochs
        self.forgetting_threshold = forgetting_threshold
        self.eval_batch_size = eval_batch_size
        self.eval_num_workers = eval_num_workers

        self._source_baseline_iou: Optional[float] = None

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Compute and store the source-domain IoU baseline at epoch 0."""
        source_ds = getattr(pl_module, "source_val_ds", None)
        if source_ds is None:
            logger.warning(
                "DomainAdaptationMonitorCallback: pl_module.source_val_ds is None. "
                "Catastrophic forgetting monitoring is disabled."
            )
            return

        iou = self._compute_mean_iou(pl_module, source_ds)
        self._source_baseline_iou = iou
        logger.info("DomainAdaptationMonitorCallback: source IoU baseline = %.4f", iou)
        if trainer.logger:
            trainer.logger.experiment.add_scalar(
                "iou/source_baseline", iou, global_step=0
            )

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Evaluate both domains and log monitoring metrics."""
        epoch = trainer.current_epoch
        if epoch % self.log_every_n_epochs != 0:
            return

        source_ds = getattr(pl_module, "source_val_ds", None)
        target_ds = getattr(pl_module, "target_val_ds", None)

        source_iou: Optional[float] = None
        target_iou: Optional[float] = None

        if source_ds is not None:
            source_iou = self._compute_mean_iou(pl_module, source_ds)
            pl_module.log("iou/source_val", source_iou, on_epoch=True, prog_bar=False)

            if self._source_baseline_iou is not None:
                drop = self._source_baseline_iou - source_iou
                pl_module.log(
                    "iou/source_drop_from_baseline",
                    drop,
                    on_epoch=True,
                    prog_bar=False,
                )
                if drop > self.forgetting_threshold:
                    logger.warning(
                        "[DA Monitor] Epoch %d: source IoU dropped %.3f since baseline "
                        "(%.3f → %.3f). Possible catastrophic forgetting.",
                        epoch,
                        drop,
                        self._source_baseline_iou,
                        source_iou,
                    )

        if target_ds is not None:
            target_iou = self._compute_mean_iou(pl_module, target_ds)
            pl_module.log("iou/target_val", target_iou, on_epoch=True, prog_bar=True)

        if source_iou is not None and target_iou is not None:
            gap = source_iou - target_iou
            pl_module.log(
                "iou/gap_source_minus_target", gap, on_epoch=True, prog_bar=False
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_mean_iou(self, pl_module: pl.LightningModule, dataset) -> float:
        """Compute mean intersection-over-union on the given dataset.

        Args:
            pl_module: The LightningModule (used for forward pass and device).
            dataset: A torch Dataset whose items are dicts with ``"image"``
                and ``"mask"`` keys.

        Returns:
            Mean IoU across all classes and samples as a Python float.
        """
        loader = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.eval_num_workers,
            pin_memory=False,
        )

        intersection = torch.zeros(self.num_classes, device=pl_module.device)
        union = torch.zeros(self.num_classes, device=pl_module.device)

        pl_module.model.eval()
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(pl_module.device)
                masks = batch["mask"].to(pl_module.device).long()
                if masks.dim() == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)

                logits = pl_module.model(images)
                preds = torch.argmax(logits, dim=1)

                for cls in range(self.num_classes):
                    pred_cls = preds == cls
                    true_cls = masks == cls
                    intersection[cls] += (pred_cls & true_cls).sum()
                    union[cls] += (pred_cls | true_cls).sum()

        iou_per_class = intersection / (union + 1e-6)
        return iou_per_class.mean().item()
