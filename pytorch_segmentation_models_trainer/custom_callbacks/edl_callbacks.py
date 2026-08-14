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

PyTorch Lightning callbacks for Evidential Deep Learning training.

EvidentialWarmupCallback
    Controls encoder freezing during fine-tuning.  When training from
    scratch set ``freeze_encoder=False`` and only KL annealing (managed by
    the CompoundLoss weight schedule) acts as warm-up.

EvidentialUncertaintyVisualizationCallback
    Logs a 4-column grid image to the trainer logger every N epochs:
    [original image | predicted class | uncertainty map | ground truth]
"""

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytorch_segmentation_models_trainer.custom_models.edl_wrapper import (
    EvidentialWrapper,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Warmup / encoder-freeze callback
# ---------------------------------------------------------------------------


class EvidentialWarmupCallback(pl.callbacks.Callback):
    """Encoder freeze + optional partial unfreeze schedule for EDL.

    For **fine-tuning** (``freeze_encoder=True``):
        * Epoch < warmup_epochs        → encoder frozen, KL weight is 0 (via CompoundLoss schedule)
        * warmup_epochs <= epoch < partial_unfreeze_epoch → last two encoder stages unfrozen
        * epoch >= partial_unfreeze_epoch → encoder fully unfrozen

    For **training from scratch** (``freeze_encoder=False``, the default):
        * Encoder is never frozen; KL annealing is the only warm-up mechanism
          (controlled by the CompoundLoss weight schedule in the YAML).
        * The callback is still useful to log the current training phase.

    Args:
        warmup_epochs:         Number of epochs with encoder fully frozen.
                               Ignored when ``freeze_encoder=False``.
        freeze_encoder:        Whether to freeze the encoder at startup.
                               Set ``False`` when training from scratch.
        partial_unfreeze_epoch: Epoch at which the last two encoder stages
                               are unfrozen (fine-tuning only).
    """

    def __init__(
        self,
        warmup_epochs: int = 5,
        freeze_encoder: bool = False,
        partial_unfreeze_epoch: int = 10,
    ) -> None:
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.freeze_encoder = freeze_encoder
        self.partial_unfreeze_epoch = partial_unfreeze_epoch
        self._phase: str = "init"

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.freeze_encoder:
            logger.info(
                "EvidentialWarmupCallback: freeze_encoder=False — "
                "training from scratch, encoder will NOT be frozen. "
                "KL annealing is controlled by the CompoundLoss schedule."
            )
            return

        if trainer.current_epoch < self.warmup_epochs:
            logger.info(
                "EvidentialWarmupCallback: freezing encoder for %d warm-up epochs.",
                self.warmup_epochs,
            )
            self._set_encoder_trainable(pl_module, trainable=False)
            self._phase = "warmup"
        else:
            self._phase = "full"

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not self.freeze_encoder:
            return

        epoch = trainer.current_epoch

        if epoch == self.warmup_epochs and self._phase == "warmup":
            logger.info(
                "EvidentialWarmupCallback: warm-up complete at epoch %d. "
                "Partially unfreezing encoder (last 2 stages).",
                epoch,
            )
            self._partial_unfreeze_encoder(pl_module)
            self._phase = "partial"

        elif epoch == self.partial_unfreeze_epoch and self._phase in (
            "warmup",
            "partial",
        ):
            logger.info(
                "EvidentialWarmupCallback: fully unfreezing encoder at epoch %d.",
                epoch,
            )
            self._set_encoder_trainable(pl_module, trainable=True)
            self._phase = "full"

    # ------------------------------------------------------------------
    # Encoder manipulation helpers
    # ------------------------------------------------------------------

    def _set_encoder_trainable(
        self, pl_module: pl.LightningModule, trainable: bool
    ) -> None:
        """Freeze / unfreeze all encoder parameters."""
        model = pl_module.model
        if isinstance(model, EvidentialWrapper):
            if trainable:
                model.unfreeze_encoder()
            else:
                model._freeze_encoder()
        else:
            # Fall back to Model.set_encoder_trainable
            if hasattr(pl_module, "set_encoder_trainable"):
                pl_module.set_encoder_trainable(trainable=trainable)
            else:
                logger.warning(
                    "EvidentialWarmupCallback: could not freeze/unfreeze encoder — "
                    "neither EvidentialWrapper nor set_encoder_trainable found."
                )

    def _partial_unfreeze_encoder(self, pl_module: pl.LightningModule) -> None:
        """Unfreeze the last ``n_stages`` stages of the encoder."""
        n_stages = 2
        model = pl_module.model
        encoder = None

        if isinstance(model, EvidentialWrapper):
            encoder = model._find_encoder()
        elif hasattr(pl_module, "model"):
            for attr in ("encoder", "backbone"):
                encoder = getattr(pl_module.model, attr, None)
                if encoder is not None:
                    break

        if encoder is None:
            logger.warning(
                "EvidentialWarmupCallback: could not find encoder for partial unfreeze."
            )
            return

        # Get child modules that look like stages/layers
        children = list(encoder.children())
        if len(children) == 0:
            # Leaf module — unfreeze all
            for param in encoder.parameters():
                param.requires_grad = True
            return

        # Unfreeze last n_stages children
        for child in children[-n_stages:]:
            for param in child.parameters():
                param.requires_grad = True

        unfrozen = sum(
            p.numel() for child in children[-n_stages:] for p in child.parameters()
        )
        logger.info(
            "EvidentialWarmupCallback: unfroze last %d encoder stages "
            "(%d parameters).",
            n_stages,
            unfrozen,
        )


# ---------------------------------------------------------------------------
# Uncertainty visualisation callback
# ---------------------------------------------------------------------------


class EvidentialUncertaintyVisualizationCallback(pl.callbacks.Callback):
    """Logs a 4-column diagnostic grid every N validation epochs.

    Columns: ``[Input image | Predicted class | Uncertainty map | Ground truth]``

    Works with any logger that has an ``experiment`` with an
    ``add_image()`` method (TensorBoard) or a ``log_image()`` method
    (WandB / MLflow / CometML via Lightning).

    Args:
        num_images:        How many samples from the validation batch to log.
        log_every_n_epochs: Log only every this many epochs.
        norm_params:       Dict passed to ``batch_denormalize_tensor`` for
                           de-normalising the input image before display.
                           Pass ``None`` to skip de-normalisation.
    """

    def __init__(
        self,
        num_images: int = 4,
        log_every_n_epochs: int = 5,
        norm_params: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.num_images = num_images
        self.log_every_n_epochs = log_every_n_epochs
        self.norm_params = norm_params or {}
        self._val_batch: Optional[dict] = None

    # ------------------------------------------------------------------
    # Cache one validation batch
    # ------------------------------------------------------------------

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0 and self._val_batch is None:
            self._val_batch = batch

    # ------------------------------------------------------------------
    # Main hook
    # ------------------------------------------------------------------

    @rank_zero_only
    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch = trainer.current_epoch
        if epoch % self.log_every_n_epochs != 0:
            return
        if self._val_batch is None:
            return
        if not isinstance(pl_module.model, EvidentialWrapper):
            return

        pl_module.eval()
        with torch.no_grad():
            images = self._val_batch["image"].to(pl_module.device)
            masks = self._val_batch["mask"].to(pl_module.device)
            output = pl_module.model(images)

        n = min(self.num_images, images.shape[0])
        fig = self._make_figure(images[:n], output, masks[:n], n)

        # Try to log to available logger
        self._log_figure(trainer, fig, epoch)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _make_figure(self, images, output, masks, n: int):
        probs = output["probs"]  # [B, K, H, W]
        uncertainty = output["uncertainty"]  # [B, 1, H, W]

        fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        col_titles = ["Input", "Predicted class", "Uncertainty (K/S)", "Ground truth"]
        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title, fontsize=10)

        uncertainty_cmap = matplotlib.colormaps["plasma"]

        for i in range(n):
            # --- Input image ---
            img = images[i].cpu().float()
            img = self._denormalize(img)
            if img.shape[0] == 1:
                axes[i, 0].imshow(img.squeeze(0), cmap="gray")
            elif img.shape[0] >= 3:
                axes[i, 0].imshow(img[:3].permute(1, 2, 0).clamp(0, 1))
            else:
                axes[i, 0].imshow(img[0], cmap="gray")

            # --- Predicted class (argmax of probs) ---
            pred_cls = probs[i].argmax(dim=0).cpu().numpy()
            axes[i, 1].imshow(pred_cls, cmap="tab20", interpolation="nearest")

            # --- Uncertainty map ---
            unc = uncertainty[i, 0].cpu().float().numpy()
            im = axes[i, 2].imshow(unc, cmap=uncertainty_cmap, vmin=0.0, vmax=1.0)
            fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

            # --- Ground truth ---
            gt = masks[i]
            if gt.dim() == 3 and gt.is_floating_point():
                gt = gt.argmax(dim=0)
            elif gt.dim() == 3:
                gt = gt.squeeze(0)
            axes[i, 3].imshow(gt.cpu().numpy(), cmap="tab20", interpolation="nearest")

            for j in range(4):
                axes[i, j].axis("off")

        fig.tight_layout()
        return fig

    def _denormalize(self, img: torch.Tensor) -> torch.Tensor:
        """Reverse ImageNet-style normalisation if norm_params provided."""
        if not self.norm_params:
            return img
        mean = self.norm_params.get("mean", [0.0] * img.shape[0])
        std = self.norm_params.get("std", [1.0] * img.shape[0])
        mean_t = torch.tensor(mean, dtype=img.dtype).view(-1, 1, 1)
        std_t = torch.tensor(std, dtype=img.dtype).view(-1, 1, 1)
        return img * std_t + mean_t

    def _log_figure(self, trainer: pl.Trainer, fig, epoch: int) -> None:
        tag = f"edl/uncertainty_epoch_{epoch:04d}"

        # Convert figure to numpy array
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(h, w, 3)
        img_tensor = torch.from_numpy(buf).permute(2, 0, 1)  # [3, H, W]

        # TensorBoard
        if hasattr(trainer.logger, "experiment") and hasattr(
            trainer.logger.experiment, "add_image"
        ):
            trainer.logger.experiment.add_image(tag, img_tensor, epoch)
            return

        # WandB / generic Lightning logger
        if trainer.logger is not None:
            try:
                import wandb

                trainer.logger.experiment.log({tag: wandb.Image(fig)}, step=epoch)
                return
            except (ImportError, AttributeError):
                pass

        # Fallback: save to log_dir
        if trainer.log_dir:
            out_dir = Path(trainer.log_dir) / "edl_uncertainty"
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                out_dir / f"epoch_{epoch:04d}.png", bbox_inches="tight", dpi=100
            )
            logger.info(
                "EvidentialUncertaintyVisualizationCallback: saved to %s", out_dir
            )
