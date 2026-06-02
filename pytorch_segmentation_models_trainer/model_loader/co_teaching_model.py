# -*- coding: utf-8 -*-
"""Dual-branch co-teaching model for curriculum-guided noisy weak supervision.

Implements the symmetric co-teaching framework (Xiao et al. 2026, JSTARS,
eq. 8-11): two networks fθA and fθB with independent initialisation are
trained with cross-update — each network is optimized on samples that the
other network considers low-loss (class-balanced selection).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from hydra.utils import instantiate

from pytorch_segmentation_models_trainer.custom_losses.co_teaching_loss import (
    CoTeachingLoss,
    CurriculumScheduler,
)
from pytorch_segmentation_models_trainer.model_loader.soft_label_model import (
    SoftLabelModel,
)

logger = logging.getLogger(__name__)


class CoTeachingModel(SoftLabelModel):
    """Dual-branch co-teaching model for soft-label training under label noise.

    Maintains two independently-initialized networks (``self.model`` = fθA and
    ``self.model_b`` = fθB) and performs cross-update optimization with manual
    gradient steps.

    Training follows eq. 11 from the paper:

        L_total = Σ_{i∈J_B} W_conf^(i) · L_CE^A(x_i)
                + Σ_{j∈J_A} W_conf^(j) · L_CE^B(x_j)
                + λ · L_reg

    Validation and inference use only ``self.model`` (fθA).

    Requires ``cfg.loss`` to point to a ``CoTeachingLoss`` configuration.
    Two optimizers are created — one per branch — using ``cfg.optimizer``.
    A ``cfg.curriculum`` sub-config (optional) overrides scheduler defaults.

    Args:
        cfg: Hydra DictConfig (same format as ``Model``).
        inference_mode: When ``True``, skips dataset and loss instantiation.

    Example YAML:
        _target_: pytorch_segmentation_models_trainer.model_loader.co_teaching_model.CoTeachingModel

        curriculum:
          e_warm: 20
          p_start: 0.5
          p_end: 0.95
    """

    def __init__(self, cfg, inference_mode=False) -> None:
        super().__init__(cfg, inference_mode)
        # Second backbone: independently initialized copy of the same architecture
        self.model_b = self.get_model()
        self.automatic_optimization = False

        # Curriculum scheduler (may override loss defaults via cfg.curriculum)
        curr_cfg = cfg.get("curriculum", {}) if hasattr(cfg, "get") else {}
        if isinstance(curr_cfg, dict):
            e_warm = curr_cfg.get("e_warm", 20)
            p_start = curr_cfg.get("p_start", 0.5)
            p_end = curr_cfg.get("p_end", 0.95)
        else:
            e_warm = getattr(curr_cfg, "e_warm", 20)
            p_start = getattr(curr_cfg, "p_start", 0.5)
            p_end = getattr(curr_cfg, "p_end", 0.95)

        self._curriculum = CurriculumScheduler(
            e_warm=e_warm, p_start=p_start, p_end=p_end
        )
        if hasattr(self, "loss_function") and isinstance(
            self.loss_function, CoTeachingLoss
        ):
            self.loss_function._curriculum = self._curriculum

    # ------------------------------------------------------------------
    # forward — always uses model_a (fθA), for val/test compatibility
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run fθA (model_a) on images; used by validation/test steps."""
        return self.model(images)

    # ------------------------------------------------------------------
    # configure_optimizers — two independent optimizers, one per branch
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Return two optimizers (fθA, fθB) and an empty scheduler list.

        Both optimizers share the same hyper-parameters from ``cfg.optimizer``
        but govern disjoint parameter sets (model_a vs model_b).
        """
        opt_a = instantiate(
            self.cfg.optimizer, params=self.model.parameters(), _recursive_=False
        )
        opt_b = instantiate(
            self.cfg.optimizer, params=self.model_b.parameters(), _recursive_=False
        )
        return [opt_a, opt_b], []

    # ------------------------------------------------------------------
    # training_step — manual co-teaching cross-update
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Perform one co-teaching training step with manual optimization.

        Steps:
        1. Unwrap dict-mask batch (SoftLabelDataset format).
        2. Forward pass through both fθA and fθB.
        3. Compute curriculum mask and class-balanced selection sets J_A, J_B.
        4. Compute cross-update losses: loss_A on J_B samples, loss_B on J_A.
        5. Optionally add L_reg to loss_A.
        6. Manual backward + step for each optimizer independently.

        Args:
            batch: Dict with ``"image"``, ``"mask"`` (plain or dict), optional ``"w_conf"``.
            batch_idx: Batch index (unused).

        Returns:
            Scalar total loss (for Lightning's progress bar logging).
        """
        opt_a, opt_b = self.optimizers()

        # --- unwrap SoftLabelDataset dict mask ---
        self._w_conf: Optional[torch.Tensor] = None
        mask_key = self.cfg.get("mask_key", "mask")
        mask_val = batch.get(mask_key)
        if isinstance(mask_val, dict):
            self._w_conf = mask_val.get("w_conf")
            batch = {**batch, mask_key: mask_val.get("mask")}

        image_key = self.cfg.get("image_key", "image")
        images = batch[image_key]
        p_soft = batch.get(mask_key)

        w_conf = self._w_conf

        # --- sync current_epoch into loss function ---
        self.loss_function.current_epoch = self.current_epoch

        epoch = self.current_epoch
        keep_rate = 1.0 - self._curriculum.forget_rate(epoch)

        # --- curriculum mask from W_conf ---
        if w_conf is not None:
            m_curr = self._curriculum.curriculum_mask(w_conf, epoch)
        else:
            B, _, H, W = images.shape
            m_curr = torch.ones(B, 1, H, W, device=images.device)

        # --- forward passes ---
        logits_a = self.model(images)
        logits_b = self.model_b(images)

        # --- per-pixel CE (live graphs) ---
        ce_a = self.loss_function._masked_ce_per_pixel(logits_a, p_soft, w_conf, m_curr)
        ce_b = self.loss_function._masked_ce_per_pixel(logits_b, p_soft, w_conf, m_curr)

        # --- class-balanced selection (detached indices) ---
        j_a = self.loss_function._class_balanced_mask(ce_a.detach(), p_soft, keep_rate)
        j_b = self.loss_function._class_balanced_mask(ce_b.detach(), p_soft, keep_rate)

        # --- cross losses ---
        # fθA is optimized on J_B (samples selected by fθB)
        loss_a_base = ce_a[j_b].mean() if j_b.any() else ce_a.mean()
        # fθB is optimized on J_A (samples selected by fθA)
        loss_b_base = ce_b[j_a].mean() if j_a.any() else ce_b.mean()

        # --- neighbourhood regularization (optional, on fθA only) ---
        l_reg = torch.tensor(0.0, device=images.device)
        if self.loss_function.lambda_reg > 0 and hasattr(
            self.loss_function, "compute_neighborhood_reg"
        ):
            l_reg = self.loss_function.compute_neighborhood_reg(logits_a, images)

        loss_a = loss_a_base + self.loss_function.lambda_reg * l_reg
        loss_b = loss_b_base

        # --- manual backward: fθA ---
        opt_a.zero_grad()
        self.manual_backward(loss_a)
        opt_a.step()

        # --- manual backward: fθB ---
        opt_b.zero_grad()
        self.manual_backward(loss_b)
        opt_b.step()

        total = (loss_a + loss_b).detach()

        self.log(
            "train/loss",
            total,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/loss_a", loss_a.detach(), on_step=True, on_epoch=True, sync_dist=True
        )
        self.log(
            "train/loss_b", loss_b.detach(), on_step=True, on_epoch=True, sync_dist=True
        )

        return total
