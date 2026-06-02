# -*- coding: utf-8 -*-
"""Curriculum-guided co-teaching loss for noisy weak supervision.

Implements equations 6-11 from Xiao et al. (2026, JSTARS):
  "Distilling 10-m Land Cover Maps from Multi-Source Consensus via
   AlphaEarth Embeddings and Noise-Aware Weak Supervision."
"""

from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F

from pytorch_segmentation_models_trainer.custom_losses.base_loss import Loss


class CurriculumScheduler:
    """Confidence-guided curriculum scheduler (eq. 6-7).

    Ramps sample retention from ``p_start`` to ``p_end`` over ``e_warm``
    warm-up epochs, then holds at ``p_end``.

    Args:
        e_warm: Warm-up duration in epochs (E_warm in paper; default 20).
        p_start: Initial retention fraction (P_start; default 0.5).
        p_end: Final retention fraction (P_end; default 0.95).

    Example:
        scheduler = CurriculumScheduler(e_warm=20, p_start=0.5, p_end=0.95)
        mask = scheduler.curriculum_mask(w_conf, epoch=5)
    """

    def __init__(
        self, e_warm: int = 20, p_start: float = 0.5, p_end: float = 0.95
    ) -> None:
        if not (0.0 < p_start <= p_end <= 1.0):
            raise ValueError(
                f"Must satisfy 0 < p_start <= p_end <= 1.0, "
                f"got p_start={p_start}, p_end={p_end}."
            )
        self.e_warm = e_warm
        self.p_start = p_start
        self.p_end = p_end

    def retention_rate(self, epoch: int) -> float:
        """Compute P_e = min(p_start + (e/E_warm)*(p_end-p_start), p_end).

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            Scalar retention fraction in [p_start, p_end].
        """
        return min(
            self.p_start + (epoch / max(self.e_warm, 1)) * (self.p_end - self.p_start),
            self.p_end,
        )

    def forget_rate(self, epoch: int) -> float:
        """Co-teaching forget rate R_e, linearly ramped 0 → 0.3 over E_warm.

        Args:
            epoch: Current training epoch.

        Returns:
            Scalar forget rate in [0, 0.3].
        """
        return min(0.3 * (epoch / max(self.e_warm, 1)), 0.3)

    def curriculum_mask(self, w_conf: torch.Tensor, epoch: int) -> torch.Tensor:
        """Compute binary mask M_curr = I(W_conf > τ) where τ = Percentile(W_conf, 1-P_e).

        Args:
            w_conf: Per-pixel confidence map ``(B, 1, H, W)`` float.
            epoch: Current training epoch.

        Returns:
            Binary float tensor ``(B, 1, H, W)``.
        """
        p_e = self.retention_rate(epoch)
        tau = torch.quantile(w_conf.detach().float(), 1.0 - p_e)
        return (w_conf > tau).float()


class CoTeachingLoss(Loss):
    """Curriculum-guided symmetric co-teaching loss (eq. 8-11).

    During training (dict ``pred_batch`` with ``logits_a``/``logits_b``):

    1. Curriculum masking: retain high-confidence pixels via M_curr (eq. 7).
    2. Class-balanced selection: J_A and J_B via lowest-loss pixels per class (eq. 9).
    3. Cross-update: model A is trained on J_B, model B on J_A (eq. 11).
    4. Optional neighbourhood regularization: λ · L_reg (eq. 10).

    During validation (plain tensor ``pred_batch``):
    Falls back to standard soft cross-entropy (same as SoftLabelWeightedCELoss).

    Args:
        name: Loss identifier.
        num_classes: Number of segmentation classes C.
        e_warm: Warm-up epochs for curriculum scheduler.
        p_start: Initial curriculum retention rate.
        p_end: Final curriculum retention rate.
        lambda_reg: Weight λ for neighbourhood regularization (0 disables it).
        weight_key: Key for W_conf in gt dict.
        mask_key: Key for P_soft in gt dict.
        **kwargs: Accepted for Hydra / ConfigStore compatibility.

    Example YAML:
        loss:
          _target_: pytorch_segmentation_models_trainer.custom_losses.co_teaching_loss.CoTeachingLoss
          name: co_teaching
          num_classes: 4
          e_warm: 20
          p_start: 0.5
          p_end: 0.95
          lambda_reg: 0.1
    """

    def __init__(
        self,
        name: str = "co_teaching",
        num_classes: int = 2,
        e_warm: int = 20,
        p_start: float = 0.5,
        p_end: float = 0.95,
        lambda_reg: float = 0.1,
        weight_key: str = "w_conf",
        mask_key: str = "mask",
        **kwargs,
    ) -> None:
        super().__init__(name)
        self.num_classes = num_classes
        self.lambda_reg = lambda_reg
        self.weight_key = weight_key
        self.mask_key = mask_key
        self._curriculum = CurriculumScheduler(
            e_warm=e_warm, p_start=p_start, p_end=p_end
        )
        self.current_epoch: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        pred_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
        gt_batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Compute co-teaching loss.

        Args:
            pred_batch: Either a plain logits tensor ``(B, C, H, W)`` (validation)
                or a dict with ``"logits_a"``, ``"logits_b"``, and optionally
                ``"features"`` ``(B, D, H, W)`` (training co-teaching path).
            gt_batch: Soft-label tensor ``(B, C, H, W)`` or a dict with
                ``mask_key`` and optionally ``weight_key``.

        Returns:
            Scalar loss tensor.
        """
        p_soft, w_conf = self._unpack_gt(gt_batch)

        if isinstance(pred_batch, torch.Tensor):
            return self._soft_ce(pred_batch, p_soft, w_conf)

        logits_a = pred_batch["logits_a"]
        logits_b = pred_batch["logits_b"]
        features: Optional[torch.Tensor] = pred_batch.get("features")

        epoch = self.current_epoch
        keep_rate = 1.0 - self._curriculum.forget_rate(epoch)

        if w_conf is not None:
            m_curr = self._curriculum.curriculum_mask(w_conf, epoch)
        else:
            m_curr = torch.ones(
                p_soft.shape[0],
                1,
                p_soft.shape[2],
                p_soft.shape[3],
                device=p_soft.device,
            )

        ce_a = self._masked_ce_per_pixel(logits_a, p_soft, w_conf, m_curr)
        ce_b = self._masked_ce_per_pixel(logits_b, p_soft, w_conf, m_curr)

        j_a = self._class_balanced_mask(ce_a.detach(), p_soft, keep_rate)
        j_b = self._class_balanced_mask(ce_b.detach(), p_soft, keep_rate)

        loss_a = ce_a[j_b].mean() if j_b.any() else ce_a.mean()
        loss_b = ce_b[j_a].mean() if j_a.any() else ce_b.mean()

        l_reg = torch.tensor(0.0, device=logits_a.device, dtype=logits_a.dtype)
        if features is not None and self.lambda_reg > 0:
            l_reg = self.compute_neighborhood_reg(logits_a, features)

        return loss_a + loss_b + self.lambda_reg * l_reg

    def compute_neighborhood_reg(
        self, logits: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """Compute neighbourhood regularization L_reg (eq. 10).

        L_reg = Σ_i Σ_{j∈N(i)} S_ij · D_KL(p(·|x_i) || p(·|x_j))

        Where S_ij is the cosine similarity between feature vectors and N(i)
        is the 3×3 spatial neighbourhood of pixel i.

        Args:
            logits: Model logits ``(B, C, H, W)``.
            features: Input feature map ``(B, D, H, W)`` used to compute S_ij.

        Returns:
            Scalar L_reg tensor.
        """
        B, C, H, W = logits.shape
        D = features.shape[1]

        feat_unfold = F.unfold(features.detach().float(), kernel_size=3, padding=1)
        feat_unfold = feat_unfold.view(B, D, 9, H * W)
        feat_center = features.detach().float().view(B, D, 1, H * W)

        # Clamp to [0, 1]: only penalize inconsistency between similar features.
        # Negative cosine similarity (dissimilar embeddings) yields no penalty.
        s_ij = torch.clamp(
            (F.normalize(feat_center, dim=1) * F.normalize(feat_unfold, dim=1)).sum(
                dim=1
            ),
            min=0.0,
        )  # (B, 9, H*W)

        p_center = F.softmax(logits, dim=1)  # (B, C, H, W)
        log_p_center = F.log_softmax(logits, dim=1)  # (B, C, H, W)

        logits_unfold = F.unfold(logits, kernel_size=3, padding=1)
        logits_unfold = logits_unfold.view(B, C, 9, H * W)
        log_p_neighbor = F.log_softmax(logits_unfold, dim=1)  # (B, C, 9, H*W)

        p_center_exp = p_center.view(B, C, 1, H * W)
        log_p_center_exp = log_p_center.view(B, C, 1, H * W)
        kl_ij = (p_center_exp * (log_p_center_exp - log_p_neighbor)).sum(
            dim=1
        )  # (B, 9, H*W)

        return (s_ij * kl_ij).mean()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _unpack_gt(self, gt_batch: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        """Return (p_soft, w_conf) from gt_batch."""
        if isinstance(gt_batch, dict):
            p_soft = gt_batch[self.mask_key]
            w_conf = gt_batch.get(self.weight_key)
        else:
            p_soft = gt_batch
            w_conf = None
        return p_soft, w_conf

    def _soft_ce(
        self,
        logits: torch.Tensor,
        p_soft: torch.Tensor,
        w_conf: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Standard confidence-weighted soft cross-entropy (fallback / val path)."""
        log_prob = F.log_softmax(logits, dim=1)
        ce = -(p_soft * log_prob).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        if w_conf is not None:
            return (ce * w_conf).mean()
        return ce.mean()

    def _masked_ce_per_pixel(
        self,
        logits: torch.Tensor,
        p_soft: torch.Tensor,
        w_conf: Optional[torch.Tensor],
        m_curr: torch.Tensor,
    ) -> torch.Tensor:
        """Compute M_curr · W_conf · CE per pixel (eq. 8, spatial form).

        Returns:
            ``(B, H, W)`` per-pixel loss tensor (differentiable w.r.t. logits).
        """
        log_prob = F.log_softmax(logits, dim=1)
        ce = -(p_soft * log_prob).sum(dim=1)  # (B, H, W)
        ce = ce * m_curr.squeeze(1)
        if w_conf is not None:
            ce = ce * w_conf.squeeze(1)
        return ce

    def _class_balanced_mask(
        self,
        per_pixel_loss: torch.Tensor,
        p_soft: torch.Tensor,
        keep_rate: float,
    ) -> torch.Tensor:
        """Select the keep_rate fraction of lowest-loss pixels per class (eq. 9).

        Args:
            per_pixel_loss: Detached per-pixel loss ``(B, H, W)``.
            p_soft: Soft label distribution ``(B, C, H, W)``.
            keep_rate: Fraction of pixels to keep per class (1 - R_e).

        Returns:
            Boolean mask ``(B, H, W)``.
        """
        B, C, H, W = p_soft.shape
        argmax_cls = p_soft.argmax(dim=1)  # (B, H, W)
        selected = torch.zeros(B, H, W, dtype=torch.bool, device=p_soft.device)

        for c in range(C):
            cls_mask = argmax_cls == c  # (B, H, W)
            for b in range(B):
                bm = cls_mask[b]  # (H, W)
                n = int(bm.sum().item())
                if n == 0:
                    continue
                k = max(1, round(n * keep_rate))
                loss_vals = per_pixel_loss[b][bm]  # (n,)
                _, order = loss_vals.sort()
                keep_idx = order[:k]
                pixel_coords = bm.nonzero(as_tuple=False)  # (n, 2)
                kept_coords = pixel_coords[keep_idx]  # (k, 2)
                selected[b, kept_coords[:, 0], kept_coords[:, 1]] = True

        return selected
