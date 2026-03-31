# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-04-09
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba - Cartographic Engineer
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
 *   Code inspired by the one in                                           *
 *   https://github.com/spmallick/learnopencv/blob/master/Bag-Of-Tricks-For-Image-Classification/model/losses.py     *
 ****
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
import segmentation_models_pytorch as smp

# Based on https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha, T, criterion):
        super().__init__()
        self.criterion = criterion
        self.KLDivLoss = nn.KLDivLoss(reduction="batchmean")
        self.alpha = alpha
        self.T = T

    def forward(self, input, target, teacher_target):
        loss = self.KLDivLoss(
            F.log_softmax(input / self.T, dim=1),
            F.softmax(teacher_target / self.T, dim=1),
        ) * (self.alpha * self.T * self.T) + self.criterion(input, target) * (
            1.0 - self.alpha
        )
        return loss


class MixUpAugmentationLoss(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, input, target, *args):
        # Validation step
        if isinstance(target, torch.Tensor):
            return self.criterion(input, target, *args)
        target_a, target_b, lmbd = target
        return lmbd * self.criterion(input, target_a, *args) + (
            1 - lmbd
        ) * self.criterion(input, target_b, *args)


# Based on https://github.com/pytorch/pytorch/issues/7455
class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_classes, smoothing=0.0, dim=-1, ignore_index=255):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = n_classes
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, output, target, *args):
        output = output.log_softmax(dim=self.dim)
        valid_mask = target != self.ignore_index
        with torch.no_grad():
            target_safe = torch.where(valid_mask, target, 0)
            true_dist = torch.zeros_like(output)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target_safe.data.unsqueeze(1), self.confidence)
            mask_expanded = valid_mask.unsqueeze(1)
            true_dist = torch.where(mask_expanded, true_dist, 0.0)
        per_pixel = torch.sum(-true_dist * output, dim=self.dim)
        if valid_mask.any():
            return per_pixel[valid_mask].mean()
        return torch.tensor(0.0, device=output.device, dtype=output.dtype)


# source: https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0, ignore_index=255):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def k_one_hot(self, targets: torch.Tensor, n_classes: int, smoothing=0.0):
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def reduce_loss(self, loss, valid_mask=None):
        if valid_mask is not None:
            loss = loss[valid_mask]
        return (
            loss.mean()
            if self.reduction == "mean"
            else loss.sum()
            if self.reduction == "sum"
            else loss
        )

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        valid_mask = targets != self.ignore_index
        targets_safe = torch.where(valid_mask, targets, 0)

        targets_oh = self.k_one_hot(targets_safe, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets_oh * log_preds).sum(dim=-1), valid_mask)


def smooth_cross_entropy_loss(
    pred, target, weight=None, reduction="mean", smoothing=0.0, ignore_index=255
):
    assert 0 <= smoothing < 1
    return SmoothCrossEntropyLoss(
        weight=weight, reduction=reduction, smoothing=smoothing,
        ignore_index=ignore_index,
    )(pred, target)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size()[0]
    index = (
        torch.randperm(batch_size).cuda() if use_cuda else torch.randperm(batch_size)
    )

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def dice_loss(y_pred, y_true, smooth=1, eps=1e-7):
    """

    @param y_pred: (N, C, H, W)
    @param y_true: (N, C, H, W)
    @param smooth:
    @param eps:
    @return: (N, C)
    """
    numerator = 2 * torch.sum(y_true * y_pred, dim=(-1, -2))
    denominator = torch.sum(y_true, dim=(-1, -2)) + torch.sum(y_pred, dim=(-1, -2))
    return 1 - (numerator + smooth) / (denominator + smooth + eps)


def tversky_loss(y_pred, y_true, alpha, beta, smooth=0, eps=1e-6):
    true_pos = torch.sum(y_pred * y_true, dim=(-1, -2))
    false_neg = torch.sum(y_true * (1 - y_pred), dim=(-1, -2))
    false_pos = torch.sum((1 - y_true) * y_pred, dim=(-1, -2))
    return (true_pos + smooth + eps) / (
        smooth + true_pos + alpha * false_pos + beta * false_neg + eps
    )


def focal_tversky_loss(y_pred, y_true, gamma=0.25, alpha=0.01, beta=0.99):
    """
    @param y_pred: (N, C, H, W)
    @param y_true: (N, C, H, W)
    @param gamma:
    @param alpha:
    @return: (N, C)
    """
    t_loss = tversky_loss(y_pred, y_true, alpha, beta)
    return torch.pow(1 - t_loss, gamma)

def _compute_reverse_ce(pred_probs, targets, valid_mask, num_classes):
    """Compute the Reverse Cross-Entropy term for SCE.

    Args:
        pred_probs: [B, C, H, W] clamped softmax probabilities.
        targets: [B, H, W] ground-truth class indices.
        valid_mask: [B, H, W] boolean mask of valid pixels.
        num_classes: number of classes.

    Returns:
        Scalar reverse-CE loss averaged over valid pixels.
    """
    targets_safe = targets.clone()
    targets_safe[~valid_mask] = 0
    one_hot = (
        F.one_hot(targets_safe, num_classes)
        .permute(0, 3, 1, 2)
        .float()
    )
    one_hot = torch.clamp(one_hot, min=1e-4, max=1.0)
    rce = (-pred_probs * torch.log(one_hot)).sum(dim=1)  # (B, H, W)
    rce = (rce * valid_mask.float()).sum() / (
        valid_mask.float().sum() + 1e-6
    )
    return rce


def _compute_jml(probas_masked, label, smooth, jml_classes):
    """Compute JML (Jaccard Metric Loss) from masked predictions and labels.

    Args:
        probas_masked: [B, C, H, W] masked softmax probabilities.
        label: [B, C, H, W] one-hot or soft label tensor (masked).
        smooth: smoothing constant for numerical stability.
        jml_classes: 'present' to average only over present classes, or 'all'.

    Returns:
        Scalar JML loss.
    """
    prob_card = probas_masked.sum(dim=(0, 2, 3))  # [C]
    label_card = label.sum(dim=(0, 2, 3))  # [C]
    diff_card = (probas_masked - label).abs().sum(dim=(0, 2, 3))  # [C]

    tp = (prob_card + label_card - diff_card) / 2
    fp = prob_card - tp
    fn = label_card - tp

    iou = (tp + smooth) / (tp + fp + fn + smooth)

    if jml_classes == "present":
        present = label_card > 0
        if present.any():
            return 1.0 - iou[present].mean()
        else:
            return torch.tensor(
                0.0, device=probas_masked.device, requires_grad=True
            )
    else:
        return 1.0 - iou.mean()


def _lovasz_grad(gt_sorted):
    """Gradient of the Lovász extension w.r.t sorted errors.

    Computes the change in Jaccard loss for each additional misclassified
    pixel, given a sorted list of ground-truth foreground indicators.

    Reference: Berman et al., "The Lovász-Softmax loss", CVPR 2018.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_softmax_flat(probas, labels, classes="present"):
    """Multi-class Lovász-Softmax loss on flattened valid pixels.

    Args:
        probas: [P, C] class probabilities (after softmax).
        labels: [P] ground-truth class indices.
        classes: 'present' to average only over classes in the batch,
                 'all' to average over all C classes.
    Returns:
        Scalar loss.
    """
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()  # foreground for class c
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            fg_class = 1.0 - probas[:, 0]
        else:
            fg_class = probas[:, c]
        errors = (fg - fg_class).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        grad = _lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if len(losses) == 0:
        return torch.tensor(0.0, device=probas.device, requires_grad=True)
    return torch.stack(losses).mean()


class WeightedLovaszSCELoss(nn.Module):
    """Combined Lovász-Softmax + Symmetric Cross-Entropy (SCE) Loss.

    Lovász-Softmax is a tractable convex surrogate for the IoU (Jaccard)
    metric, providing direct IoU optimisation unlike Dice (which
    approximates F1).  Combined with SCE for robustness to noisy labels.

    Based on:
      - Berman et al., "The Lovász-Softmax loss", CVPR 2018.
      - Wang et al., "Symmetric Cross Entropy for Robust Learning
        with Noisy Labels", ICCV 2019.

    Args:
        num_classes (int): Number of segmentation classes.
        lovasz_weight (float): Weight for the Lovász-Softmax term.
        sce_weight (float): Weight for the SCE term.
        ignore_index (int): Label value to ignore (default 255).
        sce_alpha (float): Weight of CE inside SCE (default 1.0).
        sce_beta (float): Weight of Reverse-CE inside SCE (default 0.5).
        per_image (bool): Compute Lovász per image then average (default
            False — batch mode, more stable gradients).
        lovasz_classes (str): 'present' to average only over classes in
            the batch, 'all' to average over all classes.
    """

    def __init__(
        self,
        num_classes: int,
        lovasz_weight: float = 0.25,
        sce_weight: float = 0.75,
        ignore_index: int = 255,
        sce_alpha: float = 1.0,
        sce_beta: float = 0.5,
        per_image: bool = False,
        lovasz_classes: str = "present",
    ):
        super(WeightedLovaszSCELoss, self).__init__()

        self.num_classes = num_classes
        self.lovasz_weight = lovasz_weight
        self.sce_weight = sce_weight
        self.ignore_index = ignore_index
        self.sce_alpha = sce_alpha
        self.sce_beta = sce_beta
        self.per_image = per_image
        self.lovasz_classes = lovasz_classes

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _lovasz_softmax(self, probas, labels):
        """Multi-class Lovász-Softmax loss.

        Args:
            probas: [B, C, H, W] class probabilities (after softmax).
            labels: [B, H, W] ground-truth class indices.
        """
        if self.per_image:
            losses = []
            for prob, lab in zip(probas, labels):
                valid = lab != self.ignore_index
                if not valid.any():
                    continue
                prob_valid = prob[:, valid].transpose(0, 1)  # [P, C]
                lab_valid = lab[valid]
                losses.append(
                    _lovasz_softmax_flat(
                        prob_valid, lab_valid, self.lovasz_classes
                    )
                )
            if len(losses) == 0:
                return torch.tensor(
                    0.0, device=probas.device, requires_grad=True
                )
            return torch.stack(losses).mean()
        else:
            # Flatten batch
            valid = labels != self.ignore_index
            if not valid.any():
                return torch.tensor(
                    0.0, device=probas.device, requires_grad=True
                )
            vprobas = probas.permute(0, 2, 3, 1).reshape(-1, probas.size(1))
            vlabels = labels.reshape(-1)
            vmask = valid.reshape(-1)
            vprobas = vprobas[vmask]
            vlabels = vlabels[vmask]
            return _lovasz_softmax_flat(
                vprobas, vlabels, self.lovasz_classes
            )

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(
                0.0, device=predictions.device, requires_grad=True
            )

        # --- Lovász-Softmax ---
        probas = F.softmax(predictions, dim=1)
        lovasz = self._lovasz_softmax(probas, targets)

        # --- Standard CE ---
        ce = self.ce_loss(predictions, targets)

        # --- Reverse CE ---
        pred_probs = torch.clamp(probas, min=1e-7, max=1.0)
        rce = _compute_reverse_ce(pred_probs, targets, valid_mask, self.num_classes)
        sce = self.sce_alpha * ce + self.sce_beta * rce

        return self.lovasz_weight * lovasz + self.sce_weight * sce


class WeightedJMLSCELoss(nn.Module):
    """Combined Jaccard Metric Loss (JML) + Symmetric Cross-Entropy (SCE).

    JML uses L1-norm cardinalities to compute a proper Jaccard surrogate
    that naturally supports soft labels (unlike Lovász-Softmax).
    SCE adds robustness to noisy/unreliable labels via the reverse-CE term.

    Based on:
      - Wang et al., "JDTLosses", NeurIPS 2023.
      - Wang et al., "Symmetric Cross Entropy for Robust Learning
        with Noisy Labels", ICCV 2019.

    Args:
        num_classes (int): Number of segmentation classes.
        jml_weight (float): Weight for the JML term (default 0.25).
        sce_weight (float): Weight for the SCE term (default 0.75).
        ignore_index (int): Label value to ignore (default 255).
        sce_alpha (float): Weight of CE inside SCE (default 1.0).
        sce_beta (float): Weight of Reverse-CE inside SCE (default 0.5).
        smooth (float): Smoothing constant for JML (default 1e-3).
        jml_classes (str): 'present' to average only over classes in the
            batch, 'all' to average over all classes.
        label_smoothing (float): Label smoothing for CE loss (default 0.0).
            Applied via PyTorch's built-in CrossEntropyLoss label_smoothing.
    """

    def __init__(
        self,
        num_classes: int,
        jml_weight: float = 0.25,
        sce_weight: float = 0.75,
        ignore_index: int = 255,
        sce_alpha: float = 1.0,
        sce_beta: float = 0.5,
        smooth: float = 1e-3,
        jml_classes: str = "present",
        label_smoothing: float = 0.0,
    ):
        super(WeightedJMLSCELoss, self).__init__()

        self.num_classes = num_classes
        self.jml_weight = jml_weight
        self.sce_weight = sce_weight
        self.ignore_index = ignore_index
        self.sce_alpha = sce_alpha
        self.sce_beta = sce_beta
        self.smooth = smooth
        self.jml_classes = jml_classes
        self.label_smoothing = label_smoothing

        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index, label_smoothing=label_smoothing
        )

    def _smooth_one_hot(self, targets, valid_mask):
        """Build smoothed one-hot from hard targets.

        Args:
            targets: [B, H, W] ground-truth class indices.
            valid_mask: [B, H, W] boolean mask of valid pixels.

        Returns:
            label: [B, C, H, W] smoothed one-hot (masked).
        """
        C = self.num_classes
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0
        one_hot = (
            F.one_hot(targets_safe, C).permute(0, 3, 1, 2).float()
        )  # [B, C, H, W]

        if self.label_smoothing > 0:
            eps = self.label_smoothing
            one_hot = one_hot * (1.0 - eps) + eps / C

        mask = valid_mask.unsqueeze(1).float()  # [B, 1, H, W]
        return one_hot * mask

    def _jml_loss(self, probas, targets, valid_mask):
        """JML1: Jaccard Metric Loss with L1-norm cardinalities.

        Computes dataset-level mIoU surrogate (sum tp/fp/fn across batch).

        Args:
            probas: [B, C, H, W] softmax probabilities.
            targets: [B, H, W] ground-truth class indices.
            valid_mask: [B, H, W] boolean mask of valid pixels.
        """
        label = self._smooth_one_hot(targets, valid_mask)
        mask = valid_mask.unsqueeze(1).float()
        probas_masked = probas * mask

        return _compute_jml(probas_masked, label, self.smooth, self.jml_classes)

    def _jml_loss_soft(self, probas, targets_soft, valid_mask):
        """JML with soft targets (pre-computed probability distributions).

        Args:
            probas: [B, C, H, W] softmax probabilities.
            targets_soft: [B, C, H, W] soft label distributions.
            valid_mask: [B, H, W] boolean mask of valid pixels.
        """
        mask = valid_mask.unsqueeze(1).float()  # [B, 1, H, W]
        label = targets_soft * mask
        probas_masked = probas * mask

        return _compute_jml(probas_masked, label, self.smooth, self.jml_classes)

    def _sce_soft(self, predictions, probas, targets_soft, valid_mask):
        """SCE with soft targets.

        Replaces F.cross_entropy (which requires long targets) with manual
        soft CE: -sum(y_soft * log_softmax(pred)), and adapts Reverse-CE
        to work with soft target distributions.

        Args:
            predictions: [B, C, H, W] raw logits.
            probas: [B, C, H, W] softmax probabilities.
            targets_soft: [B, C, H, W] soft label distributions.
            valid_mask: [B, H, W] boolean mask of valid pixels.
        """
        mask_float = valid_mask.float()
        n_valid = mask_float.sum() + 1e-6

        # Soft CE: -sum(y * log_softmax(pred)) per pixel
        log_pred = F.log_softmax(predictions, dim=1)  # (B, C, H, W)
        ce_per_pixel = -(targets_soft * log_pred).sum(dim=1)  # (B, H, W)
        ce = (ce_per_pixel * mask_float).sum() / n_valid

        # Reverse CE: -sum(p * log(y_soft)) per pixel
        pred_probs = torch.clamp(probas, min=1e-7, max=1.0)
        targets_clamped = torch.clamp(targets_soft, min=1e-4, max=1.0)
        rce_per_pixel = -(pred_probs * torch.log(targets_clamped)).sum(
            dim=1
        )  # (B, H, W)
        rce = (rce_per_pixel * mask_float).sum() / n_valid

        return self.sce_alpha * ce + self.sce_beta * rce

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        # Auto-detect soft vs hard targets
        is_soft = targets.is_floating_point() and targets.dim() == 4

        if is_soft:
            # Soft targets: (B, C, H, W) float — ignore where sum == 0
            valid_mask = targets.sum(dim=1) > 0  # (B, H, W)
        else:
            # Hard targets: (B, H, W) long — ignore where == ignore_index
            valid_mask = targets != self.ignore_index

        if not valid_mask.any():
            return torch.tensor(
                0.0, device=predictions.device, requires_grad=True
            )

        probas = F.softmax(predictions, dim=1)

        # --- JML ---
        if is_soft:
            jml = self._jml_loss_soft(probas, targets, valid_mask)
        else:
            jml = self._jml_loss(probas, targets, valid_mask)

        # --- SCE ---
        if is_soft:
            sce = self._sce_soft(predictions, probas, targets, valid_mask)
        else:
            # Standard CE
            ce = self.ce_loss(predictions, targets)

            # Reverse CE (uses smoothed one-hot for consistency)
            pred_probs = torch.clamp(probas, min=1e-7, max=1.0)
            smooth_oh = self._smooth_one_hot(targets, valid_mask)
            smooth_oh_clamped = torch.clamp(smooth_oh, min=1e-4, max=1.0)
            mask_float = valid_mask.float()
            n_valid = mask_float.sum() + 1e-6
            rce = (-pred_probs * torch.log(smooth_oh_clamped)).sum(dim=1)
            rce = (rce * mask_float).sum() / n_valid
            sce = self.sce_alpha * ce + self.sce_beta * rce

        return self.jml_weight * jml + self.sce_weight * sce


class GCBLLoss(nn.Module):
    """Guided Contrastive Boundary Learning loss.

    Applies pixel-wise contrastive learning at class boundary regions to
    push apart features of confusing adjacent classes (e.g. campo vs
    veg_cultivada).  Uses a lightweight projection head that maps the
    model logits to a compact embedding space and computes InfoNCE on
    sampled boundary pixels.

    The projector parameters are included in the model's optimizer
    automatically (via nn.Module registration in the loss function).

    Ref: Inspired by Li et al., "Guided Contrastive Boundary Learning
         for Semantic Segmentation", PR 2024.

    Args:
        num_classes: Number of segmentation classes.
        embed_dim: Embedding dimension for the projection head (default 32).
        temperature: InfoNCE temperature (default 0.07).
        max_boundary_samples: Max boundary pixels to sample per batch
            (default 512).  Controls memory usage.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 32,
        temperature: float = 0.07,
        max_boundary_samples: int = 512,
    ):
        super(GCBLLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.max_boundary_samples = max_boundary_samples

        # Lightweight projection head: logits → embedding
        self.projector = nn.Sequential(
            nn.Conv2d(num_classes, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
        )

    def _detect_boundaries(self, labels):
        """Detect boundary pixels between different classes (4-connected).

        Args:
            labels: (B, H, W) integer class indices.

        Returns:
            (B, H, W) boolean mask — True at boundary pixels.
        """
        valid = labels < self.num_classes
        boundary = torch.zeros_like(labels, dtype=torch.bool)

        # Right neighbor
        diff_r = (labels[:, :, :-1] != labels[:, :, 1:]) & valid[:, :, :-1] & valid[:, :, 1:]
        boundary[:, :, :-1] |= diff_r
        boundary[:, :, 1:] |= diff_r

        # Bottom neighbor
        diff_b = (labels[:, :-1, :] != labels[:, 1:, :]) & valid[:, :-1, :] & valid[:, 1:, :]
        boundary[:, :-1, :] |= diff_b
        boundary[:, 1:, :] |= diff_b

        return boundary & valid

    def forward(self, predictions, targets):
        """Compute GCBL contrastive loss at boundary pixels.

        Args:
            predictions: (B, C, H, W) logits.
            targets: (B, H, W) hard labels or (B, C, H, W) soft labels.
        """
        # Soft → hard for boundary detection
        if targets.is_floating_point() and targets.dim() == 4:
            valid_mask = targets.sum(dim=1) > 0
            hard_labels = targets.argmax(dim=1)
            hard_labels[~valid_mask] = 255
        else:
            hard_labels = targets

        # Project logits to embedding space
        embeddings = self.projector(predictions)  # (B, D, H, W)
        embeddings = F.normalize(embeddings, dim=1)

        B, D, H, W = embeddings.shape

        # Detect boundaries
        boundary = self._detect_boundaries(hard_labels)

        boundary_flat = boundary.reshape(-1)
        boundary_indices = torch.where(boundary_flat)[0]

        if len(boundary_indices) < 4:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Sample boundary pixels
        n_samples = min(self.max_boundary_samples, len(boundary_indices))
        perm = torch.randperm(len(boundary_indices), device=predictions.device)[:n_samples]
        sampled_indices = boundary_indices[perm]

        # Get coordinates
        batch_idx = sampled_indices // (H * W)
        remainder = sampled_indices % (H * W)
        y_idx = remainder // W
        x_idx = remainder % W

        # Extract embeddings and labels at sampled pixels
        anchor_embeds = embeddings[batch_idx, :, y_idx, x_idx]  # (N, D)
        anchor_labels = hard_labels[batch_idx, y_idx, x_idx]  # (N,)

        # Similarity matrix
        sim = torch.mm(anchor_embeds, anchor_embeds.t()) / self.temperature

        # Positive mask (same class) excluding self
        label_eq = anchor_labels.unsqueeze(0) == anchor_labels.unsqueeze(1)
        self_mask = ~torch.eye(n_samples, dtype=torch.bool, device=sim.device)
        pos_mask = label_eq & self_mask

        # Need both positive and negative pairs
        has_pos = pos_mask.any(dim=1)
        has_neg = (~label_eq & self_mask).any(dim=1)
        valid = has_pos & has_neg

        if not valid.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        sim = sim[valid]
        pos_mask = pos_mask[valid]
        self_mask_v = self_mask[valid]

        # Numerical stability
        sim = sim - sim.max(dim=1, keepdim=True)[0]

        # InfoNCE
        exp_sim = torch.exp(sim) * self_mask_v.float()
        pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)

        loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8).mean()

        return loss


class WeightedJMLSCEGCBLLoss(nn.Module):
    """Combined JML + SCE + GCBL loss.

    Wraps WeightedJMLSCELoss with a GCBLLoss auxiliary term for
    boundary-focused contrastive learning.

    Args:
        num_classes, jml_weight, sce_weight, ignore_index, sce_alpha,
        sce_beta, smooth, jml_classes, label_smoothing: forwarded to
        WeightedJMLSCELoss.
        gcbl_weight: Weight for GCBL term (default 0.1).
        gcbl_embed_dim: Embedding dimension (default 32).
        gcbl_temperature: InfoNCE temperature (default 0.07).
        gcbl_max_samples: Max boundary pixels to sample (default 512).
    """

    def __init__(
        self,
        num_classes: int,
        jml_weight: float = 0.25,
        sce_weight: float = 0.75,
        ignore_index: int = 255,
        sce_alpha: float = 1.0,
        sce_beta: float = 0.5,
        smooth: float = 1e-3,
        jml_classes: str = "present",
        label_smoothing: float = 0.0,
        gcbl_weight: float = 0.1,
        gcbl_embed_dim: int = 32,
        gcbl_temperature: float = 0.07,
        gcbl_max_samples: int = 512,
    ):
        super(WeightedJMLSCEGCBLLoss, self).__init__()
        self.main_loss = WeightedJMLSCELoss(
            num_classes=num_classes,
            jml_weight=jml_weight,
            sce_weight=sce_weight,
            ignore_index=ignore_index,
            sce_alpha=sce_alpha,
            sce_beta=sce_beta,
            smooth=smooth,
            jml_classes=jml_classes,
            label_smoothing=label_smoothing,
        )
        self.gcbl = GCBLLoss(
            num_classes=num_classes,
            embed_dim=gcbl_embed_dim,
            temperature=gcbl_temperature,
            max_boundary_samples=gcbl_max_samples,
        )
        self.gcbl_weight = gcbl_weight

    def forward(self, predictions, targets):
        main = self.main_loss(predictions, targets)
        gcbl = self.gcbl(predictions, targets)
        return main + self.gcbl_weight * gcbl


class WeightedDiceCrossEntropyLoss(nn.Module):
    """
    Combined Cross Entropy and Dice Loss for segmentation tasks.

    Args:
        num_classes (int): Number of classes in the segmentation task
        dice_weight (float): Weight for dice loss (default: 0.75)
        ce_weight (float): Weight for cross entropy loss (default: 0.25)
        smooth_factor (float): Smoothing factor for cross entropy loss (default: 0)
        ignore_index (int): Label value to ignore in loss computation (default: None)
    """
    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.75,
        ce_weight: float = 0.25,
        smooth_factor: float = 0.0,
        ignore_index: int = None
    ):
        super(WeightedDiceCrossEntropyLoss, self).__init__()

        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index if ignore_index is not None else 255

        # Determine mode based on number of classes
        mode = "multiclass" if num_classes > 2 else "binary"

        # Initialize Dice Loss from segmentation_models_pytorch
        self.dice_loss = smp.losses.DiceLoss(
            mode=mode,
            ignore_index=self.ignore_index,
        )

        # Initialize Cross Entropy Loss
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=smooth_factor,
            ignore_index=self.ignore_index,
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined loss.

        Args:
            predictions (torch.Tensor): Model predictions (logits)
            targets (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Weighted sum of dice loss and cross entropy loss
        """
        # Compute individual losses
        dice = self.dice_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)

        # Return weighted sum
        combined = self.dice_weight * dice + self.ce_weight * ce

        return combined


class WeightedDiceSCELoss(nn.Module):
    """
    Combined Dice + Symmetric Cross-Entropy (SCE) Loss for noisy labels.

    SCE = sce_alpha * CE(pred, target) + sce_beta * RCE(pred, target)

    The Reverse Cross-Entropy (RCE) term prevents the model from
    over-fitting to noisy labels by using the model's own predictions
    as regularisation.  Based on Wang et al., ICCV 2019.

    Args:
        num_classes (int): Number of classes.
        dice_weight (float): Weight for the Dice component.
        sce_weight (float): Weight for the SCE component.
        smooth_factor (float): Label smoothing for the CE sub-term.
        ignore_index (int): Label value to ignore (default 255).
        sce_alpha (float): Weight of CE inside SCE (default 1.0).
        sce_beta (float): Weight of Reverse-CE inside SCE (default 0.5).
    """

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.25,
        sce_weight: float = 0.75,
        smooth_factor: float = 0.0,
        ignore_index: int = 255,
        sce_alpha: float = 1.0,
        sce_beta: float = 0.5,
    ):
        super(WeightedDiceSCELoss, self).__init__()

        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.sce_weight = sce_weight
        self.ignore_index = ignore_index
        self.sce_alpha = sce_alpha
        self.sce_beta = sce_beta

        mode = "multiclass" if num_classes > 2 else "binary"

        self.dice_loss = smp.losses.DiceLoss(
            mode=mode,
            ignore_index=ignore_index,
        )

        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=smooth_factor,
            ignore_index=ignore_index,
        )

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # --- Dice ---
        dice = self.dice_loss(predictions, targets)

        # --- Standard CE ---
        ce = self.ce_loss(predictions, targets)

        # --- Reverse CE ---
        pred_probs = torch.clamp(F.softmax(predictions, dim=1), min=1e-7, max=1.0)
        rce = _compute_reverse_ce(pred_probs, targets, valid_mask, self.num_classes)
        sce = self.sce_alpha * ce + self.sce_beta * rce

        return self.dice_weight * dice + self.sce_weight * sce
