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
    def __init__(self, n_classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = n_classes
        self.dim = dim

    def forward(self, output, target, *args):
        output = output.log_softmax(dim=self.dim)
        with torch.no_grad():
            # Create matrix with shapes batch_size x n_classes
            true_dist = torch.zeros_like(output)
            # Initialize all elements with epsilon / N - 1
            true_dist.fill_(self.smoothing / (self.cls - 1))
            # Fill correct class for each sample in the batch with 1 - epsilon
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * output, dim=self.dim))


# source: https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets: torch.Tensor, n_classes: int, smoothing=0.0):
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def reduce_loss(self, loss):
        return (
            loss.mean()
            if self.reduction == "mean"
            else loss.sum()
            if self.reduction == "sum"
            else loss
        )

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))


def smooth_cross_entropy_loss(
    pred, target, weight=None, reduction="mean", smoothing=0.0
):
    assert 0 <= smoothing < 1
    return SmoothCrossEntropyLoss(
        weight=weight, reduction=reduction, smoothing=smoothing
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

class WeightedDiceCrossEntropyLoss(nn.Module):
    """
    Combined Cross Entropy and Dice Loss for segmentation tasks.
    
    Args:
        num_classes (int): Number of classes in the segmentation task
        dice_weight (float): Weight for dice loss (default: 0.75)
        ce_weight (float): Weight for cross entropy loss (default: 0.25)
        smooth_factor (float): Smoothing factor for cross entropy loss (default: 0)
    """
    def __init__(
        self, 
        num_classes: int,
        dice_weight: float = 0.75,
        ce_weight: float = 0.25,
        smooth_factor: float = 0.0
    ):
        super(WeightedDiceCrossEntropyLoss, self).__init__()
        
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth_factor = smooth_factor
        
        # Determine mode based on number of classes
        mode = "multiclass" if num_classes > 2 else "binary"
        
        # Initialize Dice Loss from segmentation_models_pytorch
        self.dice_loss = smp.losses.DiceLoss(mode=mode)
        
        # Initialize Cross Entropy Loss
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=smooth_factor)
    
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
