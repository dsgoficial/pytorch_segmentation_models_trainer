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
 *   https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning/     *
 ****
"""

import math
from functools import partial

import scipy.interpolate
import numpy as np
from omegaconf.listconfig import ListConfig
import torch
import torch.distributed
from torch.nn import functional as F
import logging

from pytorch_segmentation_models_trainer.utils import math_utils
from pytorch_segmentation_models_trainer.custom_metrics import metrics
from pytorch_segmentation_models_trainer.utils import frame_field_utils
from pytorch_segmentation_models_trainer.utils.tensor_utils import SpatialGradient


# --- Base classes --- #
logger = logging.getLogger(__name__)

class Loss(torch.nn.Module):
    def __init__(self, name):
        """
        Attribute extra_info can be used in self.compute() to add intermediary 
        results of loss computation for visualization for example.
        It is the second output of self.__call__()

        :param name:
        """
        super(Loss, self).__init__()
        self.name = name
        self.norm_meter = None
        self.norm = torch.nn.parameter.Parameter(torch.Tensor(1), requires_grad=False)
        self.reset_norm()
        self.extra_info = {}  #

    def reset_norm(self):
        self.norm_meter = math_utils.AverageMeter("{}_norm".format(self.name), init_val=1)
        self.norm[0] = self.norm_meter.val

    def update_norm(self, pred_batch, gt_batch, nums):
        loss = self.compute(pred_batch, gt_batch)
        self.norm_meter.update(loss, nums)
        self.norm[0] = self.norm_meter.val

    def sync(self, world_size):
        """
        This method should be used to synchronize loss norms across GPUs when using distributed training
        :return:
        """
        torch.distributed.all_reduce(self.norm)
        self.norm /= world_size

    def compute(self, pred_batch, gt_batch):
        raise NotImplementedError

    def forward(self, pred_batch, gt_batch, normalize=True):
        loss = self.compute(pred_batch, gt_batch)
        if normalize:
            assert 1e-9 < self.norm[0], "self.norm[0] <= 1e-9 -> this might lead to numerical instabilities."
            loss = loss / self.norm[0]
        extra_info = self.extra_info
        self.extra_info = {}  # Re-init extra_info
        return loss, extra_info

    def __repr__(self):
        return "{} (name={}, norm={:0.06})".format(self.__class__.__name__, self.name, self.norm[0])


class MultiLoss(torch.nn.Module):
    def __init__(self, loss_funcs, weights, epoch_thresholds=None, pre_processes=None):
        """
        @param loss_funcs:
        @param weights:
        @param pre_processes: List of functions to call with 2 arguments 
        (which are updated): pred_batch, gt_batch to compute only one values used by several losses.
        """
        super(MultiLoss, self).__init__()
        assert len(loss_funcs) == len(weights), \
            "Should have the same amount of loss_funcs ({}) and weights ({})".format(
                len(loss_funcs), len(weights)
            )
        self.loss_funcs = torch.nn.ModuleList(loss_funcs)
        self.weights = self._build_weights(weights, epoch_thresholds)
        self.pre_processes = pre_processes
        for loss_func, weight in zip(self.loss_funcs, self.weights):
            if weight == 0:
                logger.info(
                    f"INFO: loss '{loss_func.name}' has a weight of "
                    "zero and thus won't affect grad update."
                )
    
    def _build_weights(self, weights, epoch_thresholds):
        weight_list = []
        for weight in weights:
            if isinstance(weight, (list, ListConfig)):
                # Weight is a list of coefs corresponding to epoch_thresholds, 
                # they will be interpolated in-between
                weight_list.append(
                    scipy.interpolate.interp1d(
                        epoch_thresholds,
                        weight,
                        bounds_error=False,
                        fill_value=(weight[0], weight[-1])
                    )
                )
            elif isinstance(weight, float) or isinstance(weight, int):
                weight_list.append(float(weight))
            else:
                raise TypeError(f"Type {type(weight)} not supported as a loss coef weight.")
        return weight_list

    def reset_norm(self):
        for loss_func in self.loss_funcs:
            loss_func.reset_norm()

    def update_norm(self, pred_batch, gt_batch, nums):
        if self.pre_processes is not None:
            for pre_process in self.pre_processes:
                pred_batch, gt_batch = pre_process(pred_batch, gt_batch)
        for loss_func in self.loss_funcs:
            loss_func.update_norm(pred_batch, gt_batch, nums)

    def sync(self, world_size):
        """
        This method should be used to synchronize loss norms across GPUs 
        when using distributed training
        :return:
        """
        for loss_func in self.loss_funcs:
            loss_func.sync(world_size)

    def forward(self, pred_batch, gt_batch, normalize=True, epoch=None):
        if self.pre_processes is not None:
            for pre_process in self.pre_processes:
                pred_batch, gt_batch = pre_process(pred_batch, gt_batch)
        total_loss = 0
        individual_losses_dict, extra_dict = {}, {}
        for loss_func_i, weight_i in zip(self.loss_funcs, self.weights):
            loss_i, extra_dict_i = loss_func_i(pred_batch, gt_batch, normalize=normalize)
            current_weight = torch.from_numpy(weight_i(epoch)).to(loss_i.device) \
                if isinstance(weight_i, scipy.interpolate.interpolate.interp1d) and epoch is not None \
                    else weight_i
            total_loss += current_weight * loss_i
            individual_losses_dict[loss_func_i.name] = loss_i
            extra_dict[loss_func_i.name] = extra_dict_i
        return total_loss, individual_losses_dict, extra_dict

    def __repr__(self):
        ret = "\n\t".join([str(loss_func) for loss_func in self.loss_funcs])
        return "{}:\n\t{}".format(self.__class__.__name__, ret)



# --- Specific losses --- #
class SegLoss(Loss):
    def __init__(self, name, gt_channel_selector, bce_coef=0.5, dice_coef=0.5, use_mixed_precision=False):
        """
        :param name:
        :param gt_channel_selector: used to select which channels gt_polygons_image to use to compare to predicted seg
                                    (see docstring of method compute() for more details).
        """
        super(SegLoss, self).__init__(name)
        self.gt_channel_selector = gt_channel_selector
        self.bce_coef = bce_coef
        self.dice_coef = dice_coef
        self.use_mixed_precision = use_mixed_precision

    def compute(self, pred_batch, gt_batch):
        """
        seg and gt_polygons_image do not necessarily have the same channel count.
        gt_selector is used to select which channels of gt_polygons_image to use.
        For example, if seg has C_pred=2 (interior and edge) and
        gt_polygons_image has C_gt=3 (interior, edge and vertex), use gt_channel_selector=slice(0, 2)

        @param pred_batch: key "seg" is shape (N, C_pred, H, W)
        @param gt_batch: key "gt_polygons_image" is shape (N, C_gt, H, W)
        @return:
        """
        pred_seg = pred_batch["seg"]
        gt_seg = gt_batch["gt_polygons_image"][:, self.gt_channel_selector, ...]
        weights = gt_batch["seg_loss_weights"][:, self.gt_channel_selector, ...]
        dice = metrics.dice_loss(pred_seg, gt_seg)
        mean_dice = torch.mean(dice)
        # if mixed precision is used, the right function is binary_cross_entropy_with_logits
        # because it can handle operations with fp16 and fp32
        cross_entropy_func = F.binary_cross_entropy if not self.use_mixed_precision \
            else F.binary_cross_entropy_with_logits
        mean_cross_entropy = cross_entropy_func(
            pred_seg, gt_seg, weight=weights, reduction="mean"
        )
        return self.bce_coef * mean_cross_entropy + self.dice_coef * mean_dice


class CrossfieldAlignLoss(Loss):
    def __init__(self, name):
        super(CrossfieldAlignLoss, self).__init__(name)

    def compute(self, pred_batch, gt_batch):
        c0, c2 = pred_batch["crossfield"][:, :2], pred_batch["crossfield"][:, 2:]
        z = gt_batch["gt_field"]
        gt_polygons_image = gt_batch["gt_polygons_image"]
        assert 2 <= gt_polygons_image.shape[1], \
            "gt_polygons_image should have at least 2 channels for interior and edges"
        gt_edges = gt_polygons_image[:, 1, ...]
        align_loss = frame_field_utils.framefield_align_error(c0, c2, z, complex_dim=1)
        avg_align_loss = torch.mean(align_loss * gt_edges)
        self.extra_info["gt_field"] = gt_batch["gt_field"]
        return avg_align_loss


class CrossfieldAlign90Loss(Loss):
    def __init__(self, name):
        super(CrossfieldAlign90Loss, self).__init__(name)

    def compute(self, pred_batch, gt_batch):
        c0, c2 = pred_batch["crossfield"][:, :2], pred_batch["crossfield"][:, 2:]
        z = gt_batch["gt_field"]
        z_90deg = torch.cat((- z[:, 1:2, ...], z[:, 0:1, ...]), dim=1)
        gt_polygons_image = gt_batch["gt_polygons_image"]
        assert gt_polygons_image.shape[1] == 3, \
            "gt_polygons_image should have 3 channels for interior, edges and vertices"
        gt_edges, gt_vertices = gt_polygons_image[:, 1, ...], gt_polygons_image[:, 2, ...]
        gt_edges_minus_vertices = gt_edges - gt_vertices
        gt_edges_minus_vertices = gt_edges_minus_vertices.clamp(0, 1)
        align90_loss = frame_field_utils.framefield_align_error(c0, c2, z_90deg, complex_dim=1)
        avg_align90_loss = torch.mean(align90_loss * gt_edges_minus_vertices)
        return avg_align90_loss


class CrossfieldSmoothLoss(Loss):
    def __init__(self, name):
        super(CrossfieldSmoothLoss, self).__init__(name)
        self.laplacian_penalty = frame_field_utils.LaplacianPenalty(channels=4)

    def compute(self, pred_batch, gt_batch):
        c0c2 = pred_batch["crossfield"]
        gt_polygons_image = gt_batch["gt_polygons_image"]
        gt_edges_inv = 1 - gt_polygons_image[:, 1, ...]
        penalty = self.laplacian_penalty(c0c2)
        avg_penalty = torch.mean(penalty * gt_edges_inv[:, None, ...])
        return avg_penalty


class SegCrossfieldLoss(Loss):
    def __init__(self, name, pred_channel):
        super(SegCrossfieldLoss, self).__init__(name)
        self.pred_channel = pred_channel

    def compute(self, pred_batch, gt_batch):
        c0, c2 = pred_batch["crossfield"][:, :2], pred_batch["crossfield"][:, 2:]
        seg_slice_grads_normed = pred_batch["seg_grads_normed"][:, self.pred_channel, ...]
        seg_slice_grad_norm = pred_batch["seg_grad_norm"][:, self.pred_channel, ...]
        align_loss = frame_field_utils.framefield_align_error(
            c0, c2, seg_slice_grads_normed, complex_dim=1
        )
        avg_align_loss = torch.mean(align_loss * seg_slice_grad_norm)
        self.extra_info["seg_slice_grads"] = pred_batch["seg_grads"][:, self.pred_channel, ...]
        return avg_align_loss


class SegEdgeInteriorLoss(Loss):
    """
    Enforce seg edge to be equal to interior grad norm except inside buildings
    """

    def __init__(self, name):
        super(SegEdgeInteriorLoss, self).__init__(name)

    def compute(self, pred_batch, batch):
        seg_interior, seg_edge = pred_batch["seg"][:, 0, ...], pred_batch["seg"][:, 1, ...]
        seg_interior_grad_norm = pred_batch["seg_grad_norm"][:, 0, ...]
        raw_loss = torch.abs(seg_edge - seg_interior_grad_norm)
        # Apply the loss only on interior boundaries and outside of objects
        outside_mask = (torch.cos(np.pi * seg_interior) + 1) / 2
        boundary_mask = (1 - torch.cos(np.pi * seg_interior_grad_norm)) / 2
        mask = torch.max(outside_mask, boundary_mask).float()
        avg_loss = torch.mean(raw_loss * mask)
        return avg_loss

class ComputeSegGrads:
    def __init__(self, device):
        self.spatial_gradient = SpatialGradient(
            mode="scharr", coord="ij", normalized=True, device=device
        )

    def __call__(self, pred_batch, gt_batch):
        pred_batch["seg_grads"] = 2 * self.spatial_gradient(pred_batch["seg"])
        pred_batch["seg_grad_norm"] = pred_batch["seg_grads"].norm(dim=2)  # (b, c, h, w)
        pred_batch["seg_grads_normed"] = pred_batch["seg_grads"] / (pred_batch["seg_grad_norm"][:, :, None, ...] + 1e-6)  # (b, c, 2, h, w)
        return pred_batch, gt_batch

# --- Build combined loss: --- #
def compute_seg_loss_weigths(pred_batch, gt_batch, cfg):
    """
    Combines distances (from U-Net paper) with sizes (from https://github.com/neptune-ai/open-solution-mapping-challenge).

    @param pred_batch:
    @param gt_batch:
    @return:
    """
    # device = gt_batch["distances"].device
    use_dist = cfg.loss_params.seg_loss_params.use_dist
    use_size = cfg.loss_params.seg_loss_params.use_size
    w0 = cfg.loss_params.seg_loss_params.w0
    sigma = cfg.loss_params.seg_loss_params.sigma
    height, width = gt_batch["image"].shape[2], gt_batch["image"].shape[3]
    im_radius = math.sqrt(height * width) / 2

    # --- Class imbalance weight (not forgetting background):
    gt_polygons_mask = (0 < gt_batch["gt_polygons_image"]).float()
    background_freq = 1 - torch.sum(gt_batch["class_freq"], dim=1)
    pixel_class_freq = gt_polygons_mask * gt_batch["class_freq"][:, :, None, None] + \
                       (1 - gt_polygons_mask) * background_freq[:, None, None, None]
    if pixel_class_freq.min() == 0:
        logger.error("ERROR: pixel_class_freq has some zero values, can't divide by zero!")
        logger.error("Candidates:")
        zero_elements = (pixel_class_freq==0).nonzero()
        logger.error(zero_elements)
        for i, path in enumerate(gt_batch['path']):
            logger.error(f"{i} idx: {gt_batch['idx'][i]}, path: {path}, class_freq: {gt_batch['class_freq'][i]}")
        raise ZeroDivisionError
    freq_weights = 1 / pixel_class_freq
    size_weights = None
    if use_size:
        if gt_batch["sizes"].min() == 0:
            logger.error("Candidates:")
            zero_elements = (gt_batch["sizes"]==0).nonzero()
            logger.error(zero_elements)
            for i, path in enumerate(gt_batch['path']):
                logger.error(f"{i} idx: {gt_batch['idx'][i]}, path: {path}, class_freq: {gt_batch['class_freq'][i]}")
            logger.error("ERROR: sizes tensor has zero values, can't divide by zero!")
            raise ZeroDivisionError
        size_weights = 1 + 1 / (im_radius * gt_batch["sizes"])

    distance_weights = None
    if use_dist:
        distance_weights = gt_batch["distances"] * (height + width)  # Denormalize distances
        distance_weights = w0 * torch.exp(-(distance_weights ** 2) / (sigma ** 2))

    gt_batch["seg_loss_weights"] = freq_weights
    if use_dist:
        gt_batch["seg_loss_weights"] += distance_weights
    if use_size:
        gt_batch["seg_loss_weights"] *= size_weights
    return pred_batch, gt_batch


def compute_gt_field(pred_batch, gt_batch):
    gt_crossfield_angle = gt_batch["gt_crossfield_angle"].float()
    # gt_crossfield_angle = gt_crossfield_angle.unsqueeze(0) \
    #     if len(gt_crossfield_angle.shape) == 2 else gt_crossfield_angle.unsqueeze(1)
    gt_field = torch.cat(
        [
            torch.cos(gt_crossfield_angle),
            torch.sin(gt_crossfield_angle)
        ], dim=1)
    gt_batch["gt_field"] = gt_field
    return pred_batch, gt_batch

def build_combined_loss(cfg):
    pre_processes, loss_funcs, weights = [], [], []
    if cfg.compute_seg:
        partial_compute_seg_loss_weigths = partial(
            compute_seg_loss_weigths, cfg=cfg
        )
        pre_processes.append(partial_compute_seg_loss_weigths)
        gt_channel_selector = [
            cfg.seg_params.compute_edge,
            cfg.seg_params.compute_interior,
            cfg.seg_params.compute_vertex
        ]
        loss_funcs.append(
            SegLoss(
                name="seg",
                gt_channel_selector=gt_channel_selector,
                bce_coef=cfg.loss_params.seg_loss_params.bce_coef,
                dice_coef=cfg.loss_params.seg_loss_params.dice_coef,
                use_mixed_precision= True if "precision" in cfg.pl_trainer \
                    and cfg.pl_trainer.precision == 16 else False
            )
        )
        weights.append(cfg.loss_params.multiloss.coefs.seg)

    if cfg.compute_crossfield:
        pre_processes.append(compute_gt_field)
        loss_funcs.append(
            CrossfieldAlignLoss(name="crossfield_align")
        )
        weights.append(
            cfg.loss_params.multiloss.coefs.crossfield_align
        )
        loss_funcs.append(
            CrossfieldAlign90Loss(name="crossfield_align90")
        )
        weights.append(
            cfg.loss_params.multiloss.coefs.crossfield_align90
        )
        loss_funcs.append(
            CrossfieldSmoothLoss(name="crossfield_smooth")
        )
        weights.append(
            cfg.loss_params.multiloss.coefs.crossfield_smooth
        )

    # --- Coupling losses:
    if cfg.compute_seg:
        need_seg_grads = False
        pred_channel = -1
        # Seg interior <-> Crossfield coupling:
        if cfg.seg_params.compute_interior and cfg.compute_crossfield:
            need_seg_grads = True
            pred_channel += 1
            loss_funcs.append(
                SegCrossfieldLoss(name="seg_interior_crossfield", pred_channel=pred_channel)
            )
            weights.append(
                cfg.loss_params.multiloss.coefs.seg_interior_crossfield
            )
        # Seg edge <-> Crossfield coupling:
        if cfg.seg_params.compute_edge and cfg.compute_crossfield:
            need_seg_grads = True
            pred_channel += 1
            loss_funcs.append(
                SegCrossfieldLoss(name="seg_edge_crossfield", pred_channel=pred_channel)
            )
            weights.append(cfg.loss_params.multiloss.coefs.seg_edge_crossfield)

        # Seg edge <-> seg interior coupling:
        if cfg.seg_params.compute_interior and cfg.seg_params.compute_edge:
            need_seg_grads = True
            loss_funcs.append(SegEdgeInteriorLoss(name="seg_edge_interior"))
            weights.append(cfg.loss_params.multiloss.coefs.seg_edge_interior)

        if need_seg_grads:
            pre_processes.append(ComputeSegGrads(cfg.device))

    return MultiLoss(
        loss_funcs,
        weights,
        epoch_thresholds=cfg.loss_params.multiloss.coefs.epoch_thresholds,
        pre_processes=pre_processes
    )
