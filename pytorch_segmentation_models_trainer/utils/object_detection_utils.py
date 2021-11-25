# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-08-16
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
 ****
"""
from typing import Dict, List
import torch
from torchvision.ops import box_iou


def evaluate_box_iou(
    target: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Evaluate intersection over union (IOU) for target from dataset and output prediction from model."""
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()


def bbox_xywh_to_xyxy(bbox: List) -> List:
    """
    Convert a bbox from [x, y, w, h] to [x1, y1, x2, y2]
    """
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def bbox_xyxy_to_xywh(bbox: List) -> List:
    """
    Convert a bbox from [x1, y1, x2, y2] to [x, y, w, h]
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]
