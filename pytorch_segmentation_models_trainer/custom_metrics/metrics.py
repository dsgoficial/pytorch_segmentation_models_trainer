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

from typing import List
import numpy as np
from shapely.geometry import Polygon
import torch


def iou(y_pred, y_true, threshold):
    assert (
        len(y_pred.shape) == len(y_true.shape) == 2
    ), "Input tensor shapes should be (N, .)"
    mask_pred = threshold < y_pred
    mask_true = threshold < y_true
    intersection = torch.sum(mask_pred * mask_true, dim=-1)
    union = torch.sum(mask_pred + mask_true, dim=-1)
    r = intersection.float() / union.float()
    r[union == 0] = 1
    return r


def polygon_iou(vertices1: List, vertices2: List) -> float:
    """
    calculate iou of two polygons
    :param vertices1: vertices of the first polygon
    :param vertices2: vertices of the second polygon
    :return: the iou, the intersection area, the union area
    """
    poly1 = Polygon(np.array(vertices1).reshape(-1, 2)).convex_hull
    poly2 = Polygon(np.array(vertices2).reshape(-1, 2)).convex_hull
    if not poly1.intersects(poly2):
        return 0.0
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection
    return float(intersection / union)
