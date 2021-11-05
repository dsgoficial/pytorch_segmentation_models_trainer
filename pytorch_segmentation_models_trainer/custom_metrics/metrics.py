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

from typing import List, Tuple
import numpy as np
from shapely.geometry import Polygon, LineString, Point
import torch
from pytorch_segmentation_models_trainer.utils import polygonrnn_utils


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


def polygon_iou(vertices1: List, vertices2: List) -> Tuple[float, float, float]:
    """
    calculate iou of two polygons
    :param vertices1: vertices of the first polygon
    :param vertices2: vertices of the second polygon
    :return: the iou, the intersection area, the union area
    """
    geom1 = polygonrnn_utils.handle_vertices(vertices1)
    geom2 = polygonrnn_utils.handle_vertices(vertices2)
    intersection = geom1.intersection(geom2).area
    union = geom1.area + geom2.area - intersection
    iou = 0 if union == 0 else intersection / union
    return iou, intersection, union


def polis(polygon_a: Polygon, polygon_b: Polygon) -> float:
    """Compute the polis metric between two polygons.

    Args:
        polygon_a (Polygon): Shapely polygon
        polygon_b (Polygon): Shapely polygon

    Returns:
        float: polis metric
    """
    bounds_a, bounds_b = polygon_a.exterior, polygon_b.exterior
    return float(
        _one_side_polis(bounds_a.coords, bounds_b)
        + _one_side_polis(bounds_b.coords, bounds_a)
    )


def _one_side_polis(coords: List, bounds: LineString) -> float:
    """Compute the polis metric for one side of a polygon.

    Args:
        coords (List): Coordinates of the polygon
        bounds (LineString): Shapely line string

    Returns:
        float: polis metric
    """
    distance_sum = sum(
        bounds.distance(point) for point in (Point(p) for p in coords[:-1])
    )
    return float(distance_sum / float(2 * len(coords)))


def batch_polis(batch_polygon_a: np.ndarray, batch_polygon_b: np.ndarray) -> np.ndarray:
    """Compute the polis metric between two polygon batches.

    Args:
    """

    def _polis(numpy_polygon_a, numpy_polygon_b):
        if numpy_polygon_a.shape[0] < 3 or numpy_polygon_b.shape[0] < 3:
            return 0
        return polis(Polygon(numpy_polygon_a), Polygon(numpy_polygon_b))

    func = lambda x: _polis(x[0], x[1])
    return np.array(list(map(func, zip(batch_polygon_a, batch_polygon_b))))
