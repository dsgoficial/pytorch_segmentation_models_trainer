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
from shapely.geometry import Polygon, LineString, Point
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


def polis(polygon_a: Polygon, polygon_b: Polygon) -> float:
    """Compute the polis metric between two polygons.

    Args:
        polygon_a (Polygon): Shapely polygon
        polygon_b (Polygon): Shapely polygon

    Returns:
        float: polis metric
    """
    bounds_a, bounds_b = polygon_a.exterior, polygon_b.exterior
    return _one_side_polis(bounds_a.coords, bounds_b) + _one_side_polis(
        bounds_b.coords, bounds_a
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
    return distance_sum / float(2 * len(coords))


def batch_polis(batch_polygon_a: np.array, batch_polygon_b: np.array) -> np.array:
    """Compute the polis metric between two polygon batches.

    Args:
    """
    func = lambda x: polis(Polygon(x[0]), Polygon(x[1]))
    return np.array(list(map(func, zip(batch_polygon_a, batch_polygon_b))))
