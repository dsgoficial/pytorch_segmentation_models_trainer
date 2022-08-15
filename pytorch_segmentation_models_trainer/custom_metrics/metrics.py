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
 ****
"""

import itertools
from typing import Dict, List, Tuple, Union
import numpy as np
import similaritymeasures as sm
from shapely.geometry import Polygon, LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.geometry.multipolygon import MultiPolygon
import torch
from pytorch_segmentation_models_trainer.utils import polygon_utils, polygonrnn_utils
from shapely.strtree import STRtree
import shapely.wkt


def iou(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float) -> float:
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


def polygon_iou(
    vertices1: Union[List, BaseGeometry], vertices2: Union[List, BaseGeometry]
) -> Tuple[float, float, float]:
    """
    calculate iou of two polygons
    :param vertices1: vertices of the first polygon
    :param vertices2: vertices of the second polygon
    :return: the iou, the intersection area, the union area
    """
    geom1 = polygonrnn_utils.handle_vertices(vertices1)
    geom2 = polygonrnn_utils.handle_vertices(vertices2)
    if geom1.is_valid and geom2.is_valid:
        return _poly_iou(geom1, geom2)
    return _iou_with_invalid_geom(geom1, geom2)


def shapely_polygon_iou(polygon_a: Polygon, polygon_b: Polygon) -> float:
    """Compute the iou between two polygons.

    Args:
        polygon_a (Polygon): Shapely polygon
        polygon_b (Polygon): Shapely polygon

    Returns:
        float: iou
    """
    iou, _, __ = polygon_iou(polygon_a, polygon_b)
    return iou


def _iou_with_invalid_geom(
    geom1: Union[Polygon, MultiPolygon], geom2: Union[Polygon, MultiPolygon]
) -> Tuple[float, float, float]:
    geom1_list = polygonrnn_utils.validate_polygon(geom1)
    geom2_list = polygonrnn_utils.validate_polygon(geom2)
    if geom1_list == []:
        return 0, 0, sum(geom.area for geom in geom2_list)
    if geom2_list == []:
        return 0, 0, sum(geom.area for geom in geom1_list)

    iou_intersection_union_array = sum(
        [
            np.array(_poly_iou(geom_a, geom_b))
            for geom_a in geom1_list
            for geom_b in geom2_list
        ]
    )
    return (
        iou_intersection_union_array[0],
        iou_intersection_union_array[1],
        iou_intersection_union_array[2],
    )


def _poly_iou(
    geom1: Union[Polygon, MultiPolygon], geom2: Union[Polygon, MultiPolygon]
) -> Tuple[float, float, float]:
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


def _one_side_polis(coords: List[Tuple[float, float]], bounds: LineString) -> float:
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


def batch_polis(
    batch_polygon_a: Union[np.ndarray, List[np.ndarray]],
    batch_polygon_b: Union[np.ndarray, List[np.ndarray]],
) -> np.ndarray:
    """Compute the polis metric between two polygon batches.

    Args:
    """

    def _polis(numpy_polygon_a, numpy_polygon_b):
        if numpy_polygon_a.shape[0] < 3 or numpy_polygon_b.shape[0] < 3:
            return 0
        return polis(Polygon(numpy_polygon_a), Polygon(numpy_polygon_b))

    def func(x):
        return _polis(x[0], x[1])

    return np.array(list(map(func, zip(batch_polygon_a, batch_polygon_b))))


def hausdorff_distance(geom1: BaseGeometry, geom2: BaseGeometry) -> float:
    """Compute the hausdorf distance between two geometries.

    Args:
        geom1 (BaseGeometry): Shapely geometry
        geom2 (BaseGeometry): Shapely geometry

    Returns:
        float: hausdorf distance
    """
    return geom1.hausdorff_distance(geom2)


def frechet_distance(
    gt_polygon: Polygon, polygon2: Polygon, minkowski_p_norm: float = 2
) -> float:
    return sm.frechet_dist(
        np.array(gt_polygon.exterior.coords),
        np.array(polygon2.exterior.coords),
        p=minkowski_p_norm,
    )


def polygon_accuracy(
    gt_polygon: Polygon,
    pred_polygon: Polygon,
    grid_size: int = 28,
    sequence_length: int = 60,
) -> float:
    """Compute the polygon accuracy between two polygons.

    Args:
        polygon1 (Polygon): Shapely polygon
        polygon2 (Polygon): Shapely polygon

    Returns:
        float: polygon accuracy
    """
    _, label_index_array1 = polygonrnn_utils.build_arrays(
        np.array(gt_polygon.exterior.coords),
        len(gt_polygon.exterior.coords) - 1,
        sequence_length,
        grid_size=grid_size,
    )
    _, label_index_array2 = polygonrnn_utils.build_arrays(
        np.array(pred_polygon.exterior.coords),
        len(pred_polygon.exterior.coords) - 1,
        sequence_length,
        grid_size=grid_size,
    )
    ta1, ta2 = label_index_array1[2:], label_index_array2[2:]
    correct = (ta1 == ta2).astype(float).sum()
    return correct / ta1.shape[0]


def polygon_mean_max_tangent_angle_errors(
    gt_polygon: Polygon,
    pred_polygon: Polygon,
    sampling_spacing: float = 1.0,
    max_stretch: int = 2,
) -> float:
    return polygon_utils.compute_contour_measure(
        pred_polygon=pred_polygon,
        gt_contours=gt_polygon,
        sampling_spacing=sampling_spacing,
        max_stretch=max_stretch,
    )
