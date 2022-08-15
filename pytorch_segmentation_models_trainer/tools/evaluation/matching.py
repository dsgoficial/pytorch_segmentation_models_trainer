# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-11-29
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
from typing import Callable, DefaultDict, Dict, List, Tuple, Union
import shapely
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
import shapely.wkt

from pytorch_segmentation_models_trainer.custom_metrics.metrics import (
    frechet_distance,
    hausdorff_distance,
    shapely_polygon_iou,
)


def match_polygon_lists_by_criteria(
    reference_list: List[Polygon],
    target_list: List[Polygon],
    criteria_func: Callable,
    match_criteria: str = "max",
    match_threshold: float = 0.5,
) -> Tuple[List[Dict[str, Polygon]], List[Polygon], List[Polygon]]:
    """
    Matches the reference geometry list with the target geometry list by the intersection over union (IOU) metric.
    :param reference_geometry_list: The reference geometry list.
    :param target_geometry_list: The target geometry list.
    :return: Dictionary with the matched and unmatched polygons.
    """
    assert match_criteria in [
        "max",
        "min",
    ], "Match criteria must be either 'max' or 'min'."
    reference_tree = STRtree(reference_list)
    reference_dict = {polygon.wkt: polygon for polygon in reference_list}
    matched_pair_dict = dict()  # {reference_wkt: target_geometry}
    match_criteria_func = (
        _match_by_max_criteria if match_criteria == "max" else _match_by_min_criteria
    )
    for target in target_list:
        if target.is_empty:
            raise ValueError("Target geometry is empty.")
        candidates = [i for i in reference_tree.query(target) if i.intersects(target)]
        if len(candidates) == 0:
            continue
        metric_values_list = [criteria_func(target, i) for i in candidates]
        match_criteria_func(
            target,
            criteria_func,
            reference_dict,
            metric_values_list,
            match_threshold,
            candidates,
            matched_pair_dict,
        )
    return _build_outputs(
        reference_list, target_list, reference_dict, matched_pair_dict
    )


def _match_by_max_criteria(
    target: Polygon,
    criteria_func: Callable,
    reference_dict: Dict[str, Polygon],
    metric_list: List[float],
    match_threshold: float,
    candidates: List[BaseGeometry],
    matched_pair_dict: Dict[str, Polygon],
) -> None:
    max_metric = max(metric_list)
    if match_threshold > 0 and max_metric <= match_threshold:
        return
    candidate = candidates[np.argmax(metric_list)]
    if candidate.wkt not in matched_pair_dict:
        matched_pair_dict[candidate.wkt] = target
        return
    if criteria_func(reference_dict[candidate.wkt], target) > max_metric:
        matched_pair_dict[candidate.wkt] = target


def _match_by_min_criteria(
    target: Polygon,
    criteria_func: Callable,
    reference_dict: Dict[str, Polygon],
    metric_list: List[float],
    match_threshold: float,
    candidates: List[BaseGeometry],
    matched_pair_dict: Dict[str, Polygon],
) -> None:
    min_metric = min(metric_list)
    if match_threshold > 0 and min_metric <= match_threshold:
        return
    candidate = candidates[np.argmin(metric_list)]
    if candidate.wkt not in matched_pair_dict:
        matched_pair_dict[candidate.wkt] = target
        return
    if criteria_func(reference_dict[candidate.wkt], target) < min_metric:
        matched_pair_dict[candidate.wkt] = target


def _build_outputs(reference_list, target_list, reference_dict, matched_pair_dict):
    matched_dict_list = [
        {"reference": reference_dict[key], "target": value}
        for key, value in matched_pair_dict.items()
    ]
    matched_references_list = [reference_dict[key] for key in matched_pair_dict]
    unmatched_references_list = list(
        filter(lambda x: x not in matched_references_list, reference_list)
    )
    unmatched_targets_list = list(
        filter(lambda x: x not in matched_pair_dict.values(), target_list)
    )
    return matched_dict_list, unmatched_references_list, unmatched_targets_list


def match_polygon_lists_by_iou(
    reference_list: List[Polygon],
    target_list: List[Polygon],
    match_treshold: float = 0.5,
) -> Tuple[List[Dict[str, Polygon]], List[Polygon], List[Polygon]]:
    """
    Matches the reference geometry list with the target geometry list by the intersection over union (IOU) metric.
    :param reference_geometry_list: The reference geometry list.
    :param target_geometry_list: The target geometry list.
    :return: Dictionary with the matched and unmatched polygons.
    """
    return match_polygon_lists_by_criteria(
        reference_list,
        target_list,
        shapely_polygon_iou,
        match_criteria="max",
        match_threshold=match_treshold,
    )


def match_polygon_lists_by_hausdorff_distance(
    reference_list: List[Polygon],
    target_list: List[Polygon],
    match_treshold: float = -1,
) -> Tuple[List[Dict[str, Polygon]], List[Polygon], List[Polygon]]:
    """
    Matches the reference geometry list with the target geometry list by the intersection over union (IOU) metric.
    :param reference_geometry_list: The reference geometry list.
    :param target_geometry_list: The target geometry list.
    :return: Dictionary with the matched and unmatched polygons.
    """
    return match_polygon_lists_by_criteria(
        reference_list,
        target_list,
        hausdorff_distance,
        match_criteria="min",
        match_threshold=match_treshold,
    )


def match_polygon_lists_by_frechet_distance(
    reference_list: List[Polygon],
    target_list: List[Polygon],
    match_treshold: float = -1,
) -> Tuple[List[Dict[str, Polygon]], List[Polygon], List[Polygon]]:
    """
    Matches the reference geometry list with the target geometry list by the intersection over union (IOU) metric.
    :param reference_geometry_list: The reference geometry list.
    :param target_geometry_list: The target geometry list.
    :return: Dictionary with the matched and unmatched polygons.
    """
    return match_polygon_lists_by_criteria(
        reference_list,
        target_list,
        frechet_distance,
        match_criteria="min",
        match_threshold=match_treshold,
    )


def per_vertex_error_list(
    gt_polygon: Polygon, pred_polygon: Polygon
) -> List[Dict[str, Union[Point, float]]]:
    paired_dict = dict()
    gt_vertexes = list(map(Point, gt_polygon.exterior.coords))
    pred_vertexes = list(map(Point, pred_polygon.exterior.coords))
    gt_tree = STRtree(gt_vertexes)
    for pred_vertex in pred_vertexes:
        gt_vertex = gt_tree.nearest(pred_vertex)
        if gt_vertex.wkt not in paired_dict:
            paired_dict[gt_vertex.wkt] = {
                "pred_vertex": pred_vertex,
                "distance": pred_vertex.distance(gt_vertex),
            }
            continue
        if paired_dict[gt_vertex.wkt]["distance"] > pred_vertex.distance(gt_vertex):
            paired_dict[gt_vertex.wkt] = {
                "pred_vertex": pred_vertex,
                "distance": pred_vertex.distance(gt_vertex),
            }
    output_dict_list = [
        {
            "gt_vertex": shapely.wkt.loads(gt_vertex_wkt),
            "pred_vertex": paired_dict["pred_vertex"],
            "distance": paired_dict["distance"],
        }
        for gt_vertex_wkt, paired_dict in paired_dict.items()
    ]
    return output_dict_list
