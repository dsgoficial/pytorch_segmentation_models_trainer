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

import copy
from typing import Any, Callable, Dict, List
from shapely.geometry import Point
from pytorch_segmentation_models_trainer.tools.evaluation.matching import (
    per_vertex_error_list,
)


def compute_metrics_on_match_list_dict(
    match_dict_list: List[Dict[str, Any]], metric_list: List[Callable]
) -> List[Dict[str, Any]]:
    """
    Computes the metrics in metric_list between the reference and target polygons in a list of dictionaries.
    :param match_dict_list: list of dictionaries with the reference and target polygons
    :param metric_list: list of metric functions
    :return: list of dictionaries with the metrics in metric_list between the reference and target polygons
    """
    output_dict_list = []
    for match_dict in match_dict_list:
        item = match_dict.copy()
        for metric in metric_list:
            computed_metric = metric(match_dict["reference"], match_dict["target"])
            if isinstance(computed_metric, tuple) and "iou" in metric.__name__:
                item["iou"], item["intersection"], item["union"] = computed_metric
                continue
            item[metric.__name__] = computed_metric
        output_dict_list.append(item)
    return output_dict_list


def compute_vertex_errors_on_match_list_dict(
    match_dict_list: List[Dict[str, Any]], convert_output_to_wkt: bool = True
) -> List[Dict[str, Any]]:
    output_dict_list = []
    for idx, match_dict in enumerate(match_dict_list):
        per_vertex_error_dict_list = per_vertex_error_list(
            match_dict["reference"], match_dict["target"]
        )
        for error_dict in per_vertex_error_dict_list:
            new_dict = (
                copy.deepcopy(error_dict)
                if not convert_output_to_wkt
                else {
                    k: (v.wkt if isinstance(v, Point) else v)
                    for k, v in error_dict.items()
                }
            )
            new_dict["id"] = idx
            output_dict_list.append(new_dict)
    return output_dict_list
