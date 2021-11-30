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

from typing import Any, Callable, Dict, List


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
