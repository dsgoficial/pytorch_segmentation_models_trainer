# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-11-18
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
from typing import List, Tuple

import torch
import torchvision
from sahi.postprocess.combine import NMSPostprocess, UnionMergePostprocess
from sahi.prediction import ObjectPrediction


def shift_bbox(bbox: List[int], x_shift, y_shift) -> List[int]:
    """
    Shifts a bounding box by a given amount.
    :param bbox: Bounding box to be shifted.
    :param x_shift: Amount to shift the bounding box in x axis.
    :param y_shift: Amount to shift the bounding box in y axis.
    :return: Shifted bounding box.
    """
    return [bbox[0] + x_shift, bbox[1] + y_shift, bbox[2] + x_shift, bbox[3] + y_shift]


def nms_postprocess(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    match_metric="IOU",
) -> Tuple[torch.Tensor]:
    """
    Performs non-maximum suppression on a list of bounding boxes.
    :param bboxes: List of bounding boxes.
    :param scores: List of bounding box scores.
    :param threshold: Threshold to apply for NMS.
    :return: List of bounding boxes after NMS.
    """
    object_prediction_list = _build_object_prediction_list(boxes, scores, labels)
    selected_object_predictions = NMSPostprocess(
        match_threshold=threshold, match_metric=match_metric
    )(object_prediction_list)
    return _build_return_tuple_with_selected_object_predictions(
        selected_object_predictions
    )


def union_merge_postprocess(
    boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor]:
    object_prediction_list = _build_object_prediction_list(boxes, scores, labels)
    selected_object_predictions = UnionMergePostprocess()(object_prediction_list)
    return _build_return_tuple_with_selected_object_predictions(
        selected_object_predictions
    )


def _build_return_tuple_with_selected_object_predictions(selected_object_predictions):
    return (
        torch.tensor(
            [
                [v for k, v in obj.bbox.__dict__.items() if "_" not in k]
                for obj in selected_object_predictions
            ]
        ),
        torch.tensor([obj.score.value for obj in selected_object_predictions]),
        torch.tensor([obj.category.id for obj in selected_object_predictions]),
    )


def _build_object_prediction_list(
    boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> List[ObjectPrediction]:
    object_prediction_list = [
        ObjectPrediction(
            bbox=box.int().tolist(), score=score.item(), category_id=label.item()
        )
        for box, score, label in zip(boxes, scores, labels)
    ]

    return object_prediction_list
