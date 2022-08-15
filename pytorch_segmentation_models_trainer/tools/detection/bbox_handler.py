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
from typing import Dict, List, Tuple, Union
import numpy as np
from pytorch_toolbelt.inference.tiles import TileMerger

import torch
import torchvision
from sahi.postprocess.legacy.combine import NMSPostprocess, UnionMergePostprocess
from sahi.prediction import ObjectPrediction


def shift_bbox(
    bbox: Union[List, np.ndarray], row_shift, column_shift
) -> List[Union[int, np.ndarray]]:
    """
    Shifts a bounding box by a given amount.
    :param bbox: Bounding box to be shifted.
    :param row_shift: Amount to shift the bounding box in the row.
    :param column_shift: Amount to shift the bounding box in column.
    :return: Shifted bounding box.
    """
    if isinstance(bbox, list):
        return [
            bbox[0] + row_shift,
            bbox[1] + column_shift,
            bbox[2] + row_shift,
            bbox[3] + column_shift,
        ]
    elif isinstance(bbox, np.ndarray):
        return bbox + np.array([row_shift, column_shift, row_shift, column_shift])
    elif isinstance(bbox, torch.Tensor):
        return bbox + torch.tensor([row_shift, column_shift, row_shift, column_shift])
    else:
        raise TypeError("Bounding box must be a list, tensor or numpy array")


def crop_bboxes(
    bboxes: torch.Tensor, image_height: int, image_width: int
) -> torch.Tensor:
    """
    Crops bounding boxes to a given size.
    :param bboxes: List of bounding boxes to be cropped.
    :param image_height: Height of the image.
    :param image_width: Width of the image.
    :return: Cropped bounding boxes.
    """
    bboxes = bboxes.clone()
    bboxes[:, 0] = bboxes[:, 0].clamp(min=0, max=image_height)
    bboxes[:, 1] = bboxes[:, 1].clamp(min=0, max=image_width)
    bboxes[:, 2] = bboxes[:, 2].clamp(min=0, max=image_height)
    bboxes[:, 3] = bboxes[:, 3].clamp(min=0, max=image_width)
    return bboxes


def filter_degenerated_bboxes(
    bboxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> List[Dict[str, torch.Tensor]]:
    """
    Filters out degenerated bounding boxes.
    :param boxes: List of bounding boxes.
    :param scores: List of bounding box scores.
    :param labels: List of bounding box labels.
    :return: List of bounding boxes after filtering.
    """
    return [
        {"bboxes": box, "scores": score, "labels": label}
        for box, score, label in zip(bboxes, scores, labels)
        if not (box[0] == box[2] or box[1] == box[3])
    ]


def nms_postprocess(
    bboxes: torch.Tensor,
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
    object_prediction_list = _build_object_prediction_list(bboxes, scores, labels)
    selected_object_predictions = NMSPostprocess(
        match_threshold=threshold, match_metric=match_metric
    )(object_prediction_list)
    return _build_return_tuple_with_selected_object_predictions(
        selected_object_predictions
    )


def union_merge_postprocess(
    bboxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor]:
    object_prediction_list = _build_object_prediction_list(bboxes, scores, labels)
    selected_object_predictions = UnionMergePostprocess()(object_prediction_list)
    return _build_return_tuple_with_selected_object_predictions(
        selected_object_predictions
    )


def _build_return_tuple_with_selected_object_predictions(
    selected_object_predictions: List[ObjectPrediction],
) -> Tuple[torch.Tensor]:
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
    bboxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> List[ObjectPrediction]:
    object_prediction_list = [
        ObjectPrediction(
            bbox=box.int().tolist(), score=score.item(), category_id=label.item()
        )
        for box, score, label in zip(bboxes, scores, labels)
    ]
    return object_prediction_list


class BboxTileMerger(TileMerger):
    """
    Merges bounding boxes from predictions done in tiles.
    """

    def __init__(
        self,
        image_shape: Tuple,
        post_process_method: str = "union",
        threshold: float = 0.5,
        match_metric: str = "IOU",
        device: str = "cpu",
        dtype: torch.dtype = torch.int32,
    ):
        assert post_process_method in ["union", "nms"], "Invalid post process method."
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.device = device
        self.dtype = dtype
        self.model_output_list = []
        self.threshold = threshold
        self.match_metric = match_metric
        self.post_process_method = post_process_method

    def post_process_func(
        self, bboxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        return (
            union_merge_postprocess(bboxes, scores, labels)
            if self.post_process_method == "union"
            else nms_postprocess(
                bboxes, scores, labels, self.threshold, self.match_metric
            )
        )

    def integrate_boxes(
        self, model_output: List[Dict[str, torch.Tensor]], crop_coords: np.ndarray
    ) -> None:
        """Integrates bboxes from model_output, which is a list of dictionaries
        in the format:
        {
            'boxes': tensor([], size=(0, 4)),
            'labels': tensor([], dtype=torch.int64),
            'scores': tensor([])
        }

        Args:
            model_output (List[Dict[str, torch.Tensor]]): outputs of the model
            crop_coords (np.ndarray): crop coordinates
        """
        integrate_list = []
        for idx, output in enumerate(model_output):
            out = output.copy()
            out["bboxes"] = shift_bbox(
                out["bboxes"],
                row_shift=crop_coords[idx][0],
                column_shift=crop_coords[idx][1],
            )
            integrate_list.append(out)
        self.model_output_list.extend(integrate_list)

    def merge(self) -> List[Dict[str, torch.Tensor]]:
        """Merges accumulated bboxes and crop them to image shape.

        Returns:
            torch.Tensor: [description]
        """
        outputs = {
            key: torch.concat([item[key] for item in self.model_output_list])
            for key in self.model_output_list[0].keys()
        }
        boxes, scores, labels = self.post_process_func(**outputs)
        boxes = crop_bboxes(boxes, self.image_height, self.image_width)
        return filter_degenerated_bboxes(boxes, scores, labels)

    def accumulate_single(self, tile: torch.Tensor, coords: np.ndarray) -> None:
        raise NotImplementedError("This method is not implemented.")
