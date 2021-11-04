# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-08-03
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
 *   Part of the code is from                                              *
 *   https://github.com/AlexMa011/pytorch-polygon-rnn                      *
 ****
"""

from typing import Callable, List, Optional, Union
from PIL import Image, ImageDraw
import numpy as np
import torch
import itertools


def label2vertex(labels):
    """
    convert 1D labels to 2D vertices coordinates
    :param labels: 1D labels
    :return: 2D vertices coordinates: [(x1, y1),(x2,y2),...]
    """
    vertices = []
    for label in labels:
        if label == 784:
            break
        vertex = ((label % 28) * 8, (label / 28) * 8)
        vertices.append(vertex)
    return vertices


def get_vertex_list(
    input_list: List[float],
    scale_h: Optional[float] = 1.0,
    scale_w: Optional[float] = 1.0,
    min_col: Optional[int] = 0,
    min_row: Optional[int] = 0,
    return_cast_func: Optional[Callable] = None,
) -> List[float]:
    """Gets vertex list from input.

    Args:
        input_list (List[float]): [description]
        scale_h (Optional[float], optional): Height scale. Defaults to 1.0.
        scale_w (Optional[float], optional): Width scale. Defaults to 1.0.
        min_col (Optional[int], optional): Minimum column. Defaults to 0.
        min_row (Optional[int], optional): Minimun row. Defaults to 0.

    Returns:
        List[float]: List of the vertexes
    """
    return_cast_func = return_cast_func if return_cast_func is not None else lambda x: x
    return return_cast_func(
        [
            (
                ((label % 28) * 8.0 + 4) / scale_w + min_col,
                ((float(label // 28)) * 8.0 + 4) / scale_h + min_row,
            )
            for label in itertools.takewhile(lambda x: x != 784, input_list)
        ]
    )


def get_vertex_list_from_batch(
    input_batch: np.array,
    scale_h: Optional[float] = 1.0,
    scale_w: Optional[float] = 1.0,
    min_col: Optional[int] = 0,
    min_row: Optional[int] = 0,
) -> np.array:
    """Gets vertex list from input batch.

    Args:
        input_batch (np.array): [description]
        scale_h (Optional[float], optional): Height scale. Defaults to 1.0.
        scale_w (Optional[float], optional): Width scale. Defaults to 1.0.
        min_col (Optional[int], optional): Minimum column. Defaults to 0.
        min_row (Optional[int], optional): Minimun row. Defaults to 0.
    """
    func = lambda x: get_vertex_list(x, scale_h, scale_w, min_col, min_row)
    return np.apply_along_axis(func, 1, input_batch)


def get_vertex_list_from_batch_tensors(
    input_batch: torch.Tensor,
    scale_h: torch.Tensor,
    scale_w: torch.Tensor,
    min_col: torch.Tensor,
    min_row: torch.Tensor,
) -> List[np.array]:
    """Gets vertex list from input batch.

    Args:
        input_batch (torch.Tensor): [description]
        scale_h (Optional[Union(float, torch.Tensor)]: Height scale. Defaults to 1.0.
        scale_w (Optional[Union(float, torch.Tensor)]: Width scale. Defaults to 1.0.
        min_col (Optional[Union(int, torch.Tensor)], optional): Minimum column. Defaults to 0.
        min_row (Optional[Union(int, torch.Tensor)], optional): Minimun row. Defaults to 0.
    """
    cast_func = lambda x: torch.tensor(
        x, dtype=torch.float32, device=input_batch.device
    )
    return [
        get_vertex_list(
            x,
            scale_h[idx],
            scale_w[idx],
            min_col[idx],
            min_row[idx],
            return_cast_func=cast_func,
        )
        .cpu()
        .numpy()
        for idx, x in enumerate(torch.unbind(input_batch, dim=0))
    ]


def getbboxfromkps(kps, h, w):
    """

    :param kps:
    :return:
    """
    min_c = np.min(np.array(kps), axis=0)
    max_c = np.max(np.array(kps), axis=0)
    object_h = max_c[1] - min_c[1]
    object_w = max_c[0] - min_c[0]
    h_extend = int(round(0.1 * object_h))
    w_extend = int(round(0.1 * object_w))
    min_row = np.maximum(0, min_c[1] - h_extend)
    min_col = np.maximum(0, min_c[0] - w_extend)
    max_row = np.minimum(h, max_c[1] + h_extend)
    max_col = np.minimum(w, max_c[0] + w_extend)
    return (min_row, min_col, max_row, max_col)


def img2tensor(img):
    """

    :param img:
    :return:
    """
    img = np.rollaxis(img, 2, 0)
    return torch.from_numpy(img)


def tensor2img(tensor):
    """

    :param tensor:
    :return:
    """
    img = (tensor.numpy() * 255).astype("uint8")
    img = np.rollaxis(img, 0, 3)
    return img


def build_arrays(polygon, num_vertexes, sequence_length):
    point_count = 2
    label_array = np.zeros([sequence_length, 28 * 28 + 3])
    label_index_array = np.zeros([sequence_length])
    if num_vertexes < sequence_length - 3:
        for points in polygon:
            _initialize_label_index_array(
                point_count, label_array, label_index_array, points
            )
            point_count += 1
        _populate_label_index_array(
            polygon,
            num_vertexes,
            sequence_length,
            point_count,
            label_array,
            label_index_array,
        )
    else:
        scale = num_vertexes * 1.0 / (sequence_length - 3)
        index_list = (np.arange(0, sequence_length - 3) * scale).astype(int)
        for points in polygon[index_list]:
            _initialize_label_index_array(
                point_count, label_array, label_index_array, points
            )
            point_count += 1
        for kkk in range(point_count, sequence_length):
            index = 28 * 28
            label_array[kkk, index] = 1
            label_index_array[kkk] = index
    return label_array, label_index_array


def _populate_label_index_array(
    polygon, num_vertexes, sequence_length, point_count, label_array, label_index_array
):
    label_array[point_count, 28 * 28] = 1
    label_index_array[point_count] = 28 * 28
    for kkk in range(point_count + 1, sequence_length):
        if kkk % (num_vertexes + 3) == num_vertexes + 2:
            index = 28 * 28
        elif kkk % (num_vertexes + 3) == 0:
            index = 28 * 28 + 1
        elif kkk % (num_vertexes + 3) == 1:
            index = 28 * 28 + 2
        else:
            index_a = int(polygon[kkk % (num_vertexes + 3) - 2][0] / 8)
            index_b = int(polygon[kkk % (num_vertexes + 3) - 2][1] / 8)
            index = index_b * 28 + index_a
        label_array[kkk, index] = 1
        label_index_array[kkk] = index


def _initialize_label_index_array(point_count, label_array, label_index_array, points):
    index_a = int(points[0] / 8)
    index_b = int(points[1] / 8)
    index = index_b * 28 + index_a
    label_array[point_count, index] = 1
    label_index_array[point_count] = index
