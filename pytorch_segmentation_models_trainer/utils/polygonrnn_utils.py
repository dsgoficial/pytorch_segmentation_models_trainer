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
from shapely.geometry import Polygon, LineString, Point
from shapely.geometry.base import BaseGeometry
import cv2


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
    grid_size: Optional[int] = 28,
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
                ((label % grid_size) * 8.0 + 4) / scale_w + min_col,
                ((float(label // grid_size)) * 8.0 + 4) / scale_h + min_row,
            )
            for label in itertools.takewhile(
                lambda x: x != grid_size * grid_size, input_list
            )
        ]
    )


def get_vertex_list_from_numpy(
    input_array: np.array,
    scale_h: Optional[float] = 1.0,
    scale_w: Optional[float] = 1.0,
    min_col: Optional[int] = 0,
    min_row: Optional[int] = 0,
    grid_size: Optional[int] = 28,
    return_cast_func: Optional[Callable] = None,
) -> np.array:
    """Gets vertex list from input batch.

    Args:
        input_batch (np.array): [description]
        scale_h (Optional[float], optional): Height scale. Defaults to 1.0.
        scale_w (Optional[float], optional): Width scale. Defaults to 1.0.
        min_col (Optional[int], optional): Minimum column. Defaults to 0.
        min_row (Optional[int], optional): Minimun row. Defaults to 0.
    """
    return_cast_func = return_cast_func if return_cast_func is not None else lambda x: x
    cast_to_int = lambda x: x.astype(np.int32) if isinstance(x, np.ndarray) else x.int()
    cast_to_np = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    input_array = cast_to_np(input_array)
    if np.max(input_array) >= grid_size ** 2:
        length = np.argmax(input_array)
        input_array = input_array[:length]
    if input_array.shape[0] == 0:
        return input_array
    poly = np.stack(
        [cast_to_int(input_array % grid_size), cast_to_int(input_array // grid_size)],
        axis=-1,
    )
    poly01 = poly0g_to_poly01(poly, grid_size)
    polyg = poly01_to_poly0g(poly01, 224).astype(np.float32)
    return return_cast_func(
        np.stack(
            [
                polyg[:-2][:, 0] / cast_to_np(scale_w) + cast_to_np(min_col),
                polyg[:-2][:, 1] / cast_to_np(scale_h) + cast_to_np(min_row),
            ],
            axis=-1,
        )
    )


def scale_shapely_polygon(
    polygon: Polygon, scale_h, scale_w, min_col, min_row
) -> Polygon:
    polygon_np_array = np.array(polygon.exterior.coords)
    rescaled_array = np.apply_along_axis(
        lambda x: x * scale_w + min_col, 0, polygon_np_array
    )
    rescaled_array = np.apply_along_axis(
        lambda y: y * scale_h + min_row, 1, rescaled_array
    )
    return Polygon(rescaled_array)


def scale_polygon_list(
    polygon_list: List[Polygon],
    list_scale_h: Union[List[float], np.array],
    list_scale_w: Union[List[float], np.array],
    list_min_col: Union[List[int], np.array],
    list_min_row: Union[List[int], np.array],
) -> List[Polygon]:
    return [
        scale_shapely_polygon(
            polygon,
            list_scale_h[idx],
            list_scale_w[idx],
            list_min_col[idx],
            list_min_row[idx],
        )
        for idx, polygon in enumerate(polygon_list)
    ]


def get_vertex_list_from_batch(
    input_batch: np.array,
    scale_h: Optional[float] = 1.0,
    scale_w: Optional[float] = 1.0,
    min_col: Optional[int] = 0,
    min_row: Optional[int] = 0,
    grid_size: Optional[int] = 28,
) -> np.array:
    """Gets vertex list from input batch.

    Args:
        input_batch (np.array): [description]
        scale_h (Optional[float], optional): Height scale. Defaults to 1.0.
        scale_w (Optional[float], optional): Width scale. Defaults to 1.0.
        min_col (Optional[int], optional): Minimum column. Defaults to 0.
        min_row (Optional[int], optional): Minimun row. Defaults to 0.
    """
    func = lambda x: get_vertex_list(
        x, scale_h, scale_w, min_col, min_row, grid_size=grid_size
    )
    return np.apply_along_axis(func, 1, input_batch)


def get_vertex_list_from_batch_tensors(
    input_batch: torch.Tensor,
    scale_h: torch.Tensor,
    scale_w: torch.Tensor,
    min_col: torch.Tensor,
    min_row: torch.Tensor,
    grid_size: Optional[int] = 28,
) -> List[np.array]:
    """Gets vertex list from input batch.

    Args:
        input_batch (torch.Tensor): [description]
        scale_h (Optional[Union(float, torch.Tensor)]: Height scale. Defaults to 1.0.
        scale_w (Optional[Union(float, torch.Tensor)]: Width scale. Defaults to 1.0.
        min_col (Optional[Union(int, torch.Tensor)], optional): Minimum column. Defaults to 0.
        min_row (Optional[Union(int, torch.Tensor)], optional): Minimun row. Defaults to 0.
    """
    cast_func = lambda x: np.array(x, dtype=np.float32)
    return [
        get_vertex_list_from_numpy(
            x,
            scale_h[idx],
            scale_w[idx],
            min_col[idx],
            min_row[idx],
            return_cast_func=cast_func,
            grid_size=grid_size,
        )
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


def poly01_to_poly0g(poly, grid_size):
    """
    [0, 1] coordinates to [0, grid_size] coordinates

    Note: simplification is done at a reduced scale
    """
    poly = np.floor(poly * grid_size).astype(np.int32)
    # poly = cv2.approxPolyDP(poly, 0, False)[:, 0, :]

    return poly


def poly0g_to_poly01(polygon, grid_side):
    """
    [0, grid_side] coordinates to [0, 1].
    Note: we add 0.5 to the vertices so that the points
    lie in the middle of the cell.
    """
    result = (polygon.astype(np.float32) + 0.5) / grid_side

    return result


def build_arrays(polygon, num_vertexes, sequence_length, grid_size=28):
    point_count = 2
    label_array = np.zeros([sequence_length, grid_size * grid_size + 3])
    label_index_array = np.zeros([sequence_length])
    polygon = poly0g_to_poly01(polygon, 224)
    polygon = poly01_to_poly0g(polygon, grid_size)
    if num_vertexes < sequence_length - 3:
        for points in polygon:
            _initialize_label_index_array(
                point_count, label_array, label_index_array, points, grid_size=grid_size
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
                point_count, label_array, label_index_array, points, grid_size=grid_size
            )
            point_count += 1
        for kkk in range(point_count, sequence_length):
            index = grid_size * grid_size
            label_array[kkk, index] = 1
            label_index_array[kkk] = index
    return label_array, label_index_array


def _populate_label_index_array(
    polygon,
    num_vertexes,
    sequence_length,
    point_count,
    label_array,
    label_index_array,
    grid_size=28,
):
    label_array[point_count, grid_size * grid_size] = 1
    label_index_array[point_count] = grid_size * grid_size
    for kkk in range(point_count + 1, sequence_length):
        if kkk % (num_vertexes + 3) == num_vertexes + 2:
            index = grid_size * grid_size
        elif kkk % (num_vertexes + 3) == 0:
            index = grid_size * grid_size + 1
        elif kkk % (num_vertexes + 3) == 1:
            index = grid_size * grid_size + 2
        else:
            index_a = polygon[kkk % (num_vertexes + 3) - 2][0]
            index_b = polygon[kkk % (num_vertexes + 3) - 2][1]
            index = index_b * grid_size + index_a
        label_array[kkk, index] = 1
        label_index_array[kkk] = index


def _initialize_label_index_array(
    point_count, label_array, label_index_array, points, grid_size=28
):
    # index_a = int(points[0] / 8)
    # index_b = int(points[1] / 8)
    # index = index_b * grid_size + index_a
    index = points[0] + points[1] * grid_size
    label_array[point_count, index] = 1
    label_index_array[point_count] = index


def handle_vertices(vertices):
    if isinstance(vertices, BaseGeometry):
        return vertices
    vertices_array = np.array(vertices)
    if vertices_array.shape[0] == 0:
        return Point(0, 0)
    if vertices_array.shape[0] == 1:
        return Point(vertices_array.squeeze(0))
    if vertices_array.shape[0] == 2:
        return LineString(vertices_array)
    return Polygon(vertices_array.reshape(-1, 2))
