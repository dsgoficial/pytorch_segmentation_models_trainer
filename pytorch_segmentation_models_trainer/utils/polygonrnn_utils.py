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

from collections import defaultdict
from typing import Callable, List, Optional, Union, Tuple, Dict
from PIL import Image, ImageDraw
import numpy as np
from shapely.geometry.collection import GeometryCollection
import torch
import itertools
from shapely.geometry import Polygon, LineString, Point, MultiPolygon, box
from shapely.geometry.base import BaseGeometry
import cv2
from shapely.validation import make_valid


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
    input_array: np.ndarray,
    scale_h: Optional[float] = 1.0,
    scale_w: Optional[float] = 1.0,
    min_col: Optional[int] = 0,
    min_row: Optional[int] = 0,
    grid_size: Optional[int] = 28,
    return_cast_func: Optional[Callable] = None,
) -> np.ndarray:
    """Gets vertex list from input batch.

    Args:
        input_batch (np.ndarray): [description]
        scale_h (Optional[float], optional): Height scale. Defaults to 1.0.
        scale_w (Optional[float], optional): Width scale. Defaults to 1.0.
        min_col (Optional[int], optional): Minimum column. Defaults to 0.
        min_row (Optional[int], optional): Minimun row. Defaults to 0.
    """
    return_cast_func = return_cast_func if return_cast_func is not None else lambda x: x

    def cast_to_int(x):
        return x.astype(np.int32) if isinstance(x, np.ndarray) else x.int()

    def cast_to_np(x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    input_array = cast_to_np(input_array)
    if np.max(input_array) >= grid_size**2:
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
    rescaled_array = np.zeros(polygon_np_array.shape)
    rescaled_array[:, 0] = polygon_np_array[:, 0] / scale_w + min_col
    rescaled_array[:, 1] = polygon_np_array[:, 1] / scale_h + min_row
    return Polygon(rescaled_array)


def scale_polygon_list(
    polygon_list: List[Polygon],
    list_scale_h: Union[List[float], np.ndarray],
    list_scale_w: Union[List[float], np.ndarray],
    list_min_col: Union[List[int], np.ndarray],
    list_min_row: Union[List[int], np.ndarray],
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
    input_batch: np.ndarray,
    scale_h: Optional[float] = 1.0,
    scale_w: Optional[float] = 1.0,
    min_col: Optional[int] = 0,
    min_row: Optional[int] = 0,
    grid_size: Optional[int] = 28,
) -> np.ndarray:
    """Gets vertex list from input batch.

    Args:
        input_batch (np.ndarray): [description]
        scale_h (Optional[float], optional): Height scale. Defaults to 1.0.
        scale_w (Optional[float], optional): Width scale. Defaults to 1.0.
        min_col (Optional[int], optional): Minimum column. Defaults to 0.
        min_row (Optional[int], optional): Minimun row. Defaults to 0.
    """

    def func(x):
        return get_vertex_list(
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
) -> List[np.ndarray]:
    """Gets vertex list from input batch.

    Args:
        input_batch (torch.Tensor): [description]
        scale_h (Optional[Union(float, torch.Tensor)]: Height scale. Defaults to 1.0.
        scale_w (Optional[Union(float, torch.Tensor)]: Width scale. Defaults to 1.0.
        min_col (Optional[Union(int, torch.Tensor)], optional): Minimum column. Defaults to 0.
        min_row (Optional[Union(int, torch.Tensor)], optional): Minimun row. Defaults to 0.
    """

    def cast_func(x):
        return np.array(x, dtype=np.float32)

    if input_batch == [] or (
        isinstance(input_batch, torch.Tensor) and input_batch.numel() == 0
    ):
        return []

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


def handle_vertices(vertices: Union[List, BaseGeometry]) -> BaseGeometry:
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


def validate_polygon(geom: Polygon) -> List[Union[Polygon, MultiPolygon]]:
    if geom.is_valid:
        return [geom]
    valid_output = make_valid(geom)
    if isinstance(valid_output, (Polygon, MultiPolygon)):
        return [valid_output]
    if isinstance(valid_output, (list, GeometryCollection)):
        return [p for p in valid_output if isinstance(p, (Polygon, MultiPolygon))]
    else:
        return []


def crop_polygons_to_bounding_boxes(
    polygons: List[Polygon], bounding_boxes: List
) -> List[Polygon]:
    polygons_to_crop = []
    valid_polygons = itertools.chain.from_iterable(
        list(map(validate_polygon, polygons))
    )
    for polygon in valid_polygons:
        for bbox in bounding_boxes:
            if polygon.intersects(bbox):
                polygon = polygon.intersection(bbox)
                if isinstance(polygon, Polygon):
                    polygons_to_crop.append(polygon)
                elif isinstance(polygon, MultiPolygon):
                    polygons_to_crop.extend(list(polygon))
    valid_polygons = itertools.chain.from_iterable(
        list(map(validate_polygon, polygons_to_crop))
    )
    unique_polygons = []
    for polygon in valid_polygons:
        if not any(polygon.equals(p) for p in unique_polygons) and isinstance(
            polygon, (Polygon, MultiPolygon)
        ):
            unique_polygons.append(polygon)
    return unique_polygons


def get_scales(
    min_row: int,
    min_col: int,
    max_row: int,
    max_col: int,
    target_height: float = 224.0,
    target_width: float = 224.0,
) -> tuple:
    """
    Gets scales for the image.

    Args:
        min_row (int): min row
        min_col (int): min col
        max_row (int): max row
        max_col (int): max col

    Returns:
        tuple: scale_h, scale_w
    """
    object_h = max_row - min_row
    object_w = max_col - min_col
    scale_h = target_height / object_h
    scale_w = target_width / object_w
    return scale_h, scale_w


def get_extended_bounds(
    polygon: Polygon, image_bounds: List, extend_factor: float = 0.1
) -> Tuple:
    """
    Gets extended bounds for the image.

    Args:
        polygon (Polygon): polygon
        image_bounds (List): image bounds
        extend_factor (float): extend factor

    Returns:
        Tuple: extended bounds
    """
    np_polygon = np.array(polygon.exterior.coords)
    return get_extended_bounds_from_np_array_polygon(
        np_polygon, image_bounds, extend_factor=extend_factor
    )


def get_extended_bounds_from_np_array_polygon(
    np_polygon, image_bounds, extend_factor: float = 0.1
):
    image_width, image_height = image_bounds
    min_c = np.min(np_polygon, axis=0)
    max_c = np.max(np_polygon, axis=0)
    h_extend = int(round(extend_factor * (max_c[1] - min_c[1])))
    w_extend = int(round(extend_factor * (max_c[0] - min_c[0])))
    min_row = np.maximum(0, min_c[1] - h_extend)
    min_col = np.maximum(0, min_c[0] - w_extend)
    max_row = np.minimum(image_height, max_c[1] + h_extend)
    max_col = np.minimum(image_width, max_c[0] + w_extend)
    return min_row, min_col, max_row, max_col


def get_bboxes_from_polygons(polygons: List[Polygon]) -> List[Tuple]:
    return [p.bounds for p in polygons]


def crop_and_rescale_polygons_to_bounding_boxes(
    polygons: List[Union[BaseGeometry, Polygon]],
    bounding_boxes: Union[torch.Tensor, List],
    image_bounds_list: List,
    target_height: float = 224.0,
    target_width: float = 224.0,
    extend_factor: float = 0.1,
) -> List[Dict[str, np.ndarray]]:
    """
    Crops and rescales polygons to bounding boxes.

    Args:
        polygons (List[Union[BaseGeometry, Polygon]]): polygons
        bounding_boxes (List): bounding boxes

    Returns:
        List[Dict[str, np.ndarray]]: cropped and rescaled polygons
    """
    shapely_boxes = [
        box(minx, miny, maxx, maxy, ccw=True)
        for minx, miny, maxx, maxy in bounding_boxes
    ]
    croped_polygons = crop_polygons_to_bounding_boxes(polygons, shapely_boxes)
    bboxes = (
        torch.Tensor(
            get_bboxes_from_polygons(croped_polygons),
            device=bounding_boxes.device
            if isinstance(bounding_boxes, torch.Tensor)
            else "cpu",
        )
        if len(croped_polygons) > len(bounding_boxes)
        else bounding_boxes
    )

    def extended_bounds_func(x):
        return get_extended_bounds(x[0], x[1], extend_factor)

    extended_polygons_bounds = list(
        map(extended_bounds_func, zip(croped_polygons, image_bounds_list))
    )

    def get_scales_func(x):
        return get_scales(
            min_row=x[0],
            min_col=x[1],
            max_row=x[2],
            max_col=x[3],
            target_height=target_height,
            target_width=target_width,
        )

    scales = list(map(get_scales_func, extended_polygons_bounds))

    return [
        {
            "polygon": np.array(
                [
                    [
                        np.maximum(0, np.minimum(223, (points[0] - min_col) * scale_w)),
                        np.maximum(0, np.minimum(223, (points[1] - min_row) * scale_h)),
                    ]
                    for points in polygon.exterior.coords
                ]
            ),
            "scale_w": scale_w,
            "scale_h": scale_h,
            "min_row": min_row,
            "min_col": min_col,
            "bbox": bbox,
        }
        for polygon, bbox, (scale_h, scale_w), (min_row, min_col, _, __) in zip(
            croped_polygons, bboxes, scales, extended_polygons_bounds
        )
        if scale_w
    ]


def target_list_to_dict(
    targets: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Converts a list of targets to a dictionary of targets.

    Args:
        targets (List[Dict]): list of targets

    Returns:
        Dict[str, Dict]: dictionary of targets
    """
    result_dict = defaultdict(list)
    for target in targets:
        for key, value in target.items():
            result_dict[key].append(value)
    output_result_dict = dict()
    for key, value in result_dict.items():
        value = list(itertools.chain.from_iterable(value))
        if value == []:
            output_result_dict[key] = torch.tensor([])
            continue
        output_result_dict[key] = torch.stack(value)
    return output_result_dict


def build_polygonrnn_extra_info_from_bboxes(
    bboxes: torch.Tensor, target_height: float = 224.0, target_width: float = 224.0
) -> Dict[str, torch.Tensor]:
    object_h = bboxes[:, 3] - bboxes[:, 1]
    object_w = bboxes[:, 2] - bboxes[:, 0]
    scale_h = target_height / object_h
    scale_w = target_width / object_w
    return {
        "scale_h": scale_h,
        "scale_w": scale_w,
        "min_row": bboxes[:, 1],
        "min_col": bboxes[:, 0],
    }


def get_extended_bounds_from_tensor_bbox(
    boxes: torch.Tensor, image_bounds: Tuple, extend_factor: float = 0.1
) -> torch.Tensor:
    image_rows, image_cols = image_bounds
    y1 = boxes[:, 0]  # min_row
    x1 = boxes[:, 1]  # min_col
    y2 = boxes[:, 2]  # max_row
    x2 = boxes[:, 3]  # max_col
    h_extend = (extend_factor * (y2 - y1)).int()
    w_extend = (extend_factor * (x2 - x1)).int()

    min_row = (y1 - h_extend).clamp(min=0, max=image_rows).int()
    min_col = (x1 - w_extend).clamp(min=0, max=image_cols).int()
    max_row = (y2 + h_extend).clamp(min=0, max=image_rows).int()
    max_col = (x2 + w_extend).clamp(min=0, max=image_cols).int()

    return torch.stack([min_row, min_col, max_row, max_col], dim=-1)
