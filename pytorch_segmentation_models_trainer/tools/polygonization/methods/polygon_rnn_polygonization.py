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

import itertools
from typing import List, Union

import numpy as np
import torch
from pytorch_segmentation_models_trainer.utils import polygonrnn_utils
from pytorch_segmentation_models_trainer.utils.polygonrnn_utils import validate_polygon
from shapely.geometry import MultiPolygon, Polygon


def simplify(
    polygons: List[Union[Polygon, MultiPolygon]], tolerance: float
) -> List[Union[Polygon, MultiPolygon]]:
    out_polygons = [
        polygon.simplify(tolerance, preserve_topology=True) for polygon in polygons
    ]
    return out_polygons


def shapely_postprocess(
    polygons: List[Union[Polygon, MultiPolygon]], config
) -> List[Union[Polygon, MultiPolygon]]:
    filtered_polygons = [
        geom
        for geom in itertools.chain.from_iterable(
            [validate_polygon(polygon) for polygon in polygons]
        )
        if geom.area >= config.min_area
    ]
    if config.tolerance == 0:
        return filtered_polygons
    polygons = simplify(filtered_polygons, config.tolerance)
    return polygons


def polygonize(
    batch: Union[torch.Tensor, np.ndarray], config, pool=None
) -> List[Union[Polygon, MultiPolygon]]:
    predicted_polygon_list = polygonrnn_utils.get_vertex_list_from_batch_tensors(
        batch["output_batch_polygons"],
        batch["scale_h"],
        batch["scale_w"],
        batch["min_col"],
        batch["min_row"],
        grid_size=config.grid_size,
    )
    return shapely_postprocess(
        list(map(Polygon, filter(lambda x: x.shape[0] > 2, predicted_polygon_list))),
        config,
    )
