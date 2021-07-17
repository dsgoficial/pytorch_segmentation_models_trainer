# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-03-30
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
 *   Code inspired by the one in                                           *
 *   https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning/     *
 ****
"""

from functools import partial

import shapely.geometry
import shapely.ops
import skimage.io
import skimage.measure
from pytorch_segmentation_models_trainer.tools.polygonization import \
    polygonize_utils

def simplify(polygons, probs, tolerance):
    if isinstance(tolerance, list):
        out_polygons_dict, out_probs_dict = {}, {}
        for tol in tolerance:
            out_polygons, out_probs = simplify(polygons, probs, tol)
            out_polygons_dict["tol_{}".format(tol)] = out_polygons
            out_probs_dict["tol_{}".format(tol)] = out_probs
        return out_polygons_dict, out_probs_dict
    else:
        out_polygons = [
            polygon.simplify(
                tolerance, preserve_topology=True
            ) for polygon in polygons
        ]
        return out_polygons, probs


def shapely_postprocess(out_contours, np_indicator, config):
    height, width = np_indicator.shape[0], np_indicator.shape[1]

    line_string_list = [
        shapely.geometry.LineString(out_contour[:, ::-1]) \
            for out_contour in out_contours
    ]

    line_string_list.append(
        shapely.geometry.LinearRing([
            (0, 0),
            (0, height - 1),
            (width - 1, height - 1),
            (width - 1, 0),
        ]))

    multi_line_string = shapely.ops.unary_union(line_string_list)
    polygons, dangles, cuts, invalids = shapely.ops.polygonize_full(multi_line_string)
    polygons = [polygon for polygon in list(polygons) if
                config.min_area < polygon.area]

    filtered_polygons, filtered_polygon_probs = [], []
    for polygon in polygons:
        prob = polygonize_utils.compute_geom_prob(polygon, np_indicator)
        if config.seg_threshold < prob:
            filtered_polygons.append(polygon)
            filtered_polygon_probs.append(prob)

    polygons, probs = simplify(filtered_polygons, filtered_polygon_probs, config.tolerance)
    return polygons, probs


def polygonize(seg_batch, config, pool=None, pre_computed=None):

    assert len(seg_batch.shape) == 4 and seg_batch.shape[
        1] <= 3, "seg_batch should be (N, C, H, W) with C <= 3, not {}".format(seg_batch.shape)

    indicator_batch = seg_batch[:, 0, :, :]
    np_indicator_batch = indicator_batch.cpu().numpy()

    init_contours_batch = polygonize_utils.compute_init_contours_batch(
        np_indicator_batch, config.data_level, pool=pool
    ) if pre_computed is None or "init_contours_batch" not in pre_computed \
        else pre_computed["init_contours_batch"]

    if pool is not None:
        shapely_postprocess_partial = partial(shapely_postprocess, config=config)
        polygons_probs_batch = pool.starmap(shapely_postprocess_partial, zip(init_contours_batch, np_indicator_batch))
        polygons_batch, probs_batch = zip(*polygons_probs_batch)
    else:
        polygons_batch, probs_batch = [], []
        for i, out_contours in enumerate(init_contours_batch):
            polygons, probs = shapely_postprocess(out_contours, np_indicator_batch[i], config)
            polygons_batch.append(polygons)
            probs_batch.append(probs)
    return polygons_batch, probs_batch
