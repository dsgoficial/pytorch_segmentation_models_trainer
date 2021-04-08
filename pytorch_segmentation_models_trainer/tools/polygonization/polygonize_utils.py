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
from collections import Iterable
from functools import partial

import cv2
import numpy as np
import shapely.affinity
import shapely.geometry
import shapely.ops
import skimage.measure


def compute_init_contours(np_indicator, level):
    assert isinstance(np_indicator, np.ndarray) and len(np_indicator.shape) == 2, "indicator should have shape (H, W)"
    # Using marching squares
    contours = skimage.measure.find_contours(np_indicator, level, fully_connected='low', positive_orientation='high')
    return contours


def compute_init_contours_batch(np_indicator_batch, level, pool=None):
    post_process_partial = partial(compute_init_contours, level=level)
    return pool.map(post_process_partial, np_indicator_batch) \
        if pool is not None else \
        list(map(post_process_partial, np_indicator_batch))

def split_polylines_corner(polylines, corner_masks):
    new_polylines = []
    for polyline, corner_mask in zip(polylines, corner_masks):
        splits, = np.where(corner_mask)
        if len(splits) == 0:
            new_polylines.append(polyline)
            continue
        slice_list = [(splits[i], splits[i+1] + 1) for i in range(len(splits) - 1)]
        for s in slice_list:
            new_polylines.append(polyline[s[0]:s[1]])
        # Possibly add a merged polyline if start and end vertices are not corners (or endpoints of open polylines)
        if ~corner_mask[0] and ~corner_mask[-1]:  # In fact any of those conditon should be enough
            new_polyline = np.concatenate([polyline[splits[-1]:], polyline[:splits[0] + 1]], axis=0)
            new_polylines.append(new_polyline)
    return new_polylines


def compute_geom_prob(geom, prob_map, output_debug=False):
    assert len(prob_map.shape) == 2, "prob_map should have size (H, W), not {}".format(prob_map.shape)
    if isinstance(geom, Iterable):
        return [
            compute_geom_prob(
                _geom, prob_map, output_debug=output_debug
            ) for _geom in geom
        ]
    elif isinstance(geom, shapely.geometry.Polygon):
        minx, miny, maxx, maxy = geom.bounds
        minx, miny, maxx, maxy = int(minx), int(miny), int(maxx) + 1, int(maxy) + 1
        geom = shapely.affinity.translate(geom, xoff=-minx, yoff=-miny)
        prob_map = prob_map[miny:maxy, minx:maxx]
        raster = _fill_raster(prob_map, geom)
        raster_sum = np.sum(raster)
        return np.sum(raster * prob_map) / raster_sum if raster_sum > 0 else 0
    else:
        raise NotImplementedError(f"Geometry of type {type(geom)} not implemented!")

def _fill_raster(prob_map, geom):
    raster = np.zeros(prob_map.shape, dtype=np.uint8)
    exterior_array = np.round(np.array(geom.exterior.coords)).astype(np.int32)
    interior_array_list = [
        np.round(np.array(interior.coords)).astype(np.int32) \
            for interior in geom.interiors
    ]
    cv2.fillPoly(raster, [exterior_array], color=1)
    cv2.fillPoly(raster, interior_array_list, color=0)
    return raster

