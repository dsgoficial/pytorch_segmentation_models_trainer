# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-05-08
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
import numpy as np
from PIL import Image, ImageDraw

from shapely.geometry import (MultiPolygon,
                              Polygon)

def polygon_remove_holes(polygon):
    return np.array(polygon.exterior.coords)

def polygons_remove_holes(polygons):
    gt_polygons_no_holes = []
    for polygon in polygons:
        gt_polygons_no_holes.append(polygon_remove_holes(polygon))
    return gt_polygons_no_holes

def _draw_circle(draw, center, radius, fill):
    draw.ellipse([center[0] - radius,
                  center[1] - radius,
                  center[0] + radius,
                  center[1] + radius], fill=fill, outline=None)

def polygons_to_pixel_coords(polygons, transform):
    return [
        np.array([~transform * point for point in np.array(polygon.exterior.coords)]) for polygon in polygons
    ]

def build_crossfield(polygons, shape, transform, line_width=2):
    """
    Angle field {\theta_1} the tangent vector's angle for every pixel, specified on the polygon edges.
    Angle between 0 and pi.
    This is not invariant to symmetries.

    :param polygons:
    :param shape:
    :return: (angles: np.array((num_edge_pixels, ), dtype=np.uint8),
              mask: np.array((num_edge_pixels, 2), dtype=np.int))
    """
    assert type(polygons) == list, "polygons should be a list"

    # polygons = polygons_remove_holes(polygons)
    polygons = polygons_to_pixel_coords(polygons, transform)

    im = Image.new("L", (shape[1], shape[0]))
    im_px_access = im.load()
    draw = ImageDraw.Draw(im)

    for polygon in polygons:
        # --- edges:
        edge_vect_array = np.diff(polygon, axis=0)
        edge_angle_array = np.angle(edge_vect_array[:, 0] + 1j * edge_vect_array[:, 1])
        neg_indices = np.where(edge_angle_array < 0)
        edge_angle_array[neg_indices] += np.pi

        for i in range(polygon.shape[0] - 1):
            edge = (polygon[i], polygon[i + 1])
            angle = edge_angle_array[i]
            uint8_angle = int((255 * angle / np.pi).round())
            line = [(edge[0][1], edge[0][0]), (edge[1][1], edge[1][0])]
            draw.line(line, fill=uint8_angle, width=line_width)
            _draw_circle(draw, line[0], radius=line_width / 2, fill=uint8_angle)
        _draw_circle(draw, line[1], radius=line_width / 2, fill=uint8_angle)

    # Convert image to numpy array
    array = np.array(im)
    return array
