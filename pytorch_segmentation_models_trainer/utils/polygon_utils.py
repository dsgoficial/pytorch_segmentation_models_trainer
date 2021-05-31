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
import math

import cv2 as cv
import numpy as np
import shapely
import skimage
import skimage.morphology
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon


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
    item_list = []
    for polygon in polygons:
        item_list += polygon.geoms if polygon.geom_type == 'MultiPolygon' else [polygon]
    return [
        np.array([~transform * point for point in np.array(polygon.exterior.coords)]) for polygon in item_list
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
    return array.transpose()

def compute_raster_masks(polygons, shape, transform, fill=True, edges=True, vertices=True, compute_distances=True, compute_sizes=True, line_width=3, antialiasing=False):
    """
    Returns:
         - distances: sum of distance to closest and second-closest annotation for each pixel.
         - size_weights: relative size (normalized by image area) of annotation the pixel belongs to.
    """
    assert type(polygons) == list, "polygons should be a list"

    # Filter out zero-area polygons
    polygons = [polygon for polygon in polygons if polygon.area > 0]
    channel_count = fill + edges + vertices
    polygons_raster = np.zeros((*shape, channel_count), dtype=np.uint8)
    distance_maps = np.ones((*shape, len(polygons)))  # Init with max value (distances are normed)
    sizes = np.ones(shape)  # Init with max value (sizes are normed)
    image_area = shape[0] * shape[1]
    for i, polygon in enumerate(polygons):
        if polygon.geom_type == 'Polygon':
            _process_polygon(polygon, shape, transform, fill, edges, vertices, line_width,\
                antialiasing, polygons_raster, distance_maps, sizes, image_area, i)
        else:
            for single_polygon in polygon.geoms:
                _process_polygon(
                    single_polygon, shape, transform, fill, edges, vertices,\
                    line_width, antialiasing, polygons_raster, distance_maps,\
                    sizes, image_area, i
                )

    polygons_raster = np.clip(polygons_raster, 0, 255)
    # skimage.io.imsave("polygons_raster.png", polygons_raster)

    if edges:
        _compute_edges(fill, edges, polygons_raster, line_width)

    distances = _compute_distances(distance_maps)
    distances = distances.astype(np.float32)
    sizes = sizes.astype(np.float32)
    return_dict = {
       key:mask.transpose() for mask, key in zip(
           np.swapaxes(polygons_raster, -1, 0),
           ["polygon_masks", "boundary_masks", "vertex_masks"]
        )
    }
    if compute_distances:
        return_dict["distance_masks"] = distances
    if compute_sizes:
        return_dict["size_masks"] = sizes
    return return_dict

def _process_polygon(polygon, shape, transform, fill, edges, vertices, line_width,\
    antialiasing, polygons_raster, distance_maps, sizes, image_area, i):
    polygon = shapely.geometry.Polygon(np.array(
            [~transform * point for point in np.array(polygon.exterior.coords)]
        ))
    mini, minj, maxi, maxj = _compute_raster_bounds_coods(polygon, polygons_raster, line_width)
    bbox_shape = (maxi - mini, maxj - minj)
    bbox_polygon = shapely.affinity.translate(polygon, xoff=-minj, yoff=-mini)
    bbox_raster = _draw_polygons([bbox_polygon], bbox_shape, fill, edges, vertices, line_width, antialiasing)
    polygons_raster[mini:maxi, minj:maxj] = np.maximum(polygons_raster[mini:maxi, minj:maxj], bbox_raster)
    bbox_mask = np.sum(bbox_raster, axis=2) > 0
    # Polygon interior + edge + vertexif bbox_mask.max():  # Make sure mask is not empty
    _compute_distance_and_sizes(
            i, distance_maps, sizes, polygon, image_area, shape, mini, maxi, minj, maxj, bbox_mask, line_width)

def _compute_raster_bounds_coods(polygon, polygons_raster, line_width):
    minx, miny, maxx, maxy = polygon.bounds
    mini = max(0, math.floor(miny) - 2*line_width)
    minj = max(0, math.floor(minx) - 2*line_width)
    maxi = min(polygons_raster.shape[0], math.ceil(maxy) + 2*line_width)
    maxj = min(polygons_raster.shape[1], math.ceil(maxx) + 2*line_width)
    return mini, minj, maxi, maxj

def _compute_distance_and_sizes(i, distance_maps, sizes, polygon, image_area, shape, mini, maxi, minj, maxj, bbox_mask, line_width):
    polygon_mask = np.zeros(shape, dtype=np.bool)
    polygon_mask[mini:maxi, minj:maxj] = bbox_mask
    polygon_dist = cv.distanceTransform(1 - polygon_mask.astype(np.uint8), distanceType=cv.DIST_L2, maskSize=cv.DIST_MASK_5,
                                dstType=cv.CV_64F)
    polygon_dist /= (polygon_mask.shape[0] + polygon_mask.shape[1])  # Normalize dist
    distance_maps[:, :, i] = polygon_dist

    selem = skimage.morphology.disk(line_width)
    bbox_dilated_mask = skimage.morphology.binary_dilation(bbox_mask, selem=selem)
    sizes[mini:maxi, minj:maxj][bbox_dilated_mask] = polygon.area / image_area

def _compute_edges(fill, edges, polygons_raster, line_width):
    edge_channels = -1 + fill + edges
    # Remove border edges because they correspond to cut buildings:
    polygons_raster[:line_width, :, edge_channels] = 0
    polygons_raster[-line_width:, :, edge_channels] = 0
    polygons_raster[:, :line_width, edge_channels] = 0
    polygons_raster[:, -line_width:, edge_channels] = 0


def _compute_distances(distance_maps):
    distance_maps.sort(axis=2)
    distance_maps = distance_maps[:, :, :2]
    distances = np.sum(distance_maps, axis=2)
    return distances

def _draw_polygons(polygons, shape, fill=True, edges=True, vertices=True, line_width=3, antialiasing=False):
    assert type(polygons) == list, "polygons should be a list"
    assert type(polygons[0]) == shapely.geometry.Polygon, "polygon should be a shapely.geometry.Polygon"

    if antialiasing:
        draw_shape = (2 * shape[0], 2 * shape[1])
        polygons = [shapely.affinity.scale(polygon, xfact=2.0, yfact=2.0, origin=(0, 0)) for polygon in polygons]
        line_width *= 2
    else:
        draw_shape = shape
    # Channels
    fill_channel_index = 0  # Always first channel
    edges_channel_index = fill  # If fill == True, take second channel. If not then take first
    vertices_channel_index = fill + edges  # Same principle as above
    channel_count = fill + edges + vertices
    im_draw_list = []
    for channel_index in range(channel_count):
        im = Image.new("L", (draw_shape[1], draw_shape[0]))
        im_px_access = im.load()
        draw = ImageDraw.Draw(im)
        im_draw_list.append((im, draw))

    for polygon in polygons:
        if fill:
            draw = im_draw_list[fill_channel_index][1]
            draw.polygon(polygon.exterior.coords, fill=255)
            for interior in polygon.interiors:
                draw.polygon(interior.coords, fill=0)
        if edges:
            draw = im_draw_list[edges_channel_index][1]
            draw.line(polygon.exterior.coords, fill=255, width=line_width)
            for interior in polygon.interiors:
                draw.line(interior.coords, fill=255, width=line_width)
        if vertices:
            draw = im_draw_list[vertices_channel_index][1]
            for vertex in polygon.exterior.coords:
                _draw_circle(draw, vertex, line_width / 2, fill=255)
            for interior in polygon.interiors:
                for vertex in interior.coords:
                    _draw_circle(draw, vertex, line_width / 2, fill=255)

    im_list = []
    if antialiasing:
        # resize images:
        for im_draw in im_draw_list:
            resize_shape = (shape[1], shape[0])
            im_list.append(im_draw[0].resize(resize_shape, Image.BILINEAR))
    else:
        for im_draw in im_draw_list:
            im_list.append(im_draw[0])

    # Convert image to numpy array with the right number of channels
    array_list = [np.array(im) for im in im_list]
    array = np.stack(array_list, axis=-1)
    return array
