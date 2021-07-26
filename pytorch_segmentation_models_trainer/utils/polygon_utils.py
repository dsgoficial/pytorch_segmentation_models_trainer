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
from functools import partial
import math

import cv2 as cv
import numpy as np
import pyproj
import random
import shapely
import skimage
import skimage.morphology
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from tqdm import tqdm
from matplotlib.collections import PatchCollection
import multiprocess


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

def polygons_to_world_coords(polygons, transform, epsg_number):
    item_list = []
    for polygon in polygons:
        item_list += polygon.geoms if polygon.geom_type == 'MultiPolygon' else [polygon]
    return [
        shapely.geometry.asPolygon(
            np.array([transform * point for point in np.array(polygon.exterior.coords)])
        ) for polygon in item_list
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
    polygon_mask = np.zeros(shape, dtype=np.bool_)
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

def compute_polygon_contour_measures(pred_polygons: list, gt_polygons: list, sampling_spacing: float, min_precision: float, max_stretch: float, metric_name: str="cosine", progressbar=False):
    """
    pred_polygons are sampled with sampling_spacing before projecting those sampled points to gt_polygons.
    Then the

    @param pred_polygons:
    @param gt_polygons:
    @param sampling_spacing:
    @param min_precision: Polygons in pred_polygons must have a precision with gt_polygons above min_precision to be included in further computations
    @param max_stretch:  Exclude edges that have been stretched by the projection more than max_stretch from further computation
    @param metric_name: Metric type, can be "cosine" or ...
    @return:
    """
    assert isinstance(pred_polygons, list), "pred_polygons should be a list"
    assert isinstance(gt_polygons, list), "gt_polygons should be a list"
    if len(pred_polygons) == 0 or len(gt_polygons) == 0:
        return np.array([]), [], []
    assert isinstance(pred_polygons[0], shapely.geometry.Polygon), \
        f"Items of pred_polygons should be of type shapely.geometry.Polygon, not {type(pred_polygons[0])}"
    assert isinstance(gt_polygons[0], shapely.geometry.Polygon), \
        f"Items of gt_polygons should be of type shapely.geometry.Polygon, not {type(gt_polygons[0])}"
    gt_polygons = shapely.geometry.collection.GeometryCollection(gt_polygons)
    pred_polygons = shapely.geometry.collection.GeometryCollection(pred_polygons)
    # Filter pred_polygons to have at least a precision with gt_polygons of min_precision
    filtered_pred_polygons = [pred_polygon for pred_polygon in pred_polygons if min_precision < pred_polygon.intersection(gt_polygons).area / pred_polygon.area]
    # Extract contours of gt polygons
    gt_contours = shapely.geometry.collection.GeometryCollection([contour for polygon in gt_polygons for contour in [polygon.exterior, *polygon.interiors]])
    # Measure metric for each pred polygon
    if progressbar:
        process_id = int(multiprocess.current_process().name[-1])
        iterator = tqdm(filtered_pred_polygons, desc="Contour measure", leave=False, position=process_id)
    else:
        iterator = filtered_pred_polygons
    half_tangent_max_angles = [compute_contour_measure(pred_polygon, gt_contours, sampling_spacing=sampling_spacing, max_stretch=max_stretch, metric_name=metric_name)
                               for pred_polygon in iterator]
    return half_tangent_max_angles

def compute_contour_measure(pred_polygon, gt_contours, sampling_spacing, max_stretch, metric_name="cosine"):
    pred_contours = shapely.geometry.GeometryCollection([pred_polygon.exterior, *pred_polygon.interiors])
    sampled_pred_contours = sample_geometry(pred_contours, sampling_spacing)
    # Project sampled contour points to ground truth contours
    projected_pred_contours = project_onto_geometry(sampled_pred_contours, gt_contours)
    contour_measures = []
    for contour, proj_contour in zip(sampled_pred_contours, projected_pred_contours):
        coords = np.array(contour.coords[:])
        proj_coords = np.array(proj_contour.coords[:])
        edges = coords[1:] - coords[:-1]
        proj_edges = proj_coords[1:] - proj_coords[:-1]
        # Remove edges with a norm of zero
        edge_norms = np.linalg.norm(edges, axis=1)
        proj_edge_norms = np.linalg.norm(proj_edges, axis=1)
        norm_valid_mask = 0 < edge_norms * proj_edge_norms
        edges = edges[norm_valid_mask]
        proj_edges = proj_edges[norm_valid_mask]
        edge_norms = edge_norms[norm_valid_mask]
        proj_edge_norms = proj_edge_norms[norm_valid_mask]
        # Remove edge that have stretched more than max_stretch (invalid projection)
        stretch = edge_norms / proj_edge_norms
        stretch_valid_mask = np.logical_and(1 / max_stretch < stretch, stretch < max_stretch)
        edges = edges[stretch_valid_mask]
        if edges.shape[0] == 0:
            # Invalid projection for the whole contour, skip it
            continue
        proj_edges = proj_edges[stretch_valid_mask]
        edge_norms = edge_norms[stretch_valid_mask]
        proj_edge_norms = proj_edge_norms[stretch_valid_mask]
        scalar_products = np.abs(np.sum(np.multiply(edges, proj_edges), axis=1) / (edge_norms * proj_edge_norms))
        try:
            contour_measures.append(scalar_products.min())
        except ValueError:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharex=True, sharey=True)
            ax = axes.ravel()
            plot_geometries(ax[0], [contour])
            plot_geometries(ax[1], [proj_contour])
            plot_geometries(ax[2], gt_contours)
            fig.tight_layout()
            plt.show()
    if len(contour_measures):
        min_scalar_product = min(contour_measures)
        measure = np.arccos(min_scalar_product)
        return measure
    else:
        return None

def sample_geometry(geom, density):
    """
    Sample edges of geom with a homogeneous density.

    @param geom:
    @param density:
    @return:
    """
    sample_lambda = lambda x: _sample_linestring(x, density)
    if isinstance(geom, shapely.geometry.GeometryCollection):
        sampled_geom = shapely.geometry.GeometryCollection(
            [sample_geometry(g, density) for g in geom]
        )
    elif isinstance(geom, shapely.geometry.Polygon):
        sampled_exterior = _sample_linestring(geom.exterior, density)
        sampled_interiors = list(map(sample_lambda, geom.interiors))
        sampled_geom = shapely.geometry.Polygon(sampled_exterior, sampled_interiors)
    elif isinstance(geom, shapely.geometry.LineString):
        sampled_geom = _sample_linestring(geom, density)
    else:
        raise TypeError(f"geom of type {type(geom)} not supported!")
    return sampled_geom

def _sample_linestring(geom, density):
    sampled_x, sampled_y = [], []
    coords = np.array(geom.coords[:])
    lengths = np.linalg.norm(coords[:-1] - coords[1:], axis=1)
    for i in range(len(lengths)):
        start = geom.coords[i]
        end = geom.coords[i + 1]
        length = lengths[i]
        num = max(1, int(round(length / density))) + 1
        x_seq = np.linspace(start[0], end[0], num)
        y_seq = np.linspace(start[1], end[1], num)
        if i > 0:
            x_seq = x_seq[1:]
            y_seq = y_seq[1:]
        sampled_x.append(x_seq)
        sampled_y.append(y_seq)
    sampled_x = np.concatenate(sampled_x)
    sampled_y = np.concatenate(sampled_y)
    sampled_coords = zip(sampled_x, sampled_y)
    sampled_geom = shapely.geometry.LineString(sampled_coords)
    return sampled_geom


def plot_geometries(axis, geometries, linewidths=1, markersize=3):
    if len(geometries):
        patches = []
        for i, geometry in enumerate(geometries):
            if geometry.geom_type == "Polygon":
                polygon = shapely.geometry.Polygon(geometry)
                if not polygon.is_empty:
                    patch = PolygonPatch(polygon)
                    patches.append(patch)
                axis.plot(*polygon.exterior.xy, marker="o", markersize=markersize)
                for interior in polygon.interiors:
                    axis.plot(*interior.xy, marker="o", markersize=markersize)
            elif geometry.geom_type == "LineString" or geometry.geom_type == "LinearRing":
                axis.plot(*geometry.xy, marker="o", markersize=markersize)
            else:
                raise NotImplementedError(f"Geom type {geometry.geom_type} not recognized.")
        random.seed(1)
        colors = random.choices([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [0.5, 1, 0, 1],
            [1, 0.5, 0, 1],
            [0.5, 0, 1, 1],
            [1, 0, 0.5, 1],
            [0, 0.5, 1, 1],
            [0, 1, 0.5, 1],
        ], k=len(patches))
        edgecolors = np.array(colors)
        facecolors = edgecolors.copy()
        p = PatchCollection(patches, facecolors=facecolors, edgecolors=edgecolors, linewidths=linewidths)
        axis.add_collection(p)

def PolygonPath(polygon):
    """Constructs a compound matplotlib path from a Shapely or GeoJSON-like
    geometric object"""

    def coding(ob):
        # The codes will be all "LINETO" commands, except for "MOVETO"s at the
        # beginning of each subpath
        n = len(getattr(ob, 'coords', None) or ob)
        vals = np.ones(n, dtype=Path.code_type) * Path.LINETO
        vals[0] = Path.MOVETO
        return vals

    if hasattr(polygon, 'geom_type'):  # Shapely
        ptype = polygon.geom_type
        if ptype == 'Polygon':
            polygon = [Polygon(polygon)]
        elif ptype == 'MultiPolygon':
            polygon = [Polygon(p) for p in polygon]
        else:
            raise ValueError(
                "A polygon or multi-polygon representation is required")

    else:  # GeoJSON
        polygon = getattr(polygon, '__geo_interface__', polygon)
        ptype = polygon["type"]
        if ptype == 'Polygon':
            polygon = [Polygon(polygon)]
        elif ptype == 'MultiPolygon':
            polygon = [Polygon(p) for p in polygon['coordinates']]
        else:
            raise ValueError(
                "A polygon or multi-polygon representation is required")

    vertices = np.concatenate([
        np.concatenate([np.asarray(t.exterior)[:, :2]] +
                    [np.asarray(r)[:, :2] for r in t.interiors])
        for t in polygon])
    codes = np.concatenate([
        np.concatenate([coding(t.exterior)] +
                    [coding(r) for r in t.interiors]) for t in polygon])

    return Path(vertices, codes)


def PolygonPatch(polygon, **kwargs):
    """Constructs a matplotlib patch from a geometric object

    The `polygon` may be a Shapely or GeoJSON-like object with or without holes.
    The `kwargs` are those supported by the matplotlib.patches.Polygon class
    constructor. Returns an instance of matplotlib.patches.PathPatch.

    Example (using Shapely Point and a matplotlib axes):

      >>> b = Point(0, 0).buffer(1.0)
      >>> patch = PolygonPatch(b, fc='blue', ec='blue', alpha=0.5)
      >>> axis.add_patch(patch)

    """
    return PathPatch(PolygonPath(polygon), **kwargs)

def point_project_onto_geometry(coord, target):
    point = shapely.geometry.Point(coord)
    _, projected_point = shapely.ops.nearest_points(point, target)
    # dist = point.distance(projected_point)
    return projected_point.coords[0]


def project_onto_geometry(geom, target, pool=None):
    """
    Projects all points from line_string onto target.
    @param geom:
    @param target:
    @param pool:
    @return:
    """
    if isinstance(geom, shapely.geometry.GeometryCollection):
        # tic = time.time()

        if pool is None:
            projected_geom = [project_onto_geometry(g, target, pool=pool) for g in geom]
        else:
            partial_project_onto_geometry = partial(project_onto_geometry, target=target)
            projected_geom = pool.map(partial_project_onto_geometry, geom)
        projected_geom = shapely.geometry.GeometryCollection(projected_geom)

        # toc = time.time()
        # print(f"project_onto_geometry: {toc - tic}s")
    elif isinstance(geom, shapely.geometry.Polygon):
        projected_exterior = project_onto_geometry(geom.exterior, target)
        projected_interiors = [project_onto_geometry(interior, target) for interior in geom.interiors]
        try:
            projected_geom = shapely.geometry.Polygon(projected_exterior, projected_interiors)
        except shapely.errors.TopologicalError as e:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharex=True, sharey=True)
            ax = axes.ravel()
            plot_geometries(ax[0], [geom])
            plot_geometries(ax[1], target)
            plot_geometries(ax[2], [projected_exterior, *projected_interiors])
            fig.tight_layout()
            plt.show()
            raise e
    elif isinstance(geom, shapely.geometry.LineString):
        projected_coords = [point_project_onto_geometry(coord, target) for coord in geom.coords]
        projected_geom = shapely.geometry.LineString(projected_coords)
    else:
        raise TypeError(f"geom of type {type(geom)} not supported!")
    return projected_geom
