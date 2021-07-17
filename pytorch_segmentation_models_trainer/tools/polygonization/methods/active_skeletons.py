# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-04-07
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

import functools
from typing import List
import numpy as np
import torch
import kornia
import shapely
import skimage
import skimage.measure
from functools import partial
import logging
import skan
import tqdm
from pytorch_segmentation_models_trainer.optimizers.poly_optimizers import \
    PolygonAlignLoss, TensorPolyOptimizer, TensorSkeletonOptimizer
from pytorch_segmentation_models_trainer.tools.visualization import crossfield_plot
from pytorch_segmentation_models_trainer.utils.math_utils import compute_crossfield_uv
from pytorch_segmentation_models_trainer.utils import frame_field_utils, math_utils, tensor_utils
from pytorch_segmentation_models_trainer.tools.polygonization.skeletonize_tensor_tools import \
    Paths, Skeleton, TensorSkeleton, skeletons_to_tensorskeleton, tensorskeleton_to_skeletons
from pytorch_segmentation_models_trainer.tools.polygonization import polygonize_utils

logger = logging.getLogger(__name__)

def shapely_postprocess(polylines, np_indicator, tolerance, config):
    if isinstance(tolerance, list):
        # Use several tolerance values for simplification. return a dict with all results
        out_polygons_dict, out_probs_dict = {}, {}
        for tol in tolerance:
            (
                out_polygons_dict[f"tol_{tol}"],
                out_probs_dict[f"tol_{tol}"]
            ) = shapely_postprocess(polylines, np_indicator, tol, config)
        return out_polygons_dict, out_probs_dict
    height, width = np_indicator.shape[0:2]
    linestring_list = _get_linestring_list(polylines, width, height, tolerance)
    filtered_polygons, filtered_polygon_probs = [], []
    polygonize_lambda_func = lambda x: _polygonize_in_threshold(
        x, config.min_area, config.seg_threshold,\
        filtered_polygons, filtered_polygon_probs, np_indicator
    )
    list(
        map(
            polygonize_lambda_func,
            shapely.ops.polygonize(
                shapely.ops.unary_union(linestring_list)
            )
        )
    )
    return filtered_polygons, filtered_polygon_probs

def _get_linestring_list(polylines, width, height, tol):
    line_string_iter = (
        shapely.geometry.LineString(polyline[:, ::-1]) \
            for polyline in polylines
    )
    line_string_list = [
        line_string.simplify(tol, preserve_topology=True) \
            for line_string in line_string_iter
    ]
    line_string_list.append(
        shapely.geometry.LinearRing([
            (0, 0),
            (0, height - 1),
            (width - 1, height - 1),
            (width - 1, 0),
        ]))
    return line_string_list

def _polygonize_in_threshold(polygon, min_area_threshold, seg_threshold, filtered_polygons, \
    filtered_polygon_probs, np_indicator):
    if polygon.area <= min_area_threshold:
        return
    prob = polygonize_utils.compute_geom_prob(polygon, np_indicator)
    if prob > seg_threshold:
        filtered_polygons.append(polygon)
        filtered_polygon_probs.append(prob)


def post_process(polylines, np_indicator, np_crossfield, config):
    u, v = compute_crossfield_uv(np_crossfield)  # u, v are complex arrays
    corner_masks = frame_field_utils.detect_corners(polylines, u, v)
    polylines = polygonize_utils.split_polylines_corner(polylines, corner_masks)
    polygons, probs = shapely_postprocess(polylines, np_indicator, config.tolerance, config)
    return polygons, probs


def get_skeleton(np_edge_mask, config):
    """

    @param np_edge_mask:
    @param config:
    @return:
    """
    # Pad np_edge_mask first otherwise pixels on the bottom 
    # and right are lost after skeletonize:
    pad_width = 2
    np_edge_mask_padded = np.pad(np_edge_mask, pad_width=pad_width, mode="edge")
    skeleton_image = skimage.morphology.skeletonize(np_edge_mask_padded)
    skeleton_image = skeleton_image[pad_width:-pad_width, pad_width:-pad_width]
    skeleton = Skeleton()
    if skeleton_image.sum() <= 0:
        return skeleton
    # skan does not work in some cases (paths of 2 pixels or less, etc) 
    # which raises a ValueError, in witch case we continue with an empty skeleton.
    try:
        skeleton = skan.Skeleton(skeleton_image, keep_images=False)
        # Slice coordinates accordingly
        skeleton.coordinates = skeleton.coordinates[:skeleton.paths.indices.max() + 1]
        if skeleton.coordinates.shape[0] != skeleton.degrees.shape[0]:
            raise ValueError(
                f"skeleton.coordinates.shape[0] = {skeleton.coordinates.shape[0]} "
                "while skeleton.degrees.shape[0] = {skeleton.degrees.shape[0]}. "
                "They should be of same size."
            )
    except ValueError as e:
        logger.warning(
            f"WARNING: skan.Skeleton raised a ValueError({e}). "
            "skeleton_image has {skeleton_image.sum()} true values. "
            "Continuing without detecting skeleton in this image..."
        )
        skimage.io.imsave("np_edge_mask.png", np_edge_mask.astype(np.uint8) * 255)
        skimage.io.imsave("skeleton_image.png", skeleton_image.astype(np.uint8) * 255)
    return skeleton


def get_marching_squares_skeleton(np_int_prob, config):
    """

    @param np_int_prob:
    @param config:
    @return:
    """
    contours = skimage.measure.find_contours(
        np_int_prob, config.data_level,
        fully_connected='low',
        positive_orientation='high'
    )
    # Keep contours with more than 3 vertices and large enough area
    contours = [
        contour for contour in contours if contour.shape[0] >= 3 and \
            shapely.geometry.Polygon(contour).area > config.min_area
    ]

    # If there are no contours, return empty skeleton
    if len(contours) == 0:
        return Skeleton()

    coordinates, indices, degrees, indptr = [], [], [], [0]
    indices_offset = 0
    def _populate_aux_structures(contour, indices_offset):
        # Check if it is a closed contour
        is_closed = np.max(np.abs(contour[0] - contour[-1])) < 1e-6
        _coordinates = contour[:-1, :] if is_closed else contour
        _degrees = 2 * np.ones(_coordinates.shape[0], dtype=np.long)
        if not is_closed:
            _degrees[0], _degrees[-1] = 1, 1
        _indices = list(
            range(indices_offset, indices_offset + _coordinates.shape[0])
        )
        if is_closed:
            _indices.append(_indices[0])  # Close contour with indices
        coordinates.append(_coordinates)
        degrees.append(_degrees)
        indices.extend(_indices)
        indptr.append(indptr[-1] + len(_indices))
        indices_offset += _coordinates.shape[0]
    lambda_func = lambda x: _populate_aux_structures(x, indices_offset)

    list(map(lambda_func, contours))

    return Skeleton(
        coordinates=np.concatenate(coordinates, axis=0),
        paths=Paths(
            indices=np.array(indices),
            indptr=np.array(indptr)
        ),
        degrees=np.concatenate(degrees, axis=0)
    )

def compute_skeletons(seg_batch, config, spatial_gradient, pool=None) -> List[Skeleton]:
    assert len(seg_batch.shape) == 4 and seg_batch.shape[
        1] <= 3, "seg_batch should be (N, C, H, W) with C <= 3, not {}".format(seg_batch.shape)
    if config.init_method not in ["marching_squares", "skeleton"]:
        raise NotImplementedError(
            f"init_method '{config['init_method']}' not recognized."
            " Valid init methods are 'skeleton' and 'marching_squares'"
        )
    return _compute_skeletons_with_marching_squares(
        seg_batch, config, pool) if config.init_method == "marching_squares" \
            else _compute_seletons_with_skeletonize(seg_batch, config, spatial_gradient, pool)

def _compute_seletons_with_skeletonize(seg_batch, config, spatial_gradient, pool):
    int_prob_batch = seg_batch[:, 0, :, :]
    corrected_edge_prob_batch = int_prob_batch > config.data_level
    corrected_edge_prob_batch = corrected_edge_prob_batch[:, None, :, :].float()
    corrected_edge_prob_batch = 2 * spatial_gradient(corrected_edge_prob_batch)[:, 0, :, :]
    corrected_edge_prob_batch = corrected_edge_prob_batch.norm(dim=1)
    if seg_batch.shape[1] >= 2:
        corrected_edge_prob_batch = torch.clamp(
                seg_batch[:, 1, :, :] + corrected_edge_prob_batch, 0, 1
            )

        # --- Init skeleton
    corrected_edge_mask_batch = corrected_edge_prob_batch > config.data_level
    np_corrected_edge_mask_batch = corrected_edge_mask_batch.cpu().numpy()

    get_skeleton_partial = functools.partial(get_skeleton, config=config)
    skeletons_batch = pool.map(
            get_skeleton_partial, np_corrected_edge_mask_batch
        ) if pool is not None \
            else list(map(get_skeleton_partial, np_corrected_edge_mask_batch))
        
    return skeletons_batch

def _compute_skeletons_with_marching_squares(seg_batch, config, pool):
    int_prob_batch = seg_batch[:, 0, :, :]
    np_int_prob_batch = int_prob_batch.cpu().numpy()
    get_marching_squares_skeleton_partial = functools.partial(get_marching_squares_skeleton, config=config)
    return pool.map(
            get_marching_squares_skeleton_partial, np_int_prob_batch
        ) if pool is not None \
            else list(map(get_marching_squares_skeleton_partial, np_int_prob_batch))


def skeleton_to_polylines(skeleton: Skeleton) -> List[np.ndarray]:
    polylines = []
    for path_i in range(skeleton.paths.indptr.shape[0] - 1):
        start, stop = skeleton.paths.indptr[path_i:path_i + 2]
        path_indices = skeleton.paths.indices[start:stop]
        path_coordinates = skeleton.coordinates[path_indices]
        polylines.append(path_coordinates)
    return polylines


class PolygonizerASM:
    def __init__(self, config, pool=None):
        self.config = config
        self.pool = pool
        self.spatial_gradient = tensor_utils.SpatialGradient(
            mode="scharr",
            coord="ij",
            normalized=True,
            device=self.config.device,
            dtype=torch.float
        )

    # @profile
    def __call__(self, seg_batch, crossfield_batch, pre_computed=None):
        assert len(seg_batch.shape) == 4 and seg_batch.shape[1] <= 3, \
            "seg_batch should be (N, C, H, W) with C <= 3, not {}".format(seg_batch.shape)
        assert len(crossfield_batch.shape) == 4 and crossfield_batch.shape[1] == 4,\
            "crossfield_batch should be (N, 4, H, W)"
        assert seg_batch.shape[0] == crossfield_batch.shape[0],\
            "Batch size for seg and crossfield should match"

        seg_batch = seg_batch.to(self.config.device)
        crossfield_batch = crossfield_batch.to(self.config.device)
        skeletons_batch = compute_skeletons(seg_batch, self.config, self.spatial_gradient, pool=self.pool)
        tensorskeleton = skeletons_to_tensorskeleton(skeletons_batch, device=self.config.device)

        # --- Check if tensorskeleton is empty
        if tensorskeleton.num_paths == 0:
            batch_size = seg_batch.shape[0]
            polygons_batch = [[]]*batch_size
            probs_batch = [[]]*batch_size
            return polygons_batch, probs_batch

        int_prob_batch = seg_batch[:, 0, :, :]
        # dist_batch = dist_batch.to(config.device)
        tensorskeleton_optimizer = TensorSkeletonOptimizer(
            self.config,
            tensorskeleton,
            int_prob_batch,
            crossfield_batch
        )
        tensorskeleton = tensorskeleton_optimizer.optimize()
        out_skeletons_batch = tensorskeleton_to_skeletons(tensorskeleton)
        polygons_batch, probs_batch = self._skeletons_to_polygons(
            crossfield_batch, int_prob_batch, out_skeletons_batch)
        return polygons_batch, probs_batch

    def _skeletons_to_polygons(self, crossfield_batch, int_prob_batch, out_skeletons_batch):
        polylines_batch = [
            skeleton_to_polylines(skeleton) for skeleton in out_skeletons_batch
        ]
        np_crossfield_batch = np.transpose(crossfield_batch.cpu().numpy(), (0, 2, 3, 1))
        np_int_prob_batch = int_prob_batch.cpu().numpy()
        post_process_partial = partial(post_process, config=self.config)
        polygons_probs_batch = self.pool.starmap(
            post_process_partial,
            zip(polylines_batch, np_int_prob_batch, np_crossfield_batch)
        ) if self.pool is not None else map(
            post_process_partial, polylines_batch, np_int_prob_batch, np_crossfield_batch
        )
        polygons_batch, probs_batch = zip(*polygons_probs_batch)
        return polygons_batch,probs_batch

def polygonize(seg_batch, crossfield_batch, config, pool=None, pre_computed=None):
    polygonizer_asm = PolygonizerASM(config, pool=pool)
    return polygonizer_asm(seg_batch, crossfield_batch, pre_computed=pre_computed)
