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
    DEBUG, PolygonAlignLoss, TensorPolyOptimizer, TensorSkeletonOptimizer
from pytorch_segmentation_models_trainer.tools.visualization import crossfield_plot
from pytorch_segmentation_models_trainer.utils.math_utils import compute_crossfield_uv
from pytorch_segmentation_models_trainer.utils import frame_field_utils, math_utils, tensor_utils
from pytorch_segmentation_models_trainer.tools.polygonization.skeletonize_tensor_tools import \
    Paths, Skeleton, TensorSkeleton, skeletons_to_tensorskeleton, tensorskeleton_to_skeletons
from pytorch_segmentation_models_trainer.tools.polygonization import polygonize_utils

DEBUG = False

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
        x, config['min_area'], config['seg_threshold'],\
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
    polygons, probs = shapely_postprocess(polylines, np_indicator, config["tolerance"], config)
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
        if DEBUG:
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
        np_int_prob, config["data_level"],
        fully_connected='low',
        positive_orientation='high'
    )
    # Keep contours with more than 3 vertices and large enough area
    contours = [
        contour for contour in contours if contour.shape[0] >= 3 and \
            shapely.geometry.Polygon(contour).area > config["min_area"]
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

# @profile
def compute_skeletons(seg_batch, config, spatial_gradient, pool=None) -> List[Skeleton]:
    assert len(seg_batch.shape) == 4 and seg_batch.shape[
        1] <= 3, "seg_batch should be (N, C, H, W) with C <= 3, not {}".format(seg_batch.shape)

    int_prob_batch = seg_batch[:, 0, :, :]
    if config["init_method"] == "marching_squares":
        # Only interior segmentation is available, initialize with marching squares
        np_int_prob_batch = int_prob_batch.cpu().numpy()
        get_marching_squares_skeleton_partial = functools.partial(get_marching_squares_skeleton, config=config)
        skeletons_batch = pool.map(
            get_marching_squares_skeleton_partial, np_int_prob_batch
        ) if pool is not None \
            else list(map(get_marching_squares_skeleton_partial, np_int_prob_batch))
    elif config["init_method"] == "skeleton":
        corrected_edge_prob_batch = int_prob_batch > config["data_level"]
        corrected_edge_prob_batch = corrected_edge_prob_batch[:, None, :, :].float()
        corrected_edge_prob_batch = 2 * spatial_gradient(corrected_edge_prob_batch)[:, 0, :, :]
        corrected_edge_prob_batch = corrected_edge_prob_batch.norm(dim=1)
        if seg_batch.shape[1] >= 2:
            corrected_edge_prob_batch = torch.clamp(
                seg_batch[:, 1, :, :] + corrected_edge_prob_batch, 0, 1
            )

        # --- Init skeleton
        corrected_edge_mask_batch = corrected_edge_prob_batch > config["data_level"]
        np_corrected_edge_mask_batch = corrected_edge_mask_batch.cpu().numpy()

        get_skeleton_partial = functools.partial(get_skeleton, config=config)
        skeletons_batch = pool.map(
            get_skeleton_partial, np_corrected_edge_mask_batch
        ) if pool is not None \
            else list(map(get_skeleton_partial, np_corrected_edge_mask_batch))
    else:
        raise NotImplementedError(
            f"init_method '{config['init_method']}' not recognized."
            " Valid init methods are 'skeleton' and 'marching_squares'"
        )

    return skeletons_batch


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
            device=self.config["device"],
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

        seg_batch = seg_batch.to(self.config["device"])
        crossfield_batch = crossfield_batch.to(self.config["device"])
        skeletons_batch = compute_skeletons(seg_batch, self.config, self.spatial_gradient, pool=self.pool)
        tensorskeleton = skeletons_to_tensorskeleton(skeletons_batch, device=self.config["device"])

        # --- Check if tensorskeleton is empty
        if tensorskeleton.num_paths == 0:
            batch_size = seg_batch.shape[0]
            polygons_batch = [[]]*batch_size
            probs_batch = [[]]*batch_size
            return polygons_batch, probs_batch

        int_prob_batch = seg_batch[:, 0, :, :]
        # dist_batch = dist_batch.to(config["device"])
        tensorskeleton_optimizer = TensorSkeletonOptimizer(
            self.config,
            tensorskeleton,
            int_prob_batch,
            crossfield_batch
        )

        if DEBUG:
            # Animation of optimization
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.autoscale(False)
            ax.axis('equal')
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Plot without margins

            image = int_prob_batch.cpu().numpy()[0]
            ax.imshow(image, cmap=plt.cm.gray)

            out_skeletons_batch = tensorskeleton_to_skeletons(tensorskeleton)
            polylines_batch = [skeleton_to_polylines(skeleton) for skeleton in out_skeletons_batch]
            out_polylines = [shapely.geometry.LineString(polyline[:, ::-1]) for polyline in polylines_batch[0]]
            artists = crossfield_plot.plot_geometries(ax, out_polylines, draw_vertices=True, linewidths=1)

            optim_pbar = tqdm(desc="Gradient descent", leave=True, total=self.config["loss_params"]["coefs"]["step_thresholds"][-1])

            def init():  # only required for blitting to give a clean slate.
                for artist, polyline in zip(artists, polylines_batch[0]):
                    artist.set_xdata([np.nan] * polyline.shape[0])
                    artist.set_ydata([np.nan] * polyline.shape[0])
                return artists

            def animate(i):
                loss, losses_dict = tensorskeleton_optimizer.step(i)
                optim_pbar.update(int(2 * i / self.config["loss_params"]["coefs"]["step_thresholds"][-1]))
                optim_pbar.set_postfix(loss=loss, **losses_dict)
                out_skeletons_batch = tensorskeleton_to_skeletons(tensorskeleton)
                polylines_batch = [skeleton_to_polylines(skeleton) for skeleton in out_skeletons_batch]
                for artist, polyline in zip(artists, polylines_batch[0]):
                    artist.set_xdata(polyline[:, 1])
                    artist.set_ydata(polyline[:, 0])
                return artists

            ani = animation.FuncAnimation(
                fig, animate, init_func=init, interval=0, blit=True, frames=self.config["loss_params"]["coefs"]["step_thresholds"][-1], repeat=False)
            plt.show()
        else:
            tensorskeleton = tensorskeleton_optimizer.optimize()

        out_skeletons_batch = tensorskeleton_to_skeletons(tensorskeleton)

        # --- Convert the skeleton representation into polylines
        polylines_batch = [skeleton_to_polylines(skeleton) for skeleton in out_skeletons_batch]

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

        if DEBUG:
            # --- display results
            import matplotlib.pyplot as plt
            image = np_int_prob_batch[0]
            polygons = polygons_batch[0]
            out_polylines = [shapely.geometry.LineString(polyline[:, ::-1]) for polyline in polylines_batch[0]]

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 16), sharex=True, sharey=True)
            ax = axes.ravel()

            ax[0].imshow(image, cmap=plt.cm.gray)
            crossfield_plot.plot_geometries(ax[0], out_polylines, draw_vertices=True, linewidths=1)
            ax[0].axis('off')
            ax[0].set_title('original', fontsize=20)
            fig.tight_layout()
            plt.show()

        return polygons_batch, probs_batch


def polygonize(seg_batch, crossfield_batch, config, pool=None, pre_computed=None):
    polygonizer_asm = PolygonizerASM(config, pool=pool)
    return polygonizer_asm(seg_batch, crossfield_batch, pre_computed=pre_computed)

def main():
    
    config = {
        "init_method": "marching_squares",  # Can be either skeleton or marching_squares
        "data_level": 0.5,
        "loss_params": {
            "coefs": {
                "step_thresholds": [0, 100, 200, 300],  # From 0 to 500: gradually go from coefs[0] to coefs[1]
                "data": [1.0, 0.1, 0.0, 0.0],
                "crossfield": [0.0, 0.05, 0.0, 0.0],
                "length": [0.1, 0.01, 0.0, 0.0],
                "curvature": [0.0, 0.0, 0.0, 0.0],
                "corner": [0.0, 0.0, 0.0, 0.0],
                "junction": [0.0, 0.0, 0.0, 0.0],
            },
            "curvature_dissimilarity_threshold": 2,
            # In pixels: for each sub-paths, if the dissimilarity (in the same sense as in the Ramer-Douglas-Peucker alg) is lower than straightness_threshold, then optimize the curve angles to be zero.
            "corner_angles": [45, 90, 135],  # In degrees: target angles for corners.
            "corner_angle_threshold": 22.5,
            # If a corner angle is less than this threshold away from any angle in corner_angles, optimize it.
            "junction_angles": [0, 45, 90, 135],  # In degrees: target angles for junction corners.
            "junction_angle_weights": [1, 0.01, 0.1, 0.01],
            # Order of decreassing importance: straight, right-angle, then 45° junction corners.
            "junction_angle_threshold": 22.5,
            # If a junction corner angle is less than this threshold away from any angle in junction_angles, optimize it.
        },
        "lr": 0.01,
        "gamma": 0.995,
        "device": "cpu",
        "tolerance": 0.5,
        "seg_threshold": 0.5,
        "min_area": 10,
    }

    seg = np.zeros((6, 8, 1))
    # Triangle:
    seg[1, 4] = 1
    seg[2, 3:5] = 1
    seg[3, 2:5] = 1
    seg[4, 1:5] = 1
    # L extension:
    seg[3:5, 5:7] = 1

    u = np.zeros((6, 8), dtype=np.complex)
    v = np.zeros((6, 8), dtype=np.complex)
    # Init with grid
    u.real = 1
    v.imag = 1
    # Add slope
    u[:4, :4] *= np.exp(1j * np.pi / 4)
    v[:4, :4] *= np.exp(1j * np.pi / 4)
    # Add slope corners
    # u[:2, 4:6] *= np.exp(1j * np.pi / 4)
    # v[4:, :2] *= np.exp(- 1j * np.pi / 4)

    crossfield = math_utils.compute_crossfield_c0c2(u, v)

    seg_batch = torch.tensor(np.transpose(seg[:, :, :2], (2, 0, 1)), dtype=torch.float)[None, ...]
    crossfield_batch = torch.tensor(np.transpose(crossfield, (2, 0, 1)), dtype=torch.float)[None, ...]

    # Add samples to batch to increase batch size
    batch_size = 16
    seg_batch = seg_batch.repeat((batch_size, 1, 1, 1))
    crossfield_batch = crossfield_batch.repeat((batch_size, 1, 1, 1))

    out_contours_batch, out_probs_batch = polygonize(seg_batch, crossfield_batch, config)

    polygons = out_contours_batch[0]

    filepath = "demo_poly_asm.pdf"
    crossfield_plot.save_poly_viz(seg[:, :, 0], polygons, filepath, linewidths=0.5, draw_vertices=True, crossfield=crossfield)


if __name__ == '__main__':
    main()
