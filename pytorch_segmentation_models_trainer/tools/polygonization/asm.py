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
import numpy as np
import torch
import kornia
import shapely
import skimage
from pytorch_segmentation_models_trainer.optimizers.poly_optimizers import \
    PolygonAlignLoss, TensorPolyOptimizer
from pytorch_segmentation_models_trainer.utils.math_utils import compute_crossfield_uv
from pytorch_segmentation_models_trainer.utils import frame_field_utils
from pytorch_segmentation_models_trainer.tools.polygonization.skeletonize_tensor_tools import \
    Paths, Skeleton, TensorSkeleton, skeletons_to_tensorskeleton, tensorskeleton_to_skeletons

def get_junction_corner_index(tensorskeleton: TensorSkeleton):
    """
    Returns as a tensor the list of 3-tuples each representing a corner of a junction.
    The 3-tuple contains the indices of the 3 vertices making up the corner.

    In the text below, we use the following notation:
        - J: the number of junction nodes
        - Sd: the sum of the degrees of all the junction nodes
        - T: number of tip nodes
    @return: junction_corner_index of shape (Sd*J - T, 3) which is a list of 3-tuples (for each junction corner)
    """
    junction_edge_index = build_junction_index(tensorskeleton)
    junction_edge_index = remove_tip_junctions(tensorskeleton, junction_edge_index)
    grouped_junction_edge_index = group_junction_by_sorting(junction_edge_index)
    junction_angle_to_axis = compute_angle_to_vertical_axis_at_junction(
        tensorskeleton,
        grouped_junction_edge_index
    )
    junction_corner_index, junction_end_index = build_junction_corner_index()
    return slice_over_junction_index(
        junction_corner_index,
        junction_end_index,
        junction_angle_to_axis,
        grouped_junction_edge_index
    )

def build_junction_index(ts: TensorSkeleton) -> torch.tensor:
    junction_edge_index = torch.empty(
        (2 * ts.num_paths, 2),
        dtype=torch.long,
        device=ts.path_index.device
    )
    junction_edge_index[:ts.num_paths, 0] = ts.path_index[ts.path_delim[:-1]]
    junction_edge_index[:ts.num_paths, 1] = ts.path_index[ts.path_delim[:-1] + 1]
    junction_edge_index[ts.num_paths:, 0] = ts.path_index[ts.path_delim[1:] - 1]
    junction_edge_index[ts.num_paths:, 1] = ts.path_index[ts.path_delim[1:] - 2]
    return junction_edge_index

def remove_tip_junctions(ts: TensorSkeleton, junction_edge_index: torch.Tensor):
    degrees = ts.degrees[junction_edge_index[:, 0]]
    return junction_edge_index[1 < degrees, :]

def group_junction_by_sorting(junction_edge_index: torch.Tensor):
    group_indices = torch.argsort(junction_edge_index[:, 0], dim=0)
    return junction_edge_index[group_indices, :]

def slice_over_junction_index(junction_corner_index, junction_end_index, junction_angle_to_axis, grouped_junction_edge_index, slice_start=0):
    for slice_end in junction_end_index:
        slice_angle_to_axis = junction_angle_to_axis[slice_start:slice_end]
        slice_junction_edge_index = grouped_junction_edge_index[slice_start:slice_end]
        sort_indices = torch.argsort(slice_angle_to_axis, dim=0)
        slice_junction_edge_index = slice_junction_edge_index[sort_indices]
        junction_corner_index[slice_start:slice_end, 0] = slice_junction_edge_index[:, 1]
        junction_corner_index[slice_start:slice_end, 1] = slice_junction_edge_index[:, 0]
        junction_corner_index[slice_start:slice_end, 2] = slice_junction_edge_index[:, 1].roll(-1, dims=0)
        slice_start = slice_end
    return junction_corner_index

def compute_angle_to_vertical_axis_at_junction(ts: TensorSkeleton, grouped_junction_edge_index):
    junction_edge = ts.pos.detach()[grouped_junction_edge_index, :]
    junction_tangent = junction_edge[:, 1, :] - junction_edge[:, 0, :]
    return torch.atan2(
        junction_tangent[:, 1],
        junction_tangent[:, 0]
    )

def build_junction_corner_index(ts: TensorSkeleton, grouped_junction_edge_index):
    unique = torch.unique_consecutive(
        grouped_junction_edge_index[:, 0]
    )
    count = ts.degrees[unique]
    junction_end_index = torch.cumsum(count, dim=0)
    return (
        torch.empty(
            (grouped_junction_edge_index.shape[0], 3),
            dtype=torch.long,
            device=ts.path_index.device
        ),
        junction_end_index
    )

def shapely_postprocess(polylines, np_indicator, tolerance, config):
    if type(tolerance) == list:
        # Use several tolerance values for simplification. return a dict with all results
        out_polygons_dict = {}
        out_probs_dict = {}
        for tol in tolerance:
            out_polygons, out_probs = shapely_postprocess(polylines, np_indicator, tol, config)
            out_polygons_dict["tol_{}".format(tol)] = out_polygons
            out_probs_dict["tol_{}".format(tol)] = out_probs
        return out_polygons_dict, out_probs_dict
    else:
        height = np_indicator.shape[0]
        width = np_indicator.shape[1]

        # Convert to Shapely:
        # tic = time.time()
        line_string_list = [shapely.geometry.LineString(polyline[:, ::-1]) for polyline in polylines]
        line_string_list = [line_string.simplify(tolerance, preserve_topology=True) for line_string in line_string_list]
        # toc = time.time()
        # print(f"simplify: {toc - tic}s")

        # Add image boundary line_strings for border polygons
        line_string_list.append(
            shapely.geometry.LinearRing([
                (0, 0),
                (0, height - 1),
                (width - 1, height - 1),
                (width - 1, 0),
            ]))

        # debug_print("Merge polylines")

        # Merge polylines (for border polygons):

        # tic = time.time()
        multi_line_string = shapely.ops.unary_union(line_string_list)
        # toc = time.time()
        # print(f"shapely.ops.unary_union: {toc - tic}s")

        # debug_print("polygonize_full")

        # Find polygons:
        polygons = shapely.ops.polygonize(multi_line_string)
        polygons = list(polygons)

        # debug_print("Remove small polygons")

        # Remove small polygons
        # tic = time.time()
        polygons = [polygon for polygon in polygons if
                    config["min_area"] < polygon.area]
        # toc = time.time()
        # print(f"Remove small polygons: {toc - tic}s")

        # debug_print("Remove low prob polygons")

        # Remove low prob polygons
        # tic = time.time()

        filtered_polygons = []
        filtered_polygon_probs = []
        for polygon in polygons:
            prob = polygonize_utils.compute_geom_prob(polygon, np_indicator)
            # print("acm:", np_indicator.min(), np_indicator.mean(), np_indicator.max(), prob)
            if config["seg_threshold"] < prob:
                filtered_polygons.append(polygon)
                filtered_polygon_probs.append(prob)

        # toc = time.time()
        # print(f"Remove low prob polygons: {toc - tic}s")

        return filtered_polygons, filtered_polygon_probs


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
    # --- Skeletonize
    # tic = time.time()
    # Pad np_edge_mask first otherwise pixels on the bottom and right are lost after skeletonize:
    pad_width = 2
    np_edge_mask_padded = np.pad(np_edge_mask, pad_width=pad_width, mode="edge")
    skeleton_image = skimage.morphology.skeletonize(np_edge_mask_padded)
    skeleton_image = skeleton_image[pad_width:-pad_width, pad_width:-pad_width]

    # toc = time.time()
    # debug_print(f"skimage.morphology.skeletonize: {toc - tic}s")

    # tic = time.time()

    # if skeleton_image.max() == False:
    #     # There is no polylines to be detected
    #     return [], np.empty((0, 2), dtype=np.bool)

    skeleton = Skeleton()
    if 0 < skeleton_image.sum():
        # skan does not work in some cases (paths of 2 pixels or less, etc) which raises a ValueError, in witch case we continue with an empty skeleton.
        try:
            skeleton = skan.Skeleton(skeleton_image, keep_images=False)
            # skan.skeleton sometimes returns skeleton.coordinates.shape[0] != skeleton.degrees.shape[0] or
            # skeleton.coordinates.shape[0] != skeleton.paths.indices.max() + 1
            # Slice coordinates accordingly
            skeleton.coordinates = skeleton.coordinates[:skeleton.paths.indices.max() + 1]
            if skeleton.coordinates.shape[0] != skeleton.degrees.shape[0]:
                raise ValueError(f"skeleton.coordinates.shape[0] = {skeleton.coordinates.shape[0]} while skeleton.degrees.shape[0] = {skeleton.degrees.shape[0]}. They should be of same size.")
        except ValueError as e:
            if DEBUG:
                print_utils.print_warning(
                    f"WARNING: skan.Skeleton raised a ValueError({e}). skeleton_image has {skeleton_image.sum()} true values. Continuing without detecting skeleton in this image...")
                skimage.io.imsave("np_edge_mask.png", np_edge_mask.astype(np.uint8) * 255)
                skimage.io.imsave("skeleton_image.png", skeleton_image.astype(np.uint8) * 255)

    return skeleton


def get_marching_squares_skeleton(np_int_prob, config):
    """

    @param np_int_prob:
    @param config:
    @return:
    """
    # tic = time.time()
    contours = skimage.measure.find_contours(
        np_int_prob, config["data_level"],
        fully_connected='low',
        positive_orientation='high'
    )
    # Keep contours with more than 3 vertices and large enough area
    contours = [
        contour for contour in contours if 3 <= contour.shape[0] and \
            config["min_area"] < shapely.geometry.Polygon(contour).area
    ]

    # If there are no contours, return empty skeleton
    if len(contours) == 0:
        return Skeleton()

    coordinates = []
    indices_offset = 0
    indices = []
    indptr = [0]
    degrees = []

    for i, contour in enumerate(contours):
        # Check if it is a closed contour
        is_closed = np.max(np.abs(contour[0] - contour[-1])) < 1e-6
        if is_closed:
            _coordinates = contour[:-1, :]  # Don't include redundant vertex in coordinates
        else:
            _coordinates = contour
        _degrees = 2 * np.ones(_coordinates.shape[0], dtype=np.long)
        if not is_closed:
            _degrees[0] = 1
            _degrees[-1] = 1
        _indices = list(range(indices_offset, indices_offset + _coordinates.shape[0]))
        if is_closed:
            _indices.append(_indices[0])  # Close contour with indices
        coordinates.append(_coordinates)
        degrees.append(_degrees)
        indices.extend(_indices)
        indptr.append(indptr[-1] + len(_indices))
        indices_offset += _coordinates.shape[0]

    coordinates = np.concatenate(coordinates, axis=0)
    degrees = np.concatenate(degrees, axis=0)
    indices = np.array(indices)
    indptr = np.array(indptr)

    paths = Paths(indices, indptr)
    skeleton = Skeleton(coordinates, paths, degrees)

    return skeleton


# @profile
def compute_skeletons(seg_batch, config, spatial_gradient, pool=None) -> List[Skeleton]:
    assert len(seg_batch.shape) == 4 and seg_batch.shape[
        1] <= 3, "seg_batch should be (N, C, H, W) with C <= 3, not {}".format(seg_batch.shape)

    int_prob_batch = seg_batch[:, 0, :, :]
    if config["init_method"] == "marching_squares":
        # Only interior segmentation is available, initialize with marching squares
        np_int_prob_batch = int_prob_batch.cpu().numpy()
        get_marching_squares_skeleton_partial = functools.partial(get_marching_squares_skeleton, config=config)
        if pool is not None:
            skeletons_batch = pool.map(get_marching_squares_skeleton_partial, np_int_prob_batch)
        else:
            skeletons_batch = list(map(get_marching_squares_skeleton_partial, np_int_prob_batch))
    elif config["init_method"] == "skeleton":
        corrected_edge_prob_batch = config["data_level"] < int_prob_batch  # Convet to mask
        corrected_edge_prob_batch = corrected_edge_prob_batch[:, None, :, :].float() # Convet to float for spatial grads
        corrected_edge_prob_batch = 2 * spatial_gradient(corrected_edge_prob_batch)[:, 0, :, :]  # (b, 2, h, w), Normalize (kornia normalizes to -0.5, 0.5 for input in [0, 1])
        corrected_edge_prob_batch = corrected_edge_prob_batch.norm(dim=1)  # (b, h, w), take the gradient norm
        if 2 <= seg_batch.shape[1]:
            corrected_edge_prob_batch = torch.clamp(seg_batch[:, 1, :, :] + corrected_edge_prob_batch, 0, 1)

        # --- Init skeleton
        corrected_edge_mask_batch = config["data_level"] < corrected_edge_prob_batch
        np_corrected_edge_mask_batch = corrected_edge_mask_batch.cpu().numpy()

        get_skeleton_partial = functools.partial(get_skeleton, config=config)
        if pool is not None:
            skeletons_batch = pool.map(get_skeleton_partial, np_corrected_edge_mask_batch)
        else:
            skeletons_batch = list(map(get_skeleton_partial, np_corrected_edge_mask_batch))
    else:
        raise NotImplementedError(f"init_method '{config['init_method']}' not recognized. Valid init methods are 'skeleton' and 'marching_squares'")

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
        self.spatial_gradient = kornia.filters.SpatialGradient(
            mode="scharr",
            coord="ij",
            normalized=True,
            device=self.config["device"],
            dtype=torch.float
        )

    # @profile
    def __call__(self, seg_batch, crossfield_batch, pre_computed=None):
        assert len(seg_batch.shape) == 4 and seg_batch.shape[
            1] <= 3, "seg_batch should be (N, C, H, W) with C <= 3, not {}".format(seg_batch.shape)
        assert len(crossfield_batch.shape) == 4 and crossfield_batch.shape[
            1] == 4, "crossfield_batch should be (N, 4, H, W)"
        assert seg_batch.shape[0] == crossfield_batch.shape[0], "Batch size for seg and crossfield should match"

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
            artists = plot_utils.plot_geometries(ax, out_polylines, draw_vertices=True, linewidths=1)

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
        if self.pool is not None:
            polygons_probs_batch = self.pool.starmap(post_process_partial,
                                                zip(polylines_batch, np_int_prob_batch, np_crossfield_batch))
        else:
            polygons_probs_batch = map(post_process_partial, polylines_batch, np_int_prob_batch,
                                       np_crossfield_batch)
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
            plot_utils.plot_geometries(ax[0], out_polylines, draw_vertices=True, linewidths=1)
            ax[0].axis('off')
            ax[0].set_title('original', fontsize=20)
            fig.tight_layout()
            plt.show()

        return polygons_batch, probs_batch


def polygonize(seg_batch, crossfield_batch, config, pool=None, pre_computed=None):
    polygonizer_asm = PolygonizerASM(config, pool=pool)
    return polygonizer_asm(seg_batch, crossfield_batch, pre_computed=pre_computed)

