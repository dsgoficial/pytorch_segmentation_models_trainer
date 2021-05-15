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


import numpy as np
import skimage
import skimage.measure
import skimage.io
import shapely.geometry
import shapely.ops
import shapely.prepared
import cv2

from functools import partial

import torch

from pytorch_segmentation_models_trainer.tools.polygonization import polygonize_utils

from pytorch_segmentation_models_trainer.optimizers.poly_optimizers import TensorPolyOptimizer
from pytorch_segmentation_models_trainer.tools.visualization import crossfield_plot
from pytorch_segmentation_models_trainer.utils import frame_field_utils, math_utils
from pytorch_segmentation_models_trainer.utils.tensor_utils import polygons_to_tensorpoly, tensorpoly_pad


def contours_batch_to_tensorpoly(contours_batch):
    # Convert a batch of contours to a TensorPoly representation with PyTorch tensors
    tensorpoly = polygons_to_tensorpoly(contours_batch)
    # Pad contours so that we can treat them as closed:
    tensorpoly = tensorpoly_pad(tensorpoly, padding=(0, 1))
    return tensorpoly


def tensorpoly_to_contours_batch(tensorpoly):
    # Convert back to contours
    contours_batch = [[] for _ in range(tensorpoly.batch_size)]
    for poly_i in range(tensorpoly.poly_slice.shape[0]):
        s = tensorpoly.poly_slice[poly_i, :]
        contour = np.array(tensorpoly.pos[s[0]:s[1], :].detach().cpu())
        is_open = tensorpoly.is_endpoint[s[0]]  # Is open = if first vertex is an endpoint
        if not is_open:
            # Close contour
            contour = np.concatenate([contour, contour[:1, :]], axis=0)
        batch_i = tensorpoly.batch[s[0]]  # Batch of polygon = batch of first vertex
        contours_batch[batch_i].append(contour)
    return contours_batch


def print_contours_stats(contours):
    min_length, max_length = contours[0].shape[0], contours[0].shape[0]
    nb_vertices = 0
    for contour in contours:
        nb_vertices += contour.shape[0]
        if contour.shape[0] < min_length:
            min_length = contour.shape[0]
        if max_length < contour.shape[0]:
            max_length = contour.shape[0]
    print("Nb polygon:", len(contours), "Nb vertices:", nb_vertices, "Min lengh:", min_length, "Max lengh:", max_length)


def shapely_postprocess(contours, u, v, np_indicator, tolerance, config):
    if type(tolerance) == list:
        # Use several tolerance values for simplification. return a dict with all results
        out_polygons_dict = {}
        out_probs_dict = {}
        for tol in tolerance:
            out_polygons, out_probs = shapely_postprocess(contours, u, v, np_indicator, tol, config)
            out_polygons_dict["tol_{}".format(tol)] = out_polygons
            out_probs_dict["tol_{}".format(tol)] = out_probs
        return out_polygons_dict, out_probs_dict
    else:
        height, width = np_indicator.shape[0], np_indicator.shape[1]
        contours = [skimage.measure.approximate_polygon(contour, tolerance=min(1, tolerance)) for contour in contours]
        corner_masks = frame_field_utils.detect_corners(contours, u, v)
        contours = polygonize_utils.split_polylines_corner(contours, corner_masks)

        # Convert to Shapely:
        line_string_list = [shapely.geometry.LineString(out_contour[:, ::-1]) for out_contour in contours]

        line_string_list = [line_string.simplify(tolerance, preserve_topology=True) for line_string in line_string_list]

        # Add image boundary line_strings for border polygons
        line_string_list.append(
            shapely.geometry.LinearRing([
                (0, 0),
                (0, height - 1),
                (width - 1, height - 1),
                (width - 1, 0),
            ]))
        multi_line_string = shapely.ops.unary_union(line_string_list)
        polygons, dangles, cuts, invalids = shapely.ops.polygonize_full(multi_line_string)
        polygons = [polygon for polygon in polygons if
                    config["min_area"] < polygon.area]
        filtered_polygons, filtered_polygon_probs = [], []
        for polygon in polygons:
            prob = polygonize_utils.compute_geom_prob(polygon, np_indicator)
            # print("acm:", np_indicator.min(), np_indicator.mean(), np_indicator.max(), prob)
            if config["seg_threshold"] < prob:
                filtered_polygons.append(polygon)
                filtered_polygon_probs.append(prob)

        return filtered_polygons, filtered_polygon_probs

def post_process(contours, np_seg, np_crossfield, config):
    u, v = math_utils.compute_crossfield_uv(np_crossfield)  # u, v are complex arrays
    np_indicator = np_seg[:, :, 0]
    polygons, probs = shapely_postprocess(contours, u, v, np_indicator, config["tolerance"], config)
    return polygons, probs

def polygonize(seg_batch, crossfield_batch, config, pool=None, pre_computed=None):
    assert len(seg_batch.shape) == 4 and seg_batch.shape[
        1] <= 3, "seg_batch should be (N, C, H, W) with C <= 3, not {}".format(seg_batch.shape)
    assert len(crossfield_batch.shape) == 4 and crossfield_batch.shape[
        1] == 4, "crossfield_batch should be (N, 4, H, W)"
    assert seg_batch.shape[0] == crossfield_batch.shape[0], "Batch size for seg and crossfield should match"

    indicator_batch = seg_batch[:, 0, :, :]
    np_indicator_batch = indicator_batch.cpu().numpy()
    indicator_batch = indicator_batch.to(config["device"])

    dist_batch = None
    if "dist_coef" in config:
        np_dist_batch = np.empty(np_indicator_batch.shape)
        for batch_i in range(np_indicator_batch.shape[0]):
            dist_1 = cv2.distanceTransform(
                np_indicator_batch[batch_i].astype(np.uint8),
                distanceType=cv2.DIST_L2,
                maskSize=cv2.DIST_MASK_5,
                dstType=cv2.CV_64F
            )
            dist_2 = cv2.distanceTransform(
                1 - np_indicator_batch[batch_i].astype(np.uint8),
                distanceType=cv2.DIST_L2,
                maskSize=cv2.DIST_MASK_5,
                dstType=cv2.CV_64F
            )
            np_dist_batch[0] = dist_1 + dist_2 - 1
        dist_batch = torch.from_numpy(np_dist_batch)
        dist_batch = dist_batch.to(config["device"])

    init_contours_batch = polygonize_utils.compute_init_contours_batch(
        np_indicator_batch,
        config["data_level"],
        pool=pool
    ) if (pre_computed is None or "init_contours_batch" not in pre_computed) \
    else pre_computed["init_contours_batch"]

    tensorpoly = contours_batch_to_tensorpoly(init_contours_batch)
    tensorpoly.to(config["device"])
    crossfield_batch = crossfield_batch.to(config["device"])
    dist_coef = config["dist_coef"] if "dist_coef" in config else None
    tensorpoly_optimizer = TensorPolyOptimizer(
        config, tensorpoly, indicator_batch, crossfield_batch,
        config["data_coef"], config["length_coef"], config["crossfield_coef"],
        dist=dist_batch, dist_coef=dist_coef
    )
    tensorpoly = tensorpoly_optimizer.optimize()

    out_contours_batch = tensorpoly_to_contours_batch(tensorpoly)

    np_seg_batch = np.transpose(seg_batch.cpu().numpy(), (0, 2, 3, 1))
    np_crossfield_batch = np.transpose(crossfield_batch.cpu().numpy(), (0, 2, 3, 1))
    if pool is not None:
        post_process_partial = partial(post_process, config=config)
        polygons_probs_batch = pool.starmap(
            post_process_partial,
            zip(
                out_contours_batch,
                np_seg_batch,
                np_crossfield_batch
            )
        )
        polygons_batch, probs_batch = zip(*polygons_probs_batch)
    else:
        polygons_batch, probs_batch = [], []
        for i, out_contours in enumerate(out_contours_batch):
            polygons, probs = post_process(
                out_contours,
                np_seg_batch[i],
                np_crossfield_batch[i],
                config
            )
            polygons_batch.append(polygons)
            probs_batch.append(probs)

    return polygons_batch, probs_batch


def main():
    config = {
        "indicator_add_edge": False,
        "steps": 500,
        "data_level": 0.5,
        "data_coef": 0.1,
        "length_coef": 0.4,
        "crossfield_coef": 0.5,
        "poly_lr": 0.01,
        "warmup_iters": 100,
        "warmup_factor": 0.1,
        "device": "cpu",
        "tolerance": 0.5,
        "seg_threshold": 0.5,
        "min_area": 1,

        "inner_polylines_params": {
            "enable": False,
            "max_traces": 1000,
            "seed_threshold": 0.5,
            "low_threshold": 0.1,
            "min_width": 2,  # Minimum width of trace to take into account
            "max_width": 8,
            "step_size": 1,
        }
    }
    seg = np.zeros((6, 8, 3))
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
    u[:4, :4] *= np.exp(1j * np.pi/4)
    v[:4, :4] *= np.exp(1j * np.pi/4)

    crossfield = math_utils.compute_crossfield_c0c2(u, v)

    seg_batch = torch.tensor(np.transpose(seg[:, :, :2], (2, 0, 1)), dtype=torch.float)[None, ...]
    crossfield_batch = torch.tensor(np.transpose(crossfield, (2, 0, 1)), dtype=torch.float)[None, ...]

    out_contours_batch, out_probs_batch = polygonize(seg_batch, crossfield_batch, config)

    polygons = out_contours_batch[0]

    filepath = "demo_poly_acm.pdf"
    crossfield_plot.save_poly_viz(seg, polygons, filepath, linewidths=0.5, draw_vertices=True, crossfield=crossfield)


if __name__ == '__main__':
    main()
