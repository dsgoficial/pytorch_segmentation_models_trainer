# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-03-25
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba - 
                                    Cartographic Engineer @ Brazilian Army
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

import os
from pytorch_segmentation_models_trainer.tools.data_readers.vector_reader import save_to_file
import unittest

import hydra
import numpy as np
import segmentation_models_pytorch as smp
import torch
from hydra.experimental import compose, initialize
from matplotlib.testing.compare import compare_images
from omegaconf import OmegaConf
from pytorch_segmentation_models_trainer.tools.polygonization.methods import active_contours, active_skeletons
from pytorch_segmentation_models_trainer.tools.visualization import \
    crossfield_plot
from pytorch_segmentation_models_trainer.utils import (frame_field_utils,
                                                       math_utils)
from pytorch_segmentation_models_trainer.utils.os_utils import (create_folder,
                                                                hash_file,
                                                                remove_folder)

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, 'testing_data')

frame_field_root_dir = os.path.join(
    current_dir, 'testing_data', 'data', 'frame_field_data')

device = "cpu"

class Test_TestPolygonize(unittest.TestCase):

    def setUp(self):
        self.output_dir = create_folder(os.path.join(root_dir, 'test_output'))

    def tearDown(self):
        remove_folder(self.output_dir)
    
    def test_polygonize_active_contours_real_data(self) -> None:
        config_active_contours = {
            "steps": 500,
            "data_level": 0.5,
            "data_coef": 0.1,
            "length_coef": 0.4,
            "crossfield_coef": 0.5,
            "poly_lr": 0.01,
            "warmup_iters": 100,
            "warmup_factor": 0.1,
            "device": device,
            "tolerance": 0.125,
            "seg_threshold": 0.5,
            "min_area": 10
        }
        csv_path = os.path.join(frame_field_root_dir, 'dsg_dataset.csv')
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="frame_field_dataset.yaml",
                overrides=[
                    'input_csv_path='+csv_path,
                    'root_dir='+frame_field_root_dir
                ]
            )
            frame_field_ds = hydra.utils.instantiate(cfg)
        crossfield = frame_field_utils.compute_crossfield_to_plot(
            frame_field_ds[0]['gt_crossfield_angle']
        )

        # seg_batch = torch.tensor(np.transpose(seg[:, :, :2], (2, 0, 1)), dtype=torch.float)[None, ...]
        # crossfield_batch = torch.tensor(np.transpose(crossfield, (2, 0, 1)), dtype=torch.float)[None, ...]

        out_contours_batch, out_probs_batch = active_contours.polygonize(
            torch.movedim(frame_field_ds[0]['gt_polygons_image'], -1, 0).unsqueeze(0),
            crossfield, 
            config_active_contours
        )

        polygons = out_contours_batch[0]
        save_to_file(polygons, base_filepath=self.output_dir, name='vector', driver='GeoJSON', epsg=31982)

        filepath = os.path.join(self.output_dir, "output_poly_acm.png")
        crossfield_plot.save_poly_viz(frame_field_ds[0]['image'], polygons, filepath, linewidths=0.5, draw_vertices=True, crossfield=crossfield.squeeze(0))
        return True

    def test_polygonize_active_contours_fake_data(self) -> None:
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
            "device": device,
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
        seg = np.zeros((600, 800, 3))
        # Triangle:
        for i in range(400):
            for j in range(i+1):
                seg[i, j] = 1

        # L extension:
        seg[300:500, 500:700] = 1

        u = np.zeros((600, 800), dtype=np.complex)
        v = np.zeros((600, 800), dtype=np.complex)
        # Init with grid
        u.real = 1
        v.imag = 1
        # Add slope
        u[:400, :400] *= np.exp(1j * np.pi/4)
        v[:400, :400] *= np.exp(1j * np.pi/4)

        crossfield = math_utils.compute_crossfield_c0c2(u, v)

        seg_batch = torch.tensor(np.transpose(seg[:, :, :2], (2, 0, 1)), dtype=torch.float)[None, ...]
        crossfield_batch = torch.tensor(np.transpose(crossfield, (2, 0, 1)), dtype=torch.float)[None, ...]

        out_contours_batch, out_probs_batch = active_contours.polygonize(seg_batch, crossfield_batch, config)

        polygons = out_contours_batch[0]
        save_to_file(polygons, base_filepath=self.output_dir, name='vector', driver='GeoJSON')

        filepath = os.path.join(self.output_dir, "output_poly_acm.png")
        crossfield_plot.save_poly_viz(seg, polygons, filepath, linewidths=0.5, draw_vertices=True, crossfield=crossfield)
        return True

    def test_polygonize_active_skeletons_real_data(self) -> None:
        conf = {
            "init_method": "skeleton",  # Can be either skeleton or marching_squares
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
                "curvature_dissimilarity_threshold": 15,
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
            "lr": 0.001,
            "gamma": 0.0001,
            "device": device,
            "tolerance": 22,
            "seg_threshold": 0.5,
            "min_area": 12,
        }
        config_active_skeletons = OmegaConf.create(conf)
        csv_path = os.path.join(frame_field_root_dir, 'dsg_dataset.csv')
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="frame_field_dataset.yaml",
                overrides=[
                    'input_csv_path='+csv_path,
                    'root_dir='+frame_field_root_dir
                ]
            )
            frame_field_ds = hydra.utils.instantiate(cfg)
        crossfield = frame_field_utils.compute_crossfield_to_plot(
            frame_field_ds[0]['gt_crossfield_angle']
        )

        out_contours_batch, out_probs_batch = active_skeletons.polygonize(
            torch.movedim(frame_field_ds[0]['gt_polygons_image'], -1, 0).unsqueeze(0),
            crossfield, 
            config_active_skeletons
        )

        polygons = out_contours_batch[0]
        save_to_file(polygons, base_filepath=self.output_dir, name='output_poly_real_skeleton', driver='GeoJSON')

        filepath = os.path.join(self.output_dir, "output_poly_real_skeleton.png")
        crossfield_plot.save_poly_viz(frame_field_ds[0]['image'], polygons, filepath, linewidths=0.5, draw_vertices=True, crossfield=crossfield.squeeze(0))
        return True
    
    def test_polygonize_active_skeleton_fake_data(self) -> None:
        conf = {
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
            "device": device,
            "tolerance": 0.5,
            "seg_threshold": 0.5,
            "min_area": 10,
        }
        config_active_skeletons = OmegaConf.create(conf)

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

        out_contours_batch, out_probs_batch = active_skeletons.polygonize(seg_batch, crossfield_batch, config_active_skeletons)

        polygons = out_contours_batch[0]

        filepath = os.path.join(self.output_dir, "demo_poly_asm.png")
        crossfield_plot.save_poly_viz(seg[:, :, 0], polygons, filepath, linewidths=0.5, draw_vertices=True, crossfield=crossfield)
        return True
