# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-03-30
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

import hydra
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import shapely.geometry
import torch
from hydra import compose, initialize
from matplotlib.testing.compare import compare_images
from pytorch_segmentation_models_trainer.tools.visualization.crossfield_plot import (
    get_image_plot_crossfield,
    get_seg_display,
    get_tensorboard_image_seg_display,
    plot_geometries,
    plot_line_strings,
    plot_polygons,
    save_poly_viz,
)
from pytorch_segmentation_models_trainer.utils.frame_field_utils import (
    compute_crossfield_to_plot,
)
from tests.utils import BasicTestCase

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")

frame_field_root_dir = os.path.join(
    current_dir, "testing_data", "data", "frame_field_data"
)


class Test_Visualization(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.output_dir = self.make_temp_dir()

    def test_seg_display_real_example(self) -> None:
        csv_path = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        with initialize(config_path="./test_configs", version_base=None):
            cfg = compose(
                config_name="frame_field_dataset.yaml",
                overrides=[
                    "input_csv_path=" + csv_path,
                    "root_dir=" + frame_field_root_dir,
                ],
            )
            frame_field_ds = hydra.utils.instantiate(cfg, _recursive_=False)
        crossfield = compute_crossfield_to_plot(
            frame_field_ds[0]["gt_crossfield_angle"]
        )
        image_seg_display = get_tensorboard_image_seg_display(
            torch.movedim(frame_field_ds[0]["image"], -1, 0).unsqueeze(0),
            255
            * torch.movedim(frame_field_ds[0]["gt_polygons_image"], -1, 0).unsqueeze(0),
            crossfield=crossfield,
            crossfield_stride=8,
            width=1,
        )
        image_seg_display = image_seg_display.cpu().numpy().transpose(0, 2, 3, 1)
        skimage.io.imsave(
            os.path.join(self.output_dir, f"image_seg_display.png"),
            image_seg_display.squeeze(0).clip(0, 255).astype(np.uint8),
        )
        self.assertEqual(
            compare_images(
                os.path.join(
                    root_dir,
                    "expected_outputs",
                    "visualization",
                    "real_image_seg_display.png",
                ),
                os.path.join(self.output_dir, f"image_seg_display.png"),
                0.001,
            ),
            None,
        )

    def test_seg_display(self) -> None:
        image = torch.zeros((1, 3, 512, 512)) + 0.5
        seg = torch.zeros((1, 2, 512, 512))
        seg[:, 0, 100:200, 100:200] = 1
        crossfield = compute_crossfield_to_plot(0.25, crossfield_shape=(1, 4, 512, 512))

        image_seg_display = get_tensorboard_image_seg_display(
            255 * image, 255 * seg, crossfield=crossfield
        )
        image_seg_display = image_seg_display.cpu().numpy().transpose(0, 2, 3, 1)
        skimage.io.imsave(
            os.path.join(self.output_dir, f"image_seg_display.png"),
            image_seg_display.squeeze(0).clip(0, 255).astype(np.uint8),
        )
        self.assertEqual(
            compare_images(
                os.path.join(
                    root_dir,
                    "expected_outputs",
                    "visualization",
                    "example_image_seg_display.png",
                ),
                os.path.join(self.output_dir, f"image_seg_display.png"),
                0.001,
            ),
            None,
        )

    def test_get_seg_display_contract_for_2d_and_multichannel(self) -> None:
        seg_2d = np.ones((3, 4), dtype=np.float32)
        seg_rgb = np.zeros((3, 4, 2), dtype=np.float32)
        seg_rgb[..., 0] = 0.25
        seg_rgb[..., 1] = 0.75

        display_2d = get_seg_display(seg_2d)
        display_rgb = get_seg_display(seg_rgb)

        self.assertEqual(display_2d.shape, (3, 4, 4))
        self.assertEqual(display_rgb.shape, (3, 4, 4))
        self.assertEqual(display_2d.dtype, seg_2d.dtype)
        np.testing.assert_allclose(display_2d[..., 0], seg_2d)
        np.testing.assert_allclose(display_2d[..., 3], seg_2d)
        np.testing.assert_allclose(display_rgb[..., 3], 1.0)

    def test_get_tensorboard_image_seg_display_without_crossfield(self) -> None:
        image = torch.zeros((2, 3, 4, 5), dtype=torch.float32)
        seg = torch.zeros((2, 1, 4, 5), dtype=torch.float32)
        seg[:, 0, 1:3, 2:4] = 1

        output = get_tensorboard_image_seg_display(image, seg)

        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.dtype, image.dtype)
        self.assertTrue(torch.all(output[:, 0, 1:3, 2:4] == 1))

    def test_get_tensorboard_image_seg_display_validates_shapes(self) -> None:
        image = torch.zeros((1, 3, 4, 4))
        seg = torch.zeros((2, 1, 4, 4))

        with self.assertRaises(AssertionError):
            get_tensorboard_image_seg_display(image, seg)

    def test_crossfield_plot_image_contract(self) -> None:
        crossfield = np.zeros((8, 8, 4), dtype=np.float32)
        crossfield[..., 0] = 1

        output = get_image_plot_crossfield(crossfield, crossfield_stride=4, width=1)

        self.assertEqual(output.shape, (8, 8, 4))
        self.assertEqual(output.dtype, np.uint8)

    def test_plot_polygon_line_and_geometry_helpers(self) -> None:
        polygon = shapely.geometry.Polygon(
            [(0, 0), (3, 0), (3, 3), (0, 0)],
            holes=[[(1, 1), (2, 1), (2, 2), (1, 1)]],
        )
        line = shapely.geometry.LineString([(0, 0), (1, 1)])
        multiline = shapely.geometry.MultiLineString([[(1, 0), (1, 1)]])
        fig, ax = plt.subplots()

        plot_polygons(ax, [polygon], polygon_probs=[0.7])
        artists = plot_line_strings(ax, [line])
        returned = plot_geometries(ax, [polygon, line, multiline])

        self.assertEqual(len(artists), 1)
        self.assertEqual(len(returned), 2)
        self.assertGreaterEqual(len(ax.collections), 1)
        plt.close(fig)

    def test_plot_polygons_handles_empty_and_without_vertices(self) -> None:
        fig, ax = plt.subplots()
        polygon = shapely.geometry.Polygon([(0, 0), (2, 0), (2, 2), (0, 0)])

        self.assertIsNone(plot_polygons(ax, []))
        self.assertIsNone(plot_polygons(ax, [polygon], draw_vertices=False))

        plt.close(fig)

    def test_plot_geometries_rejects_unknown_geometry(self) -> None:
        fig, ax = plt.subplots()

        with self.assertRaises(NotImplementedError):
            plot_geometries(ax, [shapely.geometry.Point(0, 0)])

        plt.close(fig)

    def test_save_poly_viz_writes_file_with_optional_layers(self) -> None:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        polygon = np.array([[1, 1], [10, 1], [10, 10], [1, 1]], dtype=np.float32)
        crossfield = np.zeros((16, 16, 4), dtype=np.float32)
        crossfield[..., 0] = 1
        seg = np.zeros((16, 16, 4), dtype=np.float32)
        corners = [np.array([[1, 1], [10, 10]], dtype=np.float32)]
        out_filepath = os.path.join(self.output_dir, "poly.png")

        save_poly_viz(
            image,
            [polygon],
            out_filepath,
            corners=corners,
            crossfield=crossfield,
            polygon_probs=[0.5],
            seg=seg,
            dpi=10,
            crossfield_stride=8,
        )

        self.assertTrue(os.path.exists(out_filepath))

    def test_save_poly_viz_validates_polygon_probabilities(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        polygon = np.array([[1, 1], [4, 1], [4, 4], [1, 1]], dtype=np.float32)

        with self.assertRaises(AssertionError):
            save_poly_viz(image, [polygon], "unused.png", polygon_probs=[0.1, 0.2])
