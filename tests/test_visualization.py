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
import unittest
import warnings

import hydra
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torch
from hydra import compose, initialize
from matplotlib.testing.compare import compare_images
from pytorch_segmentation_models_trainer.tools.visualization.crossfield_plot import (
    get_tensorboard_image_seg_display,
)
from pytorch_segmentation_models_trainer.utils.frame_field_utils import (
    compute_crossfield_to_plot,
)
from pytorch_segmentation_models_trainer.utils.math_utils import (
    compute_crossfield_c0c2,
    compute_crossfield_uv,
)
from pytorch_segmentation_models_trainer.utils.os_utils import (
    create_folder,
    hash_file,
    remove_folder,
)

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")

frame_field_root_dir = os.path.join(
    current_dir, "testing_data", "data", "frame_field_data"
)

from matplotlib.testing.compare import compare_images


class Test_Visualization(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self.output_dir = create_folder(os.path.join(root_dir, "test_output"))

    def tearDown(self):
        remove_folder(self.output_dir)

    def test_seg_display_real_example(self) -> None:
        csv_path = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        with initialize(config_path="./test_configs"):
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
            image_seg_display.squeeze(0),
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
            image_seg_display.squeeze(0),
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
