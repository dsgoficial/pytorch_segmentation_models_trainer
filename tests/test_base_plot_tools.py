# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-07-15
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
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    ObjectDetectionDataset,
)
from pytorch_segmentation_models_trainer.tools.visualization.base_plot_tools import (
    batch_denormalize_tensor,
    denormalize_np_array,
    generate_bbox_visualization,
    generate_visualization,
    show,
    visualize_image_with_bboxes,
)
from tests.utils import BasicTestCase

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")
detection_root_dir = os.path.join(current_dir, "testing_data", "data", "detection_data")


class Test_BasePlotTools(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.output_dir = self.make_temp_dir()

    def test_visualize_image_with_bboxes(self):
        csv_path = os.path.join(detection_root_dir, "geo", "dsg_dataset.csv")
        obj_det_ds = ObjectDetectionDataset(
            input_csv_path=csv_path,
            root_dir=os.path.dirname(csv_path),
            augmentation_list=A.Compose(
                [A.RandomCrop(512, 512), ToTensorV2()],
                bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
            ),
        )
        dataloader = torch.utils.data.DataLoader(
            obj_det_ds, batch_size=4, shuffle=False, collate_fn=obj_det_ds.collate_fn
        )
        images, targets, indexes = next(iter(dataloader))
        output = visualize_image_with_bboxes(
            images, [target["boxes"] for target in targets]
        )
        self.assertEqual(len(output), 4)

    def test_show_accepts_single_tensor(self):
        image = torch.zeros((3, 4, 4), dtype=torch.float32)

        show(image)

        self.assertGreater(len(plt.get_fignums()), 0)
        plt.close("all")

    def test_denormalize_np_array_uses_defaults_and_preserves_shape(self):
        image = np.zeros((3, 2, 2), dtype=np.float32)

        output = denormalize_np_array(image)

        self.assertEqual(output.shape, image.shape)
        np.testing.assert_allclose(output[:, 0, 0], [0.485, 0.456, 0.406])

    def test_batch_denormalize_tensor_contract_and_clip(self):
        tensor = torch.ones((2, 3, 2, 2), dtype=torch.float32)
        mean = torch.zeros((2, 3), dtype=torch.float32)
        std = torch.ones((2, 3), dtype=torch.float32) * 2

        output = batch_denormalize_tensor(
            tensor,
            mean=mean,
            std=std,
            clip_range=(0, 1),
            output_type=torch.float64,
        )

        self.assertEqual(output.shape, tensor.shape)
        self.assertEqual(output.dtype, torch.float64)
        self.assertTrue(torch.all(output == 1))

    def test_batch_denormalize_tensor_without_clip_returns_output_type(self):
        tensor = torch.ones((1, 3, 2, 2), dtype=torch.float32)

        output = batch_denormalize_tensor(tensor, output_type=torch.float64)

        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(output.shape, tensor.shape)

    def test_batch_denormalize_tensor_validates_dimensions(self):
        with self.assertRaises(AssertionError):
            batch_denormalize_tensor(
                torch.zeros((1, 3, 2, 2)),
                mean=torch.zeros((3,)),
                std=torch.zeros((3,)),
            )

    def test_generate_visualization_returns_axes_and_figure(self):
        axes, fig = generate_visualization(
            fig_title="Example",
            image=np.zeros((2, 2)),
            mask=np.ones((2, 2)),
        )

        self.assertEqual(len(np.ravel(axes)), 2)
        self.assertEqual(fig._suptitle.get_text(), "Example")
        plt.close(fig)

    def test_generate_bbox_visualization_adds_patches_and_scores(self):
        fig, ax = plt.subplots()
        detection = {
            "boxes": np.array([[0, 0, 2, 3], [2, 1, 4, 4]], dtype=np.float32),
            "labels": np.array([1, 2]),
            "scores": np.array([0.5, 0.75], dtype=np.float32),
        }

        generate_bbox_visualization(ax, detection, show_scores=True)

        self.assertEqual(len(ax.patches), 2)
        self.assertEqual(len(ax.texts), 2)
        self.assertEqual(ax.texts[0].get_text(), "50.00%")
        plt.close(fig)

    def test_generate_bbox_visualization_can_skip_scores(self):
        fig, ax = plt.subplots()
        detection = {
            "boxes": np.array([[0, 0, 2, 3]], dtype=np.float32),
            "labels": np.array([1]),
            "scores": np.array([0.5], dtype=np.float32),
        }

        generate_bbox_visualization(ax, detection, show_scores=False)

        self.assertEqual(len(ax.patches), 1)
        self.assertEqual(len(ax.texts), 0)
        plt.close(fig)
