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
import torch
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    ObjectDetectionDataset,
)
from pytorch_segmentation_models_trainer.tools.visualization.base_plot_tools import (
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
