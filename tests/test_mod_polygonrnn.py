# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-08-16
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
import subprocess
import unittest
from importlib import import_module

import albumentations as A
import hydra
import numpy as np
import segmentation_models_pytorch as smp
import torch
from hydra.experimental import compose, initialize
from parameterized import parameterized
from pytorch_segmentation_models_trainer.custom_models import models as pytorch_smt_cm
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    ModPolyMapperDataset,
    ObjectDetectionDataset,
    PolygonRNNDataset,
)
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldModel,
    FrameFieldSegmentationPLModel,
)
from pytorch_segmentation_models_trainer.model_loader.polygon_rnn_model import (
    PolygonRNN,
)
from pytorch_segmentation_models_trainer.custom_models.mod_polymapper.modpolymapper import (
    ModPolyMapper,
)
from pytorch_segmentation_models_trainer.train import train

from tests.utils import CustomTestCase

current_dir = os.path.dirname(__file__)
detection_root_dir = os.path.join(
    current_dir, "testing_data", "data", "detection_data", "geo"
)
polygon_rnn_root_dir = os.path.join(
    current_dir, "testing_data", "data", "polygon_rnn_data"
)
torch.manual_seed(0)


class Test_ModPolyMapperModel(unittest.TestCase):
    def _get_model(self):
        return ModPolyMapper(num_classes=2, pretrained=False)

    def test_mod_polymapper_model(self) -> None:
        model = self._get_model()
        sample = torch.randn([2, 3, 256, 256])
        model.eval()
        with torch.no_grad():
            out = model(sample)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0].keys()), 4)

    def test_create_inference_from_model(self) -> None:
        csv_path = os.path.join(detection_root_dir, "dsg_dataset.csv")
        poly_csv_path = os.path.join(polygon_rnn_root_dir, "polygonrnn_dataset.csv")
        ds = ModPolyMapperDataset(
            object_detection_dataset=ObjectDetectionDataset(
                input_csv_path=csv_path,
                root_dir=os.path.dirname(csv_path),
                augmentation_list=A.Compose(
                    [A.CenterCrop(512, 512), A.Normalize(), A.pytorch.ToTensorV2()],
                    bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
                ),
            ),
            polygon_rnn_dataset=PolygonRNNDataset(
                input_csv_path=poly_csv_path,
                sequence_length=60,
                root_dir=polygon_rnn_root_dir,
                augmentation_list=[A.Normalize(), A.pytorch.ToTensorV2()],
            ),
        )
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=2,
            shuffle=False,
            drop_last=True,
            num_workers=1,
            collate_fn=ds.collate_fn,
        )
        model = self._get_model()
        with torch.no_grad():
            image, target = next(iter(data_loader))
            output = model(image, target)
        self.assertEqual(len(output), 6)
