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
from albumentations.pytorch.transforms import ToTensorV2
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

from tests.utils import BasicTestCase, CustomTestCase, get_config_from_hydra

current_dir = os.path.dirname(__file__)
detection_root_dir = os.path.join(
    current_dir, "testing_data", "data", "detection_data", "geo"
)
polygon_rnn_root_dir = os.path.join(
    current_dir, "testing_data", "data", "polygon_rnn_data"
)
csv_path = os.path.join(detection_root_dir, "dsg_dataset.csv")
poly_csv_path = os.path.join(polygon_rnn_root_dir, "polygonrnn_dataset.csv")
torch.manual_seed(0)


class Test_ModPolyMapperModel(BasicTestCase):
    def _get_model(
        self, backbone_trainable_layers=3, pretrained=False
    ) -> ModPolyMapper:
        return ModPolyMapper(
            num_classes=2,
            backbone_trainable_layers=backbone_trainable_layers,
            pretrained=pretrained,
        )

    def _get_dataloader(self) -> torch.utils.data.DataLoader:
        obj_det_ds = ObjectDetectionDataset(
            input_csv_path=csv_path,
            root_dir=os.path.dirname(csv_path),
            augmentation_list=A.Compose(
                [A.CenterCrop(512, 512), A.Normalize(), ToTensorV2()],
                bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
            ),
        )
        polygon_rnn_dataset = PolygonRNNDataset(
            input_csv_path=poly_csv_path,
            sequence_length=60,
            root_dir=polygon_rnn_root_dir,
            augmentation_list=A.Compose([A.Normalize(), ToTensorV2()]),
        )
        return {
            "object_detection_dataloader": torch.utils.data.DataLoader(
                obj_det_ds,
                batch_size=2,
                shuffle=False,
                drop_last=True,
                num_workers=1,
                collate_fn=obj_det_ds.collate_fn,
            ),
            "polygon_rnn_dataloader": torch.utils.data.DataLoader(
                polygon_rnn_dataset,
                batch_size=2,
                shuffle=False,
                drop_last=True,
                num_workers=1,
            ),
        }

    def test_model_forward(self) -> None:
        model = self._get_model()
        sample = torch.randn([2, 3, 256, 256])
        model.eval()
        with torch.no_grad():
            out = model(sample)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0].keys()), 8)

    def test_model_backwards(self) -> None:
        model = self._get_model(backbone_trainable_layers=5, pretrained=True)
        data_loader = self._get_dataloader()
        model.train()
        obj_det_images, obj_det_targets, index = next(
            iter(data_loader["object_detection_dataloader"])
        )
        polygon_rnn_batch = next(iter(data_loader["polygon_rnn_dataloader"]))
        losses, acc = model(obj_det_images, obj_det_targets, polygon_rnn_batch)
        loss = sum(losses.values())
        loss.backward()
        for n, x in model.named_parameters():
            assert x.grad is not None, f"No gradient for {n}"

    @unittest.skipIf(
        not torch.cuda.is_available(),
        reason="No GPU available, test is too memory expensive to be run on CPU",
    )
    @parameterized.expand(
        [
            (
                "experiment_mod_polymapper.yaml",
                ["+pl_trainer.fast_dev_run=true", "+pl_model.perform_evaluation=true"],
            ),
            ("experiment_mod_polymapper_with_callback.yaml", None),
        ]
    )
    def test_pl_model_train(self, experiment_name, extra_overrides=None) -> None:
        extra_overrides = extra_overrides if extra_overrides is not None else []
        cfg = get_config_from_hydra(
            config_name=experiment_name,
            overrides_list=[
                f"train_dataset.object_detection.input_csv_path={csv_path}",
                f"train_dataset.object_detection.root_dir={detection_root_dir}",
                f"train_dataset.polygon_rnn.input_csv_path={poly_csv_path}",
                f"train_dataset.polygon_rnn.root_dir={polygon_rnn_root_dir}",
                f"val_dataset.object_detection.input_csv_path={csv_path}",
                f"val_dataset.object_detection.root_dir={detection_root_dir}",
                f"val_dataset.polygon_rnn.input_csv_path={poly_csv_path}",
                f"val_dataset.polygon_rnn.root_dir={polygon_rnn_root_dir}",
                "pl_trainer.gpus=1",
            ]
            + extra_overrides,
        )
        trainer = train(cfg)
        return
