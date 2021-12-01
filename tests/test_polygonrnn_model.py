# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-08-02
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
from pytorch_segmentation_models_trainer.dataset_loader.dataset import PolygonRNNDataset
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldModel,
    FrameFieldSegmentationPLModel,
)
from pytorch_segmentation_models_trainer.model_loader.polygon_rnn_model import (
    PolygonRNN,
)
from pytorch_segmentation_models_trainer.train import train
from tests.utils import BasicTestCase

current_dir = os.path.dirname(__file__)
polygon_rnn_root_dir = os.path.join(
    current_dir, "testing_data", "data", "polygon_rnn_data"
)


class Test_PolygonRNNModel(BasicTestCase):
    def test_create_instance(self) -> None:
        polygonrnn = PolygonRNN()
        return True

    def _get_dataloader(self):
        csv_path = os.path.join(polygon_rnn_root_dir, "polygonrnn_dataset.csv")
        polygon_rnn_ds = PolygonRNNDataset(
            input_csv_path=csv_path,
            sequence_length=20,
            root_dir=polygon_rnn_root_dir,
            augmentation_list=[A.Normalize(), ToTensorV2()],
        )
        data_loader = torch.utils.data.DataLoader(
            polygon_rnn_ds, batch_size=2, shuffle=False, drop_last=True, num_workers=1
        )
        return data_loader

    def test_create_inference_from_model(self) -> None:
        data_loader = self._get_dataloader()
        polygonrnn = PolygonRNN()
        with torch.no_grad():
            batch = next(iter(data_loader))
            output = polygonrnn(batch["image"], batch["x1"], batch["x2"], batch["x3"])
        self.assertEqual(output.shape, (2, 18, 787))

    def test_model_backward(self) -> None:
        data_loader = self._get_dataloader()
        model = PolygonRNN()
        model.train()
        batch = next(iter(data_loader))
        output = model(batch["image"], batch["x1"], batch["x2"], batch["x3"])
        loss = output.mean()
        loss.backward()
        for n, x in model.named_parameters():
            assert x.grad is not None, f"No gradient for {n}"

    def test_train_polygon_rnn_model(self) -> None:
        csv_path = os.path.join(polygon_rnn_root_dir, "polygonrnn_dataset.csv")
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_polygonrnn.yaml",
                overrides=[
                    f"train_dataset.input_csv_path={csv_path}",
                    f"train_dataset.root_dir={polygon_rnn_root_dir}",
                    f"val_dataset.input_csv_path={csv_path}",
                    f"val_dataset.root_dir={polygon_rnn_root_dir}",
                    "+pl_trainer.fast_dev_run=true",
                ],
            )
            trainer = train(cfg)

    def test_train_polygon_rnn_model_with_callback(self) -> None:
        csv_path = os.path.join(polygon_rnn_root_dir, "polygonrnn_dataset.csv")
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="experiment_polygonrnn_with_callback.yaml",
                overrides=[
                    f"train_dataset.input_csv_path={csv_path}",
                    f"train_dataset.root_dir={polygon_rnn_root_dir}",
                    f"val_dataset.input_csv_path={csv_path}",
                    f"val_dataset.root_dir={polygon_rnn_root_dir}",
                ],
            )
            trainer = train(cfg)
        return
