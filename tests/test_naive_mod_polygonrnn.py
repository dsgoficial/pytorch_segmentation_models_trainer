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

import albumentations as A
import hydra
import torch
from albumentations.pytorch.transforms import ToTensorV2
from hydra.experimental import compose, initialize
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    NaiveModPolyMapperDataset,
    ObjectDetectionDataset,
    PolygonRNNDataset,
)

from tests.utils import BasicTestCase

current_dir = os.path.dirname(__file__)
detection_root_dir = os.path.join(
    current_dir, "testing_data", "data", "detection_data", "geo"
)
polygon_rnn_root_dir = os.path.join(
    current_dir, "testing_data", "data", "polygon_rnn_data"
)


class Test_NaiveModPolymapperModel(BasicTestCase):
    def _get_model(self):
        with initialize(config_path="./test_configs"):
            cfg = compose(config_name="naive_mod_polymapper_model.yaml")
            model = hydra.utils.instantiate(cfg, _recursive_=False)
        return model

    def test_naive_mod_polymapper_model(self) -> None:
        model = self._get_model()
        sample = torch.ones([2, 3, 256, 256])
        model.eval()
        with torch.no_grad():
            out = model(sample)
        self.assertEqual(len(out), 2)

    def test_create_inference_from_model(self) -> None:
        csv_path = os.path.join(detection_root_dir, "dsg_dataset.csv")
        poly_csv_path = os.path.join(polygon_rnn_root_dir, "polygonrnn_dataset.csv")
        ds = NaiveModPolyMapperDataset(
            object_detection_dataset=ObjectDetectionDataset(
                input_csv_path=csv_path,
                root_dir=os.path.dirname(csv_path),
                augmentation_list=A.Compose(
                    [A.CenterCrop(512, 512), A.Normalize(), ToTensorV2()],
                    bbox_params=A.BboxParams(format="coco", label_fields=["labels"]),
                ),
            ),
            polygon_rnn_dataset=PolygonRNNDataset(
                input_csv_path=poly_csv_path,
                sequence_length=60,
                root_dir=polygon_rnn_root_dir,
                augmentation_list=[A.Normalize(), ToTensorV2()],
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
            image, target, indexes = next(iter(data_loader))
            output = model(image, target)
        self.assertEqual(len(output), 6)

    # @parameterized.expand([("experiment_naive_mod_polymapper.yaml",)])
    # def test_train_naive_mod_polymapper(self, config_name) -> None:
    #     csv_path = os.path.join(detection_root_dir, "dsg_dataset.csv")
    #     with initialize(config_path="./test_configs"):
    #         cfg = compose(
    #             config_name=config_name,
    #             overrides=[
    #                 "train_dataset.input_csv_path=" + csv_path,
    #                 "train_dataset.root_dir=" + detection_root_dir,
    #                 "val_dataset.input_csv_path=" + csv_path,
    #                 "val_dataset.root_dir=" + detection_root_dir,
    #             ],
    #         )
    #         trainer = train(cfg)
