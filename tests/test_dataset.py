# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-02-25
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

import albumentations as A
import hydra
import numpy as np
from hydra.experimental import compose, initialize
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    SegmentationDataset, load_augmentation_object)

from tests.utils import CustomTestCase

class Test_TestDataset(CustomTestCase):

    def test_create_instance(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="dataset.yaml",
                overrides=['input_csv_path='+self.csv_ds_file]
            )
            ds_from_cfg = hydra.utils.instantiate(cfg)
            ds_from_ref = SegmentationDataset(input_csv_path=self.csv_ds_file)
            self.assertEqual(
                ds_from_cfg.input_csv_path,
                ds_from_ref.input_csv_path
            )

    def test_load_image(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="dataset.yaml",
                overrides=['input_csv_path='+self.csv_ds_file]
            )
            ds_from_cfg = hydra.utils.instantiate(cfg)
            ds_from_ref = SegmentationDataset(input_csv_path=self.csv_ds_file)
            assert(
                np.array_equal(ds_from_cfg[0]['image'], ds_from_ref[0]['image']) and \
                np.array_equal(ds_from_cfg[0]['mask'], ds_from_ref[0]['mask']) 
            )

    def test_load_augmentations(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(config_name="augmentations.yaml")
            transformer = load_augmentation_object(cfg['augmentation_list'])
            assert isinstance(transformer, A.Compose)
