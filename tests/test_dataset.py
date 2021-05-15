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

import os
import albumentations as A
import hydra
import numpy as np
from hydra.experimental import compose, initialize
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    SegmentationDataset, load_augmentation_object)

from tests.utils import CustomTestCase

current_dir = os.path.dirname(__file__)
frame_field_root_dir = os.path.join(
    current_dir, 'testing_data', 'data', 'frame_field_data')

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
    
    def test_load_image_relative_path(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="dataset.yaml",
                overrides=[
                    'input_csv_path='+self.csv_ds_file_without_root,
                    '+root_dir='+self.root_dir
                ]
            )
            ds_from_cfg = hydra.utils.instantiate(cfg)
            ds_from_ref = SegmentationDataset(
                input_csv_path=self.csv_ds_file_without_root,
                root_dir=self.root_dir
            )
            assert(
                np.array_equal(ds_from_cfg[0]['image'], ds_from_ref[0]['image']) and \
                np.array_equal(ds_from_cfg[0]['mask'], ds_from_ref[0]['mask']) 
            )

    def test_load_augmentations(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(config_name="augmentations.yaml")
            transformer = load_augmentation_object(cfg['augmentation_list'])
            assert isinstance(transformer, A.Compose)
    
    def test_dataset_size_limit(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="dataset.yaml",
                overrides=[
                    'input_csv_path='+self.csv_ds_file_without_root,
                    '+root_dir='+self.root_dir,
                    '+n_first_rows_to_read=2'
                ]
            )
            ds_from_cfg = hydra.utils.instantiate(cfg)
            self.assertEqual(len(ds_from_cfg), 2)
    
    def test_create_frame_field_dataset_instance(self):
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
        self.assertEqual(len(frame_field_ds), 12)
        self.assertEqual(frame_field_ds[0]['image'].shape,(571, 571, 3))
        self.assertEqual(frame_field_ds[0]['gt_polygons_image'].shape,(571, 571, 3))
        self.assertEqual(frame_field_ds[0]['gt_crossfield_angle'].shape,(571, 571))
