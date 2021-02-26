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
import shutil
import unittest

import albumentations as A
import hydra
import numpy as np
from hydra.experimental import compose, initialize
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    SegmentationDataset, load_augmentation_object)


def get_file_list(dir_path, extension):
    output_list = []
    for root, dirs, files in os.walk(dir_path):
        output_list += [os.path.join(root,f) for f in files if f.endswith(extension)]
    return sorted(output_list)

def create_csv_file(file_path, image_list, label_list):
    csv_text = 'id,image_path,label_path,rows,columns\n'
    for idx, i in enumerate(image_list):
        csv_text += f"{idx},{i},{label_list[idx]},512,512\n"
    with open(file_path, 'w') as csv_file:
        csv_file.write(csv_text)
    return file_path

class Test_TestDataset(unittest.TestCase):
    # cs = ConfigStore.instance()
    # cs.store(name="dataset", node=DS)

    def setUp(self):
        current_dir = os.path.dirname(__file__)
        image_list = get_file_list(
            os.path.join(current_dir, 'testing_data', 'data', 'images'),
            '.png'
        )
        label_list = get_file_list(
            os.path.join(current_dir, 'testing_data', 'data', 'labels'),
            '.png'
        )
        label_list = get_file_list(
            os.path.join(current_dir, 'testing_data', 'data', 'labels'),
            '.png'
        )
        self.csv_ds_file = create_csv_file(
            os.path.join(current_dir, 'testing_data', 'csv_train_ds.csv'),
            image_list[0:5],
            label_list[0:5]
        )

    def tearDown(self):
        os.remove(self.csv_ds_file)
        outputs_path = os.path.join(
                os.path.dirname(__file__),
                '..',
                'outputs'
            )
        if os.path.exists(outputs_path):
            shutil.rmtree(outputs_path)
    
    def test_create_instance(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(config_name="dataset.yaml", overrides=['input_csv_path='+self.csv_ds_file])
            ds_from_cfg = hydra.utils.instantiate(cfg)
            ds_from_ref = SegmentationDataset(input_csv_path=self.csv_ds_file)
            self.assertEqual(
                ds_from_cfg.input_csv_path,
                ds_from_ref.input_csv_path
            )

    def test_load_image(self) -> None:
        with initialize(config_path="./test_configs"):
            cfg = compose(config_name="dataset.yaml", overrides=['input_csv_path='+self.csv_ds_file])
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
