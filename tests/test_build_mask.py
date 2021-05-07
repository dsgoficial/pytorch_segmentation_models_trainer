# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-05-06
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

import hydra
import numpy as np
import pandas as pd
from hydra.experimental import compose, initialize
from parameterized import parameterized
from pytorch_segmentation_models_trainer.build_mask import build_masks
from pytorch_segmentation_models_trainer.tools.data_readers.raster_reader import (
    MaskOutputTypeEnum, RasterFile)
from pytorch_segmentation_models_trainer.tools.data_readers.vector_reader import (
    FileGeoDF, GeomTypeEnum)
from pytorch_segmentation_models_trainer.tools.mask_building.mask_builder import (
    MaskBuilder, build_destination_dirs)
from pytorch_segmentation_models_trainer.utils.os_utils import (create_folder,
                                                                hash_file,
                                                                remove_folder)

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, 'testing_data')

suffix_dict = {
    "PNG": ".png",
    "GTiff": ".tif",
    "JPEG": ".jpg"
}

class Test_TestBuildMask(unittest.TestCase):

    def setUp(self):
        self.output_dir = create_folder(os.path.join(root_dir, 'test_output'))
        self.replicated_dir = create_folder(os.path.join(root_dir, '..', 'replicated_dirs'))
    
    def tearDown(self):
        remove_folder(self.output_dir)
        remove_folder(self.replicated_dir)

    def test_build_output_dirs_raises_exception(self):
        output_base_path = os.path.join(self.output_dir, 'replicated_dirs')
        with self.assertRaises(Exception) as context:
            build_destination_dirs(
                input_base_path=root_dir,
                output_base_path=output_base_path
            )
        self.assertTrue(
            "input path must not be in output_path" in str(context.exception)
        )
    
    def test_build_output_dirs(self):
        output_list = build_destination_dirs(
            input_base_path=root_dir,
            output_base_path=self.replicated_dir
        )
        assert(len(output_list) > 0)
        input_structure = [dirpath.replace(root_dir, '') for dirpath, _, __ in os.walk(root_dir)]
        output_structure = [dirpath.replace(self.replicated_dir, '') for dirpath, _, __ in os.walk(self.replicated_dir)]
        assert(len(input_structure) == len(output_structure))
        for item in output_structure:
            assert(item in input_structure)
    
    def test_build_masks(self):
        with initialize(config_path="./test_configs"):
            image_dir = os.path.join(root_dir, 'data', 'build_masks_data', 'images')
            expected_output_path = os.path.join(root_dir, 'expected_outputs', 'build_masks')
            cfg = compose(
                config_name="build_mask.yaml",
                overrides=[
                    'root_dir='+self.output_dir,
                    'output_csv_path='+self.output_dir,
                    'image_root_dir='+image_dir,
                    'geo_df.file_name='+os.path.join(
                        root_dir, 'data', 'build_masks_data', 'buildings.geojson'
                    )
                ]
            )
            print(cfg)
            csv_output = build_masks(cfg)
            expected_df = pd.read_csv(os.path.join(expected_output_path, 'dsg_dataset.csv')).sort_values('image')
            output_df = pd.read_csv(csv_output).sort_values('image')
            pd.testing.assert_frame_equal(
                expected_df.reset_index(drop=True),
                output_df.reset_index(drop=True)
            )
            selected_keys = [i for i in expected_df.columns if '_mask' in i]
            for index, row in expected_df.iterrows():
                output_row = output_df.loc[expected_df['image'] == row['image']]
                for key in selected_keys:
                    np.testing.assert_array_equal(
                        RasterFile(
                            file_name=os.path.join(expected_output_path, row[key])
                        ).read_as_numpy_array(),
                        RasterFile(
                            file_name=os.path.join(self.output_dir, output_row[key].item())
                        ).read_as_numpy_array()
                    )
