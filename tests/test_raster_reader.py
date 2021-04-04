# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-04-02
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

from parameterized import parameterized
from pytorch_segmentation_models_trainer.tools.data_readers.raster_reader import \
    RasterFile
from pytorch_segmentation_models_trainer.tools.data_readers.vector_reader import \
    FileGeoDF
from pytorch_segmentation_models_trainer.utils.os_utils import (create_folder,
                                                                hash_file,
                                                                remove_folder)

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, 'testing_data')
test_list = [
    (
        RasterFile,
        {
            "file_name": os.path.join(root_dir, 'data', 'images', 'image1.png')
        },
        (3, 513, 513)
    ),
    (
        RasterFile,
        {
            "file_name": os.path.join(root_dir, 'data', 'images', 'image1.png'),
            "channels_last": True,
        },
        (513, 513, 3)
    ),
    (
        RasterFile,
        {
            "file_name": os.path.join(root_dir, 'data', 'labels', 'label1.png')
        },
        (1, 513, 513)
    )
]
suffix_dict = {
    "PNG": ".png",
    "GTiff": ".tif",
    "JPEG": ".jpg"
}

class Test_TestRasterReader(unittest.TestCase):

    def setUp(self):
        self.output_dir = create_folder(os.path.join(root_dir, 'test_output'))
    
    def tearDown(self):
        remove_folder(self.output_dir)

    @parameterized.expand(test_list)
    def test_instantiate_object(self, obj_class, params, expected_shape) -> None:
        obj = obj_class(**params)
        numpy_array = obj.read_as_numpy_array()
        self.assertEqual(numpy_array.shape, expected_shape)
    
    @parameterized.expand(['GTiff', 'JPEG'])
    def test_export_to(self, output_format):
        file_name = os.path.join(root_dir, 'data', 'images', 'image1.png')
        expected_output = os.path.join(
            root_dir, 'expected_outputs', 'raster_reader', 'image1'+suffix_dict[output_format]
        )
        raster = RasterFile(
            file_name=file_name
        )
        output_raster = raster.export_to(self.output_dir, output_format)
        self.assertEqual(
            hash_file(expected_output), hash_file(output_raster)
        )
    
    def test_build_polygon_mask_from_vector_layer(self):
        expected_output = os.path.join(root_dir, 'data', 'rasterize_data', 'labels', '10_polygon_mask.png')
        geo_df = FileGeoDF(
            file_name=os.path.join(root_dir, 'data', 'vectors', 'test_polygons2.geojson')
        )
        raster = RasterFile(
            file_name=os.path.join(root_dir, 'data', 'rasterize_data', 'images', '10.png')
        )
        output_mask = raster.build_polygon_mask_from_vector_layer(geo_df, self.output_dir)
        self.assertEqual(
            hash_file(expected_output), hash_file(output_mask)
        )
