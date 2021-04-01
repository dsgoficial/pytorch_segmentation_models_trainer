# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-03-01
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
from pytorch_segmentation_models_trainer.tools.data_readers.vector_reader import (
    FileGeoDF
)
current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, 'testing_data', 'data')
test_list = [
    (
        FileGeoDF,
        {
            "file_name": os.path.join(root_dir, 'vectors', 'test_polygons.geojson')
        }
    )
]
class Test_TestVectorReader(unittest.TestCase):

    @parameterized.expand(test_list)
    def test_instantiate_object(self, obj_class, params) -> None:
        obj = obj_class(**params)
        geo_df = obj.get_geo_df()
        assert len(geo_df) > 0

