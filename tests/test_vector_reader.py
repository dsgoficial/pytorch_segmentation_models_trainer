# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-04-01
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
import geopandas

from geopandas.testing import geom_equals
from parameterized import parameterized
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from pytorch_segmentation_models_trainer.tools.data_readers.vector_reader import (
    FileGeoDF, GeomTypeEnum, handle_features, handle_geometry
)
from shapely.geometry import Polygon, LineString, LinearRing
current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, 'testing_data')
test_list = [
    (
        FileGeoDF,
        {
            "file_name": os.path.join(root_dir, 'data', 'vectors', 'test_polygons.geojson')
        }
    )
]
class Test_TestVectorReader(unittest.TestCase):

    @parameterized.expand(test_list)
    def test_instantiate_object(self, obj_class, params) -> None:
        obj = obj_class(**params)
        geo_df = obj.get_geo_df()
        assert len(geo_df) > 0
    
    @parameterized.expand(
        [
            (
                Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
                GeomTypeEnum.LINE,
                LineString([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
            ),
            (
                Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
                GeomTypeEnum.POINT,
                MultiPoint([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
            ),
            (
                LineString([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
                GeomTypeEnum.POINT,
                MultiPoint([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
            ),
            (
                Polygon(
                    [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
                    holes=[
                        LinearRing(
                            [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75], [0.25, 0.25]]
                        )
                    ]
                ),
                GeomTypeEnum.LINE,
                MultiLineString(
                    [
                        [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
                        [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75], [0.25, 0.25]]
                    ]
                )
            ),
        ]
    )
    def test_handle_geometry(self, input_geom, output_type, expected_output) -> None:
        output = handle_geometry(input_geom, output_type)
        assert output.equals(expected_output)

    @parameterized.expand(
        [
            (
                GeomTypeEnum.LINE,
                os.path.join(root_dir, 'expected_outputs', 'vector_reader', 'handle_features_line_output.geojson')
            ),
            (
                GeomTypeEnum.POINT,
                os.path.join(root_dir, 'expected_outputs', 'vector_reader', 'handle_features_point_output.geojson')
            )
        ]
    )
    def test_handle_features(self, output_type, expected_output) -> None:
        input_gdf = geopandas.read_file(
            filename=os.path.join(root_dir, 'data', 'vectors', 'test_polygons2.geojson')
        )
        output_features = handle_features(input_gdf['geometry'], output_type)
        expected_output_gdf = geopandas.read_file(filename=expected_output)
        assert geom_equals(expected_output_gdf['geometry'], output_features['geometry'])



