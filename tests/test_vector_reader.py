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
from pathlib import Path
import warnings

import geopandas
from geopandas.testing import geom_equals, geom_almost_equals
from parameterized import parameterized
from pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader import (
    BatchFileGeoDF,
    COCOGeoDF,
    FileGeoDF,
    GeomTypeEnum,
    handle_features,
    handle_geometry,
)
from shapely.geometry import LinearRing, LineString, Polygon
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")
test_list = [
    (
        FileGeoDF,
        {
            "file_name": os.path.join(
                root_dir, "data", "vectors", "test_polygons.geojson"
            )
        },
    )
]


class Test_VectorReader(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)

    @parameterized.expand(test_list)
    def test_instantiate_object(self, obj_class, params) -> None:
        obj = obj_class(**params)
        geo_df = obj.get_geo_df()
        assert len(geo_df) > 0

    def test_instantiate_batch_file_reader(self) -> None:
        test_root_dir = os.path.join(root_dir, "data", "vectors")
        obj = BatchFileGeoDF(root_dir=test_root_dir)
        json_key_list = [
            str(p).split(".")[0] for p in Path(test_root_dir).glob(f"**/*.geojson")
        ]
        for key in json_key_list:
            assert len(obj.get_geodf_item(key)) > 0

    @parameterized.expand(
        [
            (
                Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
                GeomTypeEnum.LINE,
                LineString([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
            ),
            (
                Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
                GeomTypeEnum.POINT,
                MultiPoint([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
            ),
            (
                LineString([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
                GeomTypeEnum.POINT,
                MultiPoint([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
            ),
            (
                Polygon(
                    [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
                    holes=[
                        LinearRing(
                            [
                                [0.25, 0.25],
                                [0.75, 0.25],
                                [0.75, 0.75],
                                [0.25, 0.75],
                                [0.25, 0.25],
                            ]
                        )
                    ],
                ),
                GeomTypeEnum.LINE,
                MultiLineString(
                    [
                        [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
                        [
                            [0.25, 0.25],
                            [0.75, 0.25],
                            [0.75, 0.75],
                            [0.25, 0.75],
                            [0.25, 0.25],
                        ],
                    ]
                ),
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
                os.path.join(
                    root_dir,
                    "expected_outputs",
                    "vector_reader",
                    "handle_features_line_output.geojson",
                ),
            ),
            (
                GeomTypeEnum.POINT,
                os.path.join(
                    root_dir,
                    "expected_outputs",
                    "vector_reader",
                    "handle_features_point_output.geojson",
                ),
            ),
        ]
    )
    def test_handle_features(self, output_type, expected_output) -> None:
        input_gdf = geopandas.read_file(
            filename=os.path.join(root_dir, "data", "vectors", "test_polygons2.geojson")
        )
        output_features = handle_features(input_gdf["geometry"], output_type)
        expected_output_gdf = geopandas.read_file(filename=expected_output)
        assert geom_equals(expected_output_gdf["geometry"], output_features["geometry"])

    def test_instantiate_coco_geo_df(self) -> None:
        input_gdf = COCOGeoDF(
            file_name=os.path.join(
                root_dir, "data", "build_masks_data", "annotation.json"
            )
        )
        for key in [160847, 232566]:
            output_features_gdf = input_gdf.get_geodf_item(key).gdf
            expected_output_gdf = geopandas.read_file(
                filename=os.path.join(
                    root_dir, "expected_outputs", "vector_reader", f"{key}.geojson"
                )
            )
            assert geom_almost_equals(
                expected_output_gdf["geometry"], output_features_gdf["geometry"]
            )
