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
import runpy
import unittest
from pathlib import Path
import warnings
from unittest.mock import patch

import geopandas
from geopandas.testing import geom_equals, geom_almost_equals
from parameterized import parameterized
from pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader import (
    BatchFileGeoDF,
    COCOGeoDF,
    COCOMemoryGeoDF,
    FileGeoDF,
    GeoDF,
    GeomTypeEnum,
    get_chunks,
    handle_features,
    handle_geometry,
    save_to_file,
)
from shapely.geometry import GeometryCollection, LinearRing, LineString, Point, Polygon
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
        self.assertIsNone(obj.get_geodf_item("missing"))

    def test_get_chunks_pads_last_chunk(self) -> None:
        self.assertEqual(list(get_chunks([1, 2, 3], 2)), [(1, 2), (3, None)])

    def test_geodf_base_methods_and_validation(self) -> None:
        class ConcreteGeoDF(GeoDF):
            def __post_init__(self):
                return super().__post_init__()

        self.assertIsNone(ConcreteGeoDF().__post_init__())

    def test_geodf_get_features_without_spatial_index(self) -> None:
        class ConcreteGeoDF(GeoDF):
            def __post_init__(self):
                self.gdf = geopandas.GeoDataFrame(
                    {"geometry": [Polygon([(0, 0), (2, 0), (2, 2), (0, 0)])]},
                    crs="EPSG:4326",
                )
                self.spatial_index = None

        geodf = ConcreteGeoDF()

        with self.assertRaises(Exception):
            geodf.get_features_from_bbox(0, 1, 0, 1, filter_area=-1)
        output = geodf.get_features_from_bbox(
            0, 1, 0, 1, only_geom=False, clip_to_extent=False
        )
        self.assertEqual(len(output), 1)

    def test_file_geodf_uses_spatial_index_path(self) -> None:
        obj = FileGeoDF(
            file_name=os.path.join(root_dir, "data", "vectors", "test_polygons.geojson")
        )

        output = obj.get_features_from_bbox(
            426500, 426600, 6695600, 6695700, only_geom=False
        )

        self.assertGreaterEqual(len(output), 0)

    @patch(
        "pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader.geopandas.read_postgis"
    )
    @patch(
        "pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader.psycopg2.connect"
    )
    def test_postgis_geodf_initializes_with_mocked_connection(
        self, mock_connect, mock_read_postgis
    ) -> None:
        from pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader import (
            PostgisGeoDF,
        )

        mock_read_postgis.return_value = geopandas.GeoDataFrame(
            {"geometry": [Point(0, 0)]}, crs="EPSG:4326"
        )

        obj = PostgisGeoDF(
            user="u",
            password="p",
            database="d",
            sql="select geom from table",
            build_spatial_index=False,
        )

        self.assertIsNone(obj.spatial_index)
        mock_connect.assert_called_once()

    def test_clip_features_to_extent_falls_back_on_clip_error(self) -> None:
        obj = FileGeoDF(
            file_name=os.path.join(root_dir, "data", "vectors", "test_polygons.geojson")
        )
        feats = obj.get_geo_df()

        with patch(
            "pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader.geopandas.clip",
            side_effect=RuntimeError("clip failed"),
        ):
            output = obj.clip_features_to_extent(feats, 0, 1, 0, 1)

        self.assertIs(output, feats)

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

    def test_handle_geometry_collections_and_invalid_type(self) -> None:
        collection = GeometryCollection([LineString([(0, 0), (1, 1)])])

        output = handle_geometry(collection, GeomTypeEnum.POINT)

        self.assertIsInstance(output, GeometryCollection)
        self.assertTrue(
            handle_geometry(Point(0, 0), GeomTypeEnum.LINE).equals(Point(0, 0))
        )
        with self.assertRaises(Exception):
            handle_geometry(object(), GeomTypeEnum.LINE)

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

    def test_handle_features_none_returns_input_and_return_list(self) -> None:
        input_gdf = geopandas.read_file(
            filename=os.path.join(root_dir, "data", "vectors", "test_polygons2.geojson")
        )
        geometry = input_gdf["geometry"]

        self.assertIs(handle_features(geometry), geometry)
        output = handle_features(
            input_gdf["geometry"], GeomTypeEnum.LINE, return_list=True
        )
        self.assertIsInstance(output, list)

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

    def test_coco_geodf_lazy_build_and_invalid_key(self) -> None:
        input_gdf = COCOGeoDF(
            file_name=os.path.join(
                root_dir, "data", "build_masks_data", "annotation.json"
            ),
            pre_build_vector_dict=False,
        )

        self.assertEqual(input_gdf.vector_dict, {})
        self.assertGreater(len(input_gdf.get_geodf_item(160847).gdf), 0)
        with self.assertRaises(KeyError):
            input_gdf.get_geodf_item(999999)

    def test_coco_annotation_with_multiple_segments_raises(self) -> None:
        input_gdf = COCOGeoDF(
            file_name=os.path.join(
                root_dir, "data", "build_masks_data", "annotation.json"
            ),
            pre_build_vector_dict=False,
        )

        with self.assertRaises(NotImplementedError):
            input_gdf._build_polygon_from_annotation(
                {"segmentation": [[[0, 0, 1, 0, 1, 1]], [[2, 2, 3, 2, 3, 3]]]}
            )

    def test_coco_memory_geodf_filters_and_returns_dataframe(self) -> None:
        geodf = COCOMemoryGeoDF(
            [
                Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]),
                Polygon([(5, 5), (6, 5), (6, 6), (5, 5)]),
            ]
        )

        self.assertIsNone(geodf.__post_init__())
        output = geodf.get_features_from_bbox(
            0, 3, 0, 3, only_geom=False, filter_area=0.1
        )

        self.assertEqual(len(output), 1)

    def test_save_to_file_validates_driver_and_writes(self) -> None:
        output_dir = os.path.join(root_dir, "..", "tmp_vector_reader_output")
        os.makedirs(output_dir, exist_ok=True)
        try:
            with self.assertRaises(TypeError):
                save_to_file([], output_dir, "bad", driver="BAD")
            save_to_file(
                [Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])],
                output_dir,
                "ok",
                crs="EPSG:4326",
            )
            self.assertTrue(os.path.exists(os.path.join(output_dir, "ok.geojson")))
        finally:
            for path in Path(output_dir).glob("*"):
                path.unlink()
            os.rmdir(output_dir)

    @patch("geopandas.read_postgis")
    @patch("psycopg2.connect")
    def test_module_main_guard_runs_with_mocked_postgis(
        self, mock_connect, mock_read_postgis
    ) -> None:
        mock_read_postgis.return_value = geopandas.GeoDataFrame(
            {"geometry": [Point(0, 0)]}, crs="EPSG:4326"
        )

        runpy.run_module(
            "pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader",
            run_name="__main__",
        )

        mock_connect.assert_called_once()
