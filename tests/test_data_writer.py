# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-07-15
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
from unittest.mock import MagicMock, patch
from pathlib import Path
import warnings

import geopandas
import numpy as np
import psycopg2
import pyproj
import rasterio
from affine import Affine
from geopandas.testing import geom_almost_equals, geom_equals
from numpy.testing import assert_array_equal
from parameterized import parameterized
from pytorch_segmentation_models_trainer.tools.data_handlers.data_writer import (
    AbstractDataWriter,
    BatchVectorFileDataWriter,
    ObjectDetectionDataWriter,
    RasterDataWriter,
    VectorDatabaseDataWriter,
    VectorFileDataWriter,
)
from pytorch_segmentation_models_trainer.utils.os_utils import (
    create_folder,
    remove_folder,
)
from rasterio.plot import reshape_as_raster
from shapely.geometry import Polygon

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")
from tests.utils import BasicTestCase

current_dir = os.path.dirname(__file__)
...


class Test_DataWriter(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.output_dir = self.make_temp_dir()

    def test_raster_data_writer(self) -> None:
        input_data = np.ones([256, 256, 1], dtype=np.uint8)
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "nodata": None,
            "width": 256,
            "height": 256,
            "count": 1,
            "crs": pyproj.CRS.from_epsg(31982),
            "transform": Affine(
                0.35, 0.0, 456828.3563822131, 0.0, -0.35, 6717252.490058491
            ),
            "tiled": False,
            "interleave": "band",
            "input_name": "output",
            "suffix": ".tif",
        }
        output_file_path = os.path.join(self.output_dir, "output.tif")
        data_writer = RasterDataWriter(output_file_path=output_file_path)
        data_writer.write_data(input_data=input_data, profile=profile)
        assert os.path.isfile(output_file_path)
        with rasterio.open(output_file_path, "r") as raster_ds:
            output_data = raster_ds.read()
        assert_array_equal(reshape_as_raster(input_data), output_data)

    def test_abstract_data_writer_cannot_be_instantiated(self) -> None:
        with self.assertRaises(TypeError):
            AbstractDataWriter()

    def test_abstract_data_writer_method_body_is_noop(self) -> None:
        class ConcreteWriter(AbstractDataWriter):
            def write_data(self, input_data: np.array) -> None:
                return super().write_data(input_data)

        self.assertIsNone(ConcreteWriter().write_data(np.array([1])))

    def test_raster_data_writer_uses_output_profile_and_jpeg_two_channel_padding(
        self,
    ) -> None:
        input_data = np.ones([4, 4, 2], dtype=np.uint8)
        profile = {
            "driver": "JPEG",
            "dtype": "uint8",
            "width": 4,
            "height": 4,
            "count": 2,
            "crs": None,
            "transform": Affine.identity(),
        }
        output_file_path = os.path.join(self.output_dir, "nested", "output.jpg")
        data_writer = RasterDataWriter(
            output_file_path=output_file_path,
            output_profile={
                "driver": "JPEG",
                "dtype": "uint8",
                "width": 4,
                "height": 4,
                "count": 2,
            },
        )

        data_writer.write_data(input_data=input_data, profile=profile)

        with rasterio.open(output_file_path, "r") as raster_ds:
            self.assertEqual(raster_ds.count, 3)

    def test_vector_file_data_writer(self) -> None:
        input_data = [Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])]
        output_file_path = os.path.join(self.output_dir, "output.geojson")
        data_writer = VectorFileDataWriter(
            output_file_folder=self.output_dir, output_file_name="output.geojson"
        )
        data_writer.write_data(input_data=input_data, profile={"crs": "EPSG:4326"})
        assert os.path.isfile(output_file_path)
        output_data = geopandas.read_file(filename=output_file_path)
        assert input_data[0].equals(output_data["geometry"][0])

    def test_vector_file_data_writer_missing_folder_raises(self) -> None:
        data_writer = VectorFileDataWriter()

        with self.assertRaises(ValueError):
            data_writer.get_output_file_path(None)

    def test_vector_file_data_writer_creates_extra_folder(self) -> None:
        data_writer = VectorFileDataWriter(
            output_file_folder=self.output_dir, output_file_name="output.geojson"
        )

        output_path = data_writer.get_output_file_path("nested")

        self.assertTrue(os.path.isdir(os.path.join(self.output_dir, "nested")))
        self.assertEqual(
            output_path, os.path.join(self.output_dir, "nested", "output.geojson")
        )

    def test_vector_file_data_writer_empty_data_returns_without_file(self) -> None:
        data_writer = VectorFileDataWriter(
            output_file_folder=self.output_dir, output_file_name="empty.geojson"
        )
        empty_gdf = MagicMock()
        empty_gdf.__len__.return_value = 0

        with patch(
            "pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.GeoDataFrame.from_features",
            return_value=empty_gdf,
        ):
            data_writer.write_data(input_data=[], profile={"crs": "EPSG:4326"})

        self.assertFalse(os.path.exists(os.path.join(self.output_dir, "empty.geojson")))

    def test_vector_file_data_writer_appends_existing_geojson(self) -> None:
        input_data = [Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])]
        output_file_path = os.path.join(self.output_dir, "append.geojson")
        data_writer = VectorFileDataWriter(
            output_file_folder=self.output_dir, output_file_name="append.geojson"
        )

        data_writer.write_data(input_data=input_data, profile={"crs": "EPSG:4326"})
        data_writer.write_data(input_data=input_data, profile={"crs": "EPSG:4326"})

        output_data = geopandas.read_file(filename=output_file_path)
        self.assertEqual(len(output_data), 2)

    def test_vector_file_data_writer_appends_non_geojson_driver(self) -> None:
        data_writer = VectorFileDataWriter(
            output_file_folder=self.output_dir,
            output_file_name="output.gpkg",
            driver="GPKG",
        )
        gdf = MagicMock()
        gdf.__len__.return_value = 1

        with (
            patch(
                "pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.GeoDataFrame.from_features",
                return_value=gdf,
            ),
            patch(
                "pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.os.path.isfile",
                return_value=True,
            ),
        ):
            data_writer.write_data(
                input_data=[Polygon([[0, 0], [1, 0], [1, 1], [0, 0]])],
                profile={"crs": "EPSG:4326"},
            )

        gdf.to_file.assert_called_once()
        self.assertEqual(gdf.to_file.call_args.kwargs["mode"], "a")

    def test_batch_vector_file_data_writer(self) -> None:
        input_data = [Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])]
        output_file_path = os.path.join(self.output_dir, "output.geojson")
        data_writer = BatchVectorFileDataWriter(
            output_file_folder=self.output_dir, output_file_name="output.geojson"
        )
        for i in range(4):
            data_writer.write_data(input_data=input_data, profile={"crs": "EPSG:4326"})
            current_output_file_path = os.path.join(
                self.output_dir, f"output_{i:08}.geojson"
            )
            assert os.path.isfile(current_output_file_path)
            output_data = geopandas.read_file(filename=current_output_file_path)
            assert input_data[0].equals(output_data["geometry"][0])

    def test_batch_vector_file_data_writer_empty_data_increments_index(self) -> None:
        data_writer = BatchVectorFileDataWriter(
            output_file_folder=self.output_dir, output_file_name="empty.geojson"
        )
        empty_gdf = MagicMock()
        empty_gdf.__len__.return_value = 0

        with patch(
            "pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.GeoDataFrame.from_features",
            return_value=empty_gdf,
        ):
            data_writer.write_data(input_data=[], profile={"crs": "EPSG:4326"})

        self.assertEqual(data_writer.current_index, 1)
        self.assertFalse(
            os.path.exists(os.path.join(self.output_dir, "empty_00000000.geojson"))
        )

    def test_batch_vector_file_data_writer_appends_existing_geojson(self) -> None:
        input_data = [Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])]
        data_writer = BatchVectorFileDataWriter(
            output_file_folder=self.output_dir, output_file_name="append.geojson"
        )
        data_writer.write_data(input_data=input_data, profile={"crs": "EPSG:4326"})
        data_writer.current_index = 0

        data_writer.write_data(input_data=input_data, profile={"crs": "EPSG:4326"})

        output_data = geopandas.read_file(
            filename=os.path.join(self.output_dir, "append_00000000.geojson")
        )
        self.assertEqual(len(output_data), 2)

    def test_batch_vector_file_data_writer_appends_non_geojson_driver(self) -> None:
        data_writer = BatchVectorFileDataWriter(
            output_file_folder=self.output_dir,
            output_file_name="output.gpkg",
            driver="GPKG",
        )
        gdf = MagicMock()
        gdf.__len__.return_value = 1

        with (
            patch(
                "pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.GeoDataFrame.from_features",
                return_value=gdf,
            ),
            patch(
                "pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.os.path.isfile",
                return_value=True,
            ),
        ):
            data_writer.write_data(
                input_data=[Polygon([[0, 0], [1, 0], [1, 1], [0, 0]])],
                profile={"crs": "EPSG:4326"},
            )

        gdf.to_file.assert_called_once()
        self.assertEqual(gdf.to_file.call_args.kwargs["mode"], "a")

    @patch(
        "pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.create_engine"
    )
    @patch("geopandas.GeoDataFrame.to_postgis")
    def test_vector_database_data_writer(
        self, mock_to_postgis, mock_create_engine
    ) -> None:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        input_data = [Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])]
        data_writer = VectorDatabaseDataWriter(
            user="postgres",
            password="postgres",
            database="test_db",
            table_name="test",
            if_exists="replace",
        )
        data_writer.write_data(input_data=input_data, profile={"crs": "EPSG:4326"})

        # Verify that to_postgis was called with correct parameters
        mock_to_postgis.assert_called_once()
        args, kwargs = mock_to_postgis.call_args
        self.assertEqual(args[0], "test")  # table_name
        self.assertEqual(args[1], mock_engine)  # engine
        self.assertEqual(kwargs["if_exists"], "replace")

    @patch("geopandas.GeoDataFrame.to_postgis")
    def test_vector_database_data_writer_empty_data_returns(
        self, mock_to_postgis
    ) -> None:
        data_writer = VectorDatabaseDataWriter(
            user="postgres",
            password="postgres",
            database="test_db",
            table_name="test",
        )
        empty_gdf = MagicMock()
        empty_gdf.__len__.return_value = 0

        with patch(
            "pytorch_segmentation_models_trainer.tools.data_handlers.data_writer.GeoDataFrame.from_features",
            return_value=empty_gdf,
        ):
            data_writer.write_data(input_data=[], profile={"crs": "EPSG:4326"})

        mock_to_postgis.assert_not_called()

    def test_object_detection_data_writer_writes_json(self) -> None:
        output_file_path = os.path.join(self.output_dir, "detections.json")
        data_writer = ObjectDetectionDataWriter(output_file_path=output_file_path)

        data_writer.write_data([{"bbox": [0, 1, 2, 3], "score": 0.9}])

        self.assertTrue(os.path.exists(output_file_path))
