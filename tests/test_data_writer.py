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
    BatchVectorFileDataWriter,
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


class Test_DataWriter(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self.output_dir = create_folder(os.path.join(root_dir, "test_output"))

    def tearDown(self):
        remove_folder(self.output_dir)

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
        }
        output_file_path = os.path.join(self.output_dir, "output.tif")
        data_writer = RasterDataWriter(output_file_path=output_file_path)
        data_writer.write_data(input_data=input_data, profile=profile)
        assert os.path.isfile(output_file_path)
        with rasterio.open(output_file_path, "r") as raster_ds:
            output_data = raster_ds.read()
        assert_array_equal(reshape_as_raster(input_data), output_data)

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

    def test_vector_database_data_writer(self) -> None:
        input_data = [Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])]
        data_writer = VectorDatabaseDataWriter(
            user="postgres",
            password="postgres",
            database="test_db",
            table_name="test",
            if_exists="replace",
        )
        data_writer.write_data(input_data=input_data, profile={"crs": "EPSG:4326"})
        con = psycopg2.connect(
            host="localhost",
            port=5432,
            database="test_db",
            user="postgres",
            password="postgres",
        )
        output_gdf = geopandas.read_postgis(
            sql="select * from test", con=con, geom_col="geom"
        )
        assert input_data[0].equals(output_gdf["geom"][0])
