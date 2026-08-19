# -*- coding: utf-8 -*-
"""
Unit tests for export_inference.py
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import os
import shutil

from pytorch_segmentation_models_trainer.tools.inference.export_inference import (
    RasterExportInferenceStrategy,
    MultipleRasterExportInferenceStrategy,
    ConfidenceExportStrategy,
    VectorFileExportInferenceStrategy,
    VectorDatabaseExportInferenceStrategy,
    ObjectDetectionExportInferenceStrategy,
)
from pytorch_segmentation_models_trainer.tools.data_handlers.data_writer import (
    RasterDataWriter,
    VectorFileDataWriter,
    VectorDatabaseDataWriter,
    ObjectDetectionDataWriter,
)


class TestRasterExportInferenceStrategy(unittest.TestCase):
    def setUp(self):
        self.output_dir = "temp_export_test_dir"
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file_path = os.path.join(self.output_dir, "test_output.tif")
        self.profile = {
            "driver": "GTiff",
            "height": 100,
            "width": 100,
            "count": 1,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 100.0],
            "input_name": "test_image",
            "suffix": ".tif",
        }
        self.inference_data = np.random.randint(0, 2, (1, 100, 100), dtype=np.uint8)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    @patch(
        "pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterDataWriter"
    )
    def test_raster_export_strategy_dict_input(self, MockRasterDataWriter):
        mock_writer_instance = MagicMock(spec=RasterDataWriter)
        MockRasterDataWriter.return_value = mock_writer_instance

        strategy = RasterExportInferenceStrategy(self.output_file_path)
        strategy.save_inference(
            {"seg": self.inference_data, "other": "data"}, self.profile
        )

        MockRasterDataWriter.assert_called_once_with(
            output_file_path=self.output_file_path, output_profile=None
        )
        mock_writer_instance.write_data.assert_called_once_with(
            self.inference_data, self.profile
        )

    @patch(
        "pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterDataWriter"
    )
    def test_raster_export_strategy_ndarray_input(self, MockRasterDataWriter):
        mock_writer_instance = MagicMock(spec=RasterDataWriter)
        MockRasterDataWriter.return_value = mock_writer_instance

        strategy = RasterExportInferenceStrategy(self.output_file_path)
        strategy.save_inference(self.inference_data, self.profile)

        mock_writer_instance.write_data.assert_called_once_with(
            self.inference_data, self.profile
        )

    @patch(
        "pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterDataWriter"
    )
    def test_raster_export_strategy_with_output_profile(self, MockRasterDataWriter):
        mock_writer_instance = MagicMock(spec=RasterDataWriter)
        MockRasterDataWriter.return_value = mock_writer_instance

        custom_output_profile = {"driver": "GTiff", "compression": "LZW"}
        strategy = RasterExportInferenceStrategy(
            self.output_file_path, output_profile=custom_output_profile
        )
        strategy.save_inference(self.inference_data, self.profile)

        MockRasterDataWriter.assert_called_once_with(
            output_file_path=self.output_file_path, output_profile=custom_output_profile
        )
        mock_writer_instance.write_data.assert_called_once_with(
            self.inference_data, self.profile
        )


class TestMultipleRasterExportInferenceStrategy(unittest.TestCase):
    def setUp(self):
        self.output_folder = "temp_multi_raster_export_test_dir"
        os.makedirs(self.output_folder, exist_ok=True)
        self.output_basename = "inference_result"
        self.profile = {
            "driver": "GTiff",
            "height": 100,
            "width": 100,
            "count": 1,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 100.0],
            "input_name": "test_image_id",
            "suffix": ".tif",
        }
        self.inference_dict = {
            "seg": np.random.randint(0, 2, (1, 100, 100), dtype=np.uint8),
            "border": np.random.randint(0, 2, (1, 100, 100), dtype=np.uint8),
        }

    def tearDown(self):
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

    @patch(
        "pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterDataWriter"
    )
    def test_multiple_raster_export_strategy(self, MockRasterDataWriter):
        mock_writer_instance = MagicMock(spec=RasterDataWriter)
        MockRasterDataWriter.return_value = mock_writer_instance

        strategy = MultipleRasterExportInferenceStrategy(
            self.output_folder, self.output_basename
        )
        strategy.save_inference(self.inference_dict, self.profile)

        # Check that RasterDataWriter was instantiated for each item in inference_dict
        self.assertEqual(MockRasterDataWriter.call_count, 2)

        # Check calls - since we use numpy arrays, we can't use assert_any_call directly easily
        # We check the calls manually
        output_seg_path = os.path.join(
            self.output_folder, f"seg_test_image_id_{self.output_basename}"
        )
        output_border_path = os.path.join(
            self.output_folder, f"border_test_image_id_{self.output_basename}"
        )

        found_seg = False
        found_border = False

        for call_args in mock_writer_instance.write_data.call_args_list:
            args, kwargs = call_args
            if np.array_equal(args[0], self.inference_dict["seg"]):
                found_seg = True
            if np.array_equal(args[0], self.inference_dict["border"]):
                found_border = True

        self.assertTrue(found_seg)
        self.assertTrue(found_border)


class TestConfidenceExportStrategy(unittest.TestCase):
    def setUp(self):
        self.output_file_path = "temp_confidence_export_test_dir"
        self.profile = {
            "driver": "GTiff",
            "height": 100,
            "width": 100,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 100.0],
            "input_name": "test_image",
            "suffix": ".tif",
        }
        self.data = np.random.rand(100, 100, 1).astype(np.float32)

    def tearDown(self):
        if os.path.exists(self.output_file_path):
            shutil.rmtree(self.output_file_path)

    @patch(
        "pytorch_segmentation_models_trainer.tools.inference.export_inference.RasterDataWriter"
    )
    def test_save_confidence(self, MockRasterDataWriter):
        mock_writer_instance = MagicMock(spec=RasterDataWriter)
        MockRasterDataWriter.return_value = mock_writer_instance

        strategy = ConfidenceExportStrategy(self.output_file_path)
        metric_name = "max_prob"
        strategy.save_confidence(metric_name, self.data, self.profile)

        expected_subfolder = os.path.join(
            self.output_file_path, f"confidence_{metric_name}"
        )
        expected_output_path = os.path.join(
            expected_subfolder, f"{self.profile['input_name']}{self.profile['suffix']}"
        )

        MockRasterDataWriter.assert_called_once_with(
            output_file_path=expected_output_path
        )
        mock_writer_instance.write_data.assert_called_once_with(self.data, self.profile)


class TestVectorFileExportInferenceStrategy(unittest.TestCase):
    def setUp(self):
        self.output_file_path = "temp_vector_export.geojson"
        self.driver = "GeoJSON"
        self.profile = {"input_name": "test_vector_image", "suffix": ".geojson"}
        self.inference_data = [
            {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                },
                "properties": {"class": 1},
            }
        ]

    def tearDown(self):
        if os.path.exists(self.output_file_path):
            os.remove(self.output_file_path)

    @patch(
        "pytorch_segmentation_models_trainer.tools.inference.export_inference.VectorFileDataWriter"
    )
    def test_vector_file_export_strategy(self, MockVectorFileDataWriter):
        mock_writer_instance = MagicMock(spec=VectorFileDataWriter)
        MockVectorFileDataWriter.return_value = mock_writer_instance

        strategy = VectorFileExportInferenceStrategy(self.output_file_path, self.driver)
        # VectorFileDataWriter needs output_file_folder and output_file_name.
        # ExportInferenceStrategy passes output_file_path to VectorFileDataWriter as output_file_name.
        # But VectorFileDataWriter.write_data calls get_output_file_path which checks output_file_folder.

        strategy.save_inference(self.inference_data, self.profile)

        MockVectorFileDataWriter.assert_called_once_with(
            output_file_name=self.output_file_path, driver=self.driver
        )
        mock_writer_instance.write_data.assert_called_once_with(
            self.inference_data, self.profile
        )


class TestVectorDatabaseExportInferenceStrategy(unittest.TestCase):
    @patch(
        "pytorch_segmentation_models_trainer.tools.inference.export_inference.VectorDatabaseDataWriter"
    )
    def test_vector_database_export_strategy(self, MockVectorDatabaseDataWriter):
        mock_writer_instance = MagicMock(spec=VectorDatabaseDataWriter)
        MockVectorDatabaseDataWriter.return_value = mock_writer_instance

        user = "test_user"
        database = "test_db"
        password = "test_password"
        sql = "test_sql"
        host = "localhost"
        port = 5432
        table_name = "test_table"
        geometry_column = "test_geom"

        strategy = VectorDatabaseExportInferenceStrategy(
            user, database, password, sql, host, port, table_name, geometry_column
        )
        inference_data = [
            {
                "geometry": {"type": "Point", "coordinates": [10, 20]},
                "properties": {"id": 1},
            }
        ]
        profile = {"input_name": "test_data", "crs": "EPSG:4326"}

        strategy.save_inference(inference_data, profile)

        MockVectorDatabaseDataWriter.assert_called_once_with(
            user=user,
            password=password,
            database=database,
            sql=sql,
            host=host,
            port=port,
            table_name=table_name,
            geometry_column=geometry_column,
        )
        mock_writer_instance.write_data.assert_called_once_with(inference_data, profile)


class TestObjectDetectionExportInferenceStrategy(unittest.TestCase):
    def setUp(self):
        self.output_file_path = "temp_object_detection_export.json"
        self.inference_data = [{"box": [10, 10, 20, 20], "score": 0.9, "category": 1}]
        self.profile = {"input_name": "test_detection", "suffix": ".json"}

    def tearDown(self):
        if os.path.exists(self.output_file_path):
            os.remove(self.output_file_path)

    @patch(
        "pytorch_segmentation_models_trainer.tools.inference.export_inference.ObjectDetectionDataWriter"
    )
    def test_object_detection_export_inference_strategy(
        self, MockObjectDetectionDataWriter
    ):
        mock_writer_instance = MagicMock(spec=ObjectDetectionDataWriter)
        MockObjectDetectionDataWriter.return_value = mock_writer_instance

        strategy = ObjectDetectionExportInferenceStrategy(self.output_file_path)
        strategy.save_inference(self.inference_data, self.profile)

        MockObjectDetectionDataWriter.assert_called_once_with(
            output_file_path=self.output_file_path
        )
        mock_writer_instance.write_data.assert_called_once_with(
            self.inference_data, self.profile
        )


if __name__ == "__main__":
    unittest.main()
