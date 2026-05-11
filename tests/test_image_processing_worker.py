# -*- coding: utf-8 -*-
import unittest
import os
import shutil
import tempfile
import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from pytorch_segmentation_models_trainer.tools.evaluation.image_processing_worker import (
    process_single_image_worker,
    read_aligned_rasters_worker,
    get_spatial_overlap_worker,
)


class TestImageProcessingWorker(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.pred_path = os.path.join(self.tmp_dir, "pred.tif")
        self.gt_path = os.path.join(self.tmp_dir, "gt.tif")

        # Create overlapping rasters
        self._create_dummy_raster(self.pred_path, origin=(0, 10))
        self._create_dummy_raster(self.gt_path, origin=(0, 10))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _create_dummy_raster(self, path, width=10, height=10, origin=(0, 10)):
        data = np.zeros((1, height, width), dtype=np.uint8)
        transform = from_origin(origin[0], origin[1], 1, 1)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

    def test_process_single_image_worker_dict_task(self):
        task = {
            "pred_path": self.pred_path,
            "gt_path": self.gt_path,
            "image_name": "test_img",
            "index": 1,
        }
        result = process_single_image_worker(task)
        self.assertIsNotNone(result)
        self.assertEqual(result["image_name"], "test_img")
        self.assertEqual(result["index"], 1)
        self.assertIn("pred_flat", result)

    def test_process_single_image_worker_kwargs(self):
        result = process_single_image_worker(
            image_id="test_id", pred_path=self.pred_path, gt_path=self.gt_path
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["image_id"], "test_id")

    def test_process_single_image_worker_missing_files(self):
        result = process_single_image_worker(
            pred_path="/non/existent.tif", gt_path="/non/existent.tif"
        )
        self.assertIsNone(result)

    def test_process_single_image_worker_exception(self):
        # Trigger exception by passing None to read_aligned
        result = process_single_image_worker(pred_path=None, gt_path=None)
        self.assertIsNone(result)

    def test_get_spatial_overlap_no_overlap(self):
        pred_no_overlap = os.path.join(self.tmp_dir, "pred_no.tif")
        # Origin far away
        self._create_dummy_raster(pred_no_overlap, origin=(100, 100))

        res = get_spatial_overlap_worker(pred_no_overlap, self.gt_path)
        self.assertIsNone(res)

    def test_read_aligned_rasters_mismatch_crop(self):
        # Create different sized rasters that overlap
        pred_big = os.path.join(self.tmp_dir, "pred_big.tif")
        self._create_dummy_raster(pred_big, width=20, height=20)

        # This will trigger the mismatch warning and crop
        p_mask, g_mask = read_aligned_rasters_worker(
            pred_big, self.gt_path, num_classes=2
        )
        self.assertIsNotNone(p_mask)
        self.assertEqual(p_mask.shape, (10, 10))

    def test_process_single_image_worker_precalculated_metrics(self):
        # Test the fallback for pre-calculated metrics in kwargs
        result = process_single_image_worker(image_id="test_pre", iou=0.8, accuracy=0.9)
        self.assertIsNotNone(result)
        self.assertEqual(result["iou"], 0.8)
        self.assertEqual(result["image_name"], "test_pre")

    def test_process_single_image_worker_relative_to_temp_dir(self):
        # Create temp_test_dir
        os.makedirs("temp_test_dir", exist_ok=True)
        try:
            p_rel = "p_rel.tif"
            g_rel = "g_rel.tif"
            self._create_dummy_raster(os.path.join("temp_test_dir", p_rel))
            self._create_dummy_raster(os.path.join("temp_test_dir", g_rel))

            result = process_single_image_worker(
                image_id="test_rel", pred_path=p_rel, gt_path=g_rel
            )
            self.assertIsNotNone(result)
        finally:
            shutil.rmtree("temp_test_dir")

    def test_get_spatial_overlap_mismatch_windows(self):
        # Create rasters with slightly different resolutions or alignments to trigger window mismatch
        p_path = os.path.join(self.tmp_dir, "p_mis.tif")
        g_path = os.path.join(self.tmp_dir, "g_mis.tif")

        # Pred: 1m resolution
        self._create_dummy_raster(p_path, width=10, height=10)

        # GT: 1.1m resolution (slightly different)
        data = np.zeros((1, 10, 10), dtype=np.uint8)
        transform = from_origin(0, 10, 1.1, 1.1)
        with rasterio.open(
            g_path,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)

        res = get_spatial_overlap_worker(p_path, g_path)
        self.assertIsNotNone(res)
        # Should have matched_shape and adjusted windows

    def test_read_aligned_rasters_error(self):
        # Trigger exception in read_aligned_rasters_worker
        res = read_aligned_rasters_worker(None, None, 2)
        self.assertEqual(res, (None, None))

    def test_get_spatial_overlap_error(self):
        # Trigger exception in get_spatial_overlap_worker
        res = get_spatial_overlap_worker(None, None)
        self.assertIsNone(res)

    def test_process_single_image_worker_exception_logged(self):
        # Trigger the logger.error in process_single_image_worker
        # by passing something that causes an error inside the try block
        # e.g. a task dict with missing keys that are used
        result = process_single_image_worker(
            {"pred_path": 123}
        )  # 123 is not a valid path string/type for os.path.exists
        self.assertIsNone(result)
