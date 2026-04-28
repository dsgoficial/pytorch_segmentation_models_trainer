# -*- coding: utf-8 -*-
"""Tests for grid_cache and class_presence_cache auto-save/load."""
import csv
import json
import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
import rasterio
from rasterio.transform import from_bounds

import pytorch_segmentation_models_trainer  # noqa: F401 — DLL fix

from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    RandomCropSegmentationDataset,
)


def _create_geotiff(path, data, count=None, dtype="uint8"):
    """Write a numpy array as a GeoTIFF with dummy georef."""
    if data.ndim == 2:
        h, w = data.shape
        bands = 1
        data = data[np.newaxis, ...]
    else:
        bands, h, w = data.shape
    if count is not None:
        bands = count
    transform = from_bounds(0, 0, w, h, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=bands, dtype=dtype, transform=transform, crs="EPSG:3857",
    ) as dst:
        for b in range(bands):
            dst.write(data[b], b + 1)


class TestGridCache(unittest.TestCase):
    """Test grid_cache save/load cycle."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self.tmpdir = tempfile.mkdtemp(prefix="test_grid_cache_")
        img_dir = os.path.join(self.tmpdir, "images")
        msk_dir = os.path.join(self.tmpdir, "masks")
        os.makedirs(img_dir)
        os.makedirs(msk_dir)

        h = w = 64
        self.image_paths = []
        self.mask_paths = []
        for i in range(3):
            img = np.full((3, h, w), 100 + i * 30, dtype=np.uint8)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[h // 2:, :w // 2] = 1
            mask[h // 2:, w // 2:] = 2
            img_path = os.path.join(img_dir, f"img_{i}.tif")
            msk_path = os.path.join(msk_dir, f"msk_{i}.tif")
            _create_geotiff(img_path, img)
            _create_geotiff(msk_path, mask)
            self.image_paths.append(img_path)
            self.mask_paths.append(msk_path)

        self.csv_path = os.path.join(self.tmpdir, "dataset.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            for ip, mp in zip(self.image_paths, self.mask_paths):
                writer.writerow([ip, mp])

        self.cache_path = os.path.join(self.tmpdir, "grid_cache.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_cache_created_on_first_run(self):
        """Grid cache JSON should be created after first dataset init."""
        self.assertFalse(os.path.exists(self.cache_path))
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=[32, 32], n_classes=3,
            grid_mode=True, overlap_x=0.0, overlap_y=0.0,
            min_valid_ratio=0.0,
            grid_cache=self.cache_path,
        )
        self.assertTrue(os.path.exists(self.cache_path))
        self.assertGreater(len(ds), 0)

    def test_cache_loaded_on_second_run(self):
        """Second init should load from cache without reading masks."""
        ds1 = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=[32, 32], n_classes=3,
            grid_mode=True, overlap_x=0.0, overlap_y=0.0,
            min_valid_ratio=0.0,
            grid_cache=self.cache_path,
        )
        n1 = len(ds1)
        positions1 = ds1._grid_positions.copy()

        ds2 = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=[32, 32], n_classes=3,
            grid_mode=True, overlap_x=0.0, overlap_y=0.0,
            min_valid_ratio=0.0,
            grid_cache=self.cache_path,
        )
        n2 = len(ds2)
        self.assertEqual(n1, n2)
        np.testing.assert_array_equal(positions1, ds2._grid_positions)

    def test_cache_invalidated_on_crop_size_change(self):
        """If crop_size changes, cache should be rebuilt."""
        ds1 = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=[32, 32], n_classes=3,
            grid_mode=True, overlap_x=0.0, overlap_y=0.0,
            min_valid_ratio=0.0,
            grid_cache=self.cache_path,
        )
        n_32 = len(ds1)

        # Change crop_size — should rebuild
        ds2 = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=[16, 16], n_classes=3,
            grid_mode=True, overlap_x=0.0, overlap_y=0.0,
            min_valid_ratio=0.0,
            grid_cache=self.cache_path,
        )
        n_16 = len(ds2)
        self.assertGreater(n_16, n_32)

    def test_cache_json_structure(self):
        """Verify the JSON has expected keys."""
        RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=[32, 32], n_classes=3,
            grid_mode=True, overlap_x=0.0, overlap_y=0.0,
            min_valid_ratio=0.0,
            grid_cache=self.cache_path,
        )
        with open(self.cache_path) as f:
            cache = json.load(f)

        self.assertIn("grid_positions", cache)
        self.assertIn("config", cache)
        self.assertEqual(cache["config"]["crop_size"], [32, 32])
        self.assertGreater(len(cache["grid_positions"]), 0)
        self.assertEqual(len(cache["grid_positions"][0]), 3)

    def test_cache_with_class_balanced(self):
        """Cache should include class_positions when class_balanced_sampling=True."""
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=[32, 32], n_classes=3,
            grid_mode=True, overlap_x=0.0, overlap_y=0.0,
            min_valid_ratio=0.0,
            grid_cache=self.cache_path,
            class_balanced_sampling=True,
        )
        with open(self.cache_path) as f:
            cache = json.load(f)

        self.assertIn("class_positions", cache)
        has_positions = any(len(v) > 0 for v in cache["class_positions"].values())
        self.assertTrue(has_positions)

        ds2 = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=[32, 32], n_classes=3,
            grid_mode=True, overlap_x=0.0, overlap_y=0.0,
            min_valid_ratio=0.0,
            grid_cache=self.cache_path,
            class_balanced_sampling=True,
        )
        self.assertTrue(hasattr(ds2, "_class_positions"))
        self.assertTrue(len(ds2._class_positions) > 0)


class TestClassPresenceCacheAutoSave(unittest.TestCase):
    """Test auto-save of class_presence_cache."""

    def setUp(self):
        warnings.simplefilter("ignore")
        self.tmpdir = tempfile.mkdtemp(prefix="test_cp_cache_")
        img_dir = os.path.join(self.tmpdir, "images")
        msk_dir = os.path.join(self.tmpdir, "masks")
        os.makedirs(img_dir)
        os.makedirs(msk_dir)

        h = w = 64
        self.image_paths = []
        self.mask_paths = []
        for i in range(3):
            img = np.full((3, h, w), 128, dtype=np.uint8)
            mask = np.full((h, w), i, dtype=np.uint8)  # each image = 1 class
            img_path = os.path.join(img_dir, f"img_{i}.tif")
            msk_path = os.path.join(msk_dir, f"msk_{i}.tif")
            _create_geotiff(img_path, img)
            _create_geotiff(msk_path, mask)
            self.image_paths.append(img_path)
            self.mask_paths.append(msk_path)

        self.csv_path = os.path.join(self.tmpdir, "dataset.csv")
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "mask"])
            for ip, mp in zip(self.image_paths, self.mask_paths):
                writer.writerow([ip, mp])

        self.cache_path = os.path.join(self.tmpdir, "class_presence.json")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_auto_save_on_first_build(self):
        """class_presence_cache should be auto-saved when built from scratch."""
        self.assertFalse(os.path.exists(self.cache_path))
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=[32, 32], n_classes=3,
            samples_per_epoch=10,
            min_valid_ratio=0.0,
            class_presence_cache=self.cache_path,
            cutmix_rare_classes=[0, 1],
        )
        self.assertTrue(os.path.exists(self.cache_path))

    def test_auto_save_json_structure(self):
        """Auto-saved JSON should map filenames to class lists (plus optional _config key)."""
        ds = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=[32, 32], n_classes=3,
            samples_per_epoch=10,
            min_valid_ratio=0.0,
            class_presence_cache=self.cache_path,
            cutmix_rare_classes=[0, 1],
        )
        with open(self.cache_path) as f:
            cache = json.load(f)

        data_entries = {k: v for k, v in cache.items() if k != "_config"}
        self.assertEqual(len(data_entries), 3)
        for name, classes in data_entries.items():
            self.assertIsInstance(classes, list)
            self.assertTrue(all(isinstance(c, int) for c in classes))

    def test_load_matches_build(self):
        """Loading from cache should produce same class_to_images as building."""
        ds1 = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=[32, 32], n_classes=3,
            samples_per_epoch=10,
            min_valid_ratio=0.0,
            class_presence_cache=self.cache_path,
            cutmix_rare_classes=[0, 1],
        )
        c2i_built = {c: set(imgs.tolist()) for c, imgs in ds1._class_to_images.items()}

        ds2 = RandomCropSegmentationDataset(
            input_csv_path=self.csv_path,
            crop_size=[32, 32], n_classes=3,
            samples_per_epoch=10,
            min_valid_ratio=0.0,
            class_presence_cache=self.cache_path,
            cutmix_rare_classes=[0, 1],
        )
        c2i_loaded = {c: set(imgs.tolist()) for c, imgs in ds2._class_to_images.items()}

        self.assertEqual(c2i_built, c2i_loaded)


if __name__ == "__main__":
    unittest.main()
