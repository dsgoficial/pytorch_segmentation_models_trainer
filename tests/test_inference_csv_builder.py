# -*- coding: utf-8 -*-
import unittest
import os
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.tools.inference.inference_csv_builder import (
    InferenceCSVBuilder,
    build_inference_csv_from_config,
)


class TestInferenceCSVBuilder(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.tmp_dir, "images")
        self.masks_dir = os.path.join(self.tmp_dir, "masks")
        os.makedirs(self.images_dir)
        os.makedirs(self.masks_dir)

        # Create dummy rasters
        self.image_paths = []
        for i in range(3):
            path = os.path.join(self.images_dir, f"img_{i}.tif")
            self._create_dummy_raster(path)
            self.image_paths.append(path)

        # Create matching masks for 2 images
        self.mask_paths = []
        for i in range(2):
            path = os.path.join(self.masks_dir, f"img_{i}_mask.tif")
            self._create_dummy_raster(path)
            self.mask_paths.append(path)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _create_dummy_raster(self, path, width=10, height=10):
        data = np.zeros((1, height, width), dtype=np.uint8)
        transform = from_origin(0, 10, 1, 1)
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

    def test_init_validation(self):
        # Folder not found
        with self.assertRaises(FileNotFoundError):
            InferenceCSVBuilder(images_folder="/non/existent/path")

        # Mask folder not found
        with self.assertRaises(FileNotFoundError):
            InferenceCSVBuilder(
                images_folder=self.images_dir, masks_folder="/non/existent/masks"
            )

    def test_find_images(self):
        builder = InferenceCSVBuilder(images_folder=self.images_dir)
        images = builder.find_images()
        self.assertEqual(len(images), 3)
        self.assertTrue(all(isinstance(p, Path) for p in images))

    def test_find_images_recursive(self):
        sub_dir = os.path.join(self.images_dir, "sub")
        os.makedirs(sub_dir)
        path = os.path.join(sub_dir, "img_sub.tif")
        self._create_dummy_raster(path)

        builder = InferenceCSVBuilder(images_folder=self.images_dir, recursive=True)
        images = builder.find_images()
        self.assertEqual(len(images), 4)

    def test_find_corresponding_mask(self):
        builder = InferenceCSVBuilder(
            images_folder=self.images_dir,
            masks_folder=self.masks_dir,
            mask_suffix="_mask",
        )

        # Match with suffix
        img_0 = Path(self.image_paths[0])
        mask_0 = builder.find_corresponding_mask(img_0)
        self.assertIsNotNone(mask_0)
        self.assertEqual(mask_0.name, "img_0_mask.tif")

        # No match for img_2
        img_2 = Path(self.image_paths[2])
        mask_2 = builder.find_corresponding_mask(img_2)
        self.assertIsNone(mask_2)

        # Match exact name (no suffix)
        exact_mask_path = os.path.join(self.masks_dir, "img_2.tif")
        self._create_dummy_raster(exact_mask_path)
        mask_2_exact = builder.find_corresponding_mask(img_2)
        self.assertIsNotNone(mask_2_exact)
        self.assertEqual(mask_2_exact.name, "img_2.tif")

    def test_get_image_dimensions(self):
        builder = InferenceCSVBuilder(images_folder=self.images_dir)
        w, h = builder.get_image_dimensions(Path(self.image_paths[0]))
        self.assertEqual(w, 10)
        self.assertEqual(h, 10)

        # Invalid image
        invalid_path = os.path.join(self.images_dir, "invalid.tif")
        with open(invalid_path, "w") as f:
            f.write("not a tif")
        w, h = builder.get_image_dimensions(Path(invalid_path))
        self.assertEqual(w, 0)
        self.assertEqual(h, 0)

    def test_build_csv(self):
        builder = InferenceCSVBuilder(
            images_folder=self.images_dir,
            masks_folder=self.masks_dir,
            mask_suffix="_mask",
        )
        csv_path = os.path.join(self.tmp_dir, "test.csv")
        df = builder.build_csv(csv_path)

        self.assertEqual(len(df), 3)
        self.assertTrue(os.path.exists(csv_path))
        self.assertIn("image", df.columns)
        self.assertIn("mask", df.columns)
        self.assertIn("width", df.columns)

        # Check masks found
        self.assertEqual((df["mask"] != "").sum(), 2)

    def test_build_csv_no_images(self):
        empty_dir = os.path.join(self.tmp_dir, "empty")
        os.makedirs(empty_dir)
        builder = InferenceCSVBuilder(images_folder=empty_dir)
        with self.assertRaises(ValueError):
            builder.build_csv(os.path.join(self.tmp_dir, "fail.csv"))

    def test_load_or_build_csv(self):
        builder = InferenceCSVBuilder(images_folder=self.images_dir)
        csv_path = os.path.join(self.tmp_dir, "test_load.csv")

        # First time: builds
        df1 = builder.load_or_build_csv(csv_path)
        self.assertEqual(len(df1), 3)

        # Modify dir, but don't force rebuild: loads old
        os.remove(self.image_paths[0])
        df2 = builder.load_or_build_csv(csv_path)
        self.assertEqual(len(df2), 3)

        # Force rebuild: should have 2
        df3 = builder.load_or_build_csv(csv_path, force_rebuild=True)
        self.assertEqual(len(df3), 2)

    def test_build_from_config(self):
        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "mask_suffix": "_mask",
                "recursive": False,
            }
        )

        csv_path = build_inference_csv_from_config(config)
        self.assertTrue(os.path.exists(csv_path))
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 3)

    def test_relative_paths(self):
        builder = InferenceCSVBuilder(
            images_folder=self.images_dir, root_dir=self.tmp_dir
        )
        csv_path = os.path.join(self.tmp_dir, "rel.csv")
        df = builder.build_csv(csv_path)

        # Image path should be relative to tmp_dir
        self.assertEqual(df["image"].iloc[0], "images/img_0.tif")

    def test_find_corresponding_mask_strategy_3(self):
        # Strategy 3: Buscar por padrão (glob)
        # Create a mask that doesn't follow exact name or suffix
        mask_path = os.path.join(self.masks_dir, "img_0_some_other_thing.tif")
        self._create_dummy_raster(mask_path)

        builder = InferenceCSVBuilder(
            images_folder=self.images_dir,
            masks_folder=self.masks_dir,
            mask_suffix="_non_existent",  # Force it to fail strategy 1 and 2
        )

        img_0 = Path(self.image_paths[0])
        mask_0 = builder.find_corresponding_mask(img_0)
        self.assertIsNotNone(mask_0)
        self.assertEqual(mask_0.name, "img_0_some_other_thing.tif")

    def test_find_corresponding_mask_multiple_candidates(self):
        # Multiple candidates warning
        mask_1 = os.path.join(self.masks_dir, "img_0_a.tif")
        mask_2 = os.path.join(self.masks_dir, "img_0_b.tif")
        self._create_dummy_raster(mask_1)
        self._create_dummy_raster(mask_2)

        builder = InferenceCSVBuilder(
            images_folder=self.images_dir, masks_folder=self.masks_dir
        )

        img_0 = Path(self.image_paths[0])
        # Should log warning but return one of them
        mask = builder.find_corresponding_mask(img_0)
        self.assertIn(mask.name, ["img_0_mask.tif", "img_0_a.tif", "img_0_b.tif"])

    def test_build_from_config_default_output_path(self):
        config = OmegaConf.create(
            {"images_folder": self.images_dir, "recursive": False}
        )
        # Should generate a path with hash and timestamp
        csv_path = build_inference_csv_from_config(config)
        self.assertTrue(os.path.exists(csv_path))
        self.assertIn("inference_dataset_", csv_path)
        os.remove(csv_path)

    def test_relative_path_value_error(self):
        # Test the 'except ValueError' block in build_csv when relative_to fails
        builder = InferenceCSVBuilder(
            images_folder=self.images_dir, root_dir="/some/other/root"
        )
        csv_path = os.path.join(self.tmp_dir, "abs_fallback.csv")
        df = builder.build_csv(csv_path)
        # Should fallback to absolute path
        self.assertTrue(os.path.isabs(df["image"].iloc[0]))

    def test_mask_relative_path_value_error(self):
        # Test the 'except ValueError' block for mask paths
        builder = InferenceCSVBuilder(
            images_folder=self.images_dir,
            masks_folder=self.masks_dir,
            root_dir="/some/other/root",
        )
        csv_path = os.path.join(self.tmp_dir, "mask_abs_fallback.csv")
        df = builder.build_csv(csv_path)
        # Should fallback to absolute path for mask too
        self.assertTrue(os.path.isabs(df["mask"].iloc[0]))

    def test_build_from_config_force_rebuild(self):
        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "output_csv_path": os.path.join(self.tmp_dir, "force_rebuild.csv"),
                "force_rebuild": True,
            }
        )
        # Initial build
        build_inference_csv_from_config(config)

        # Modify dir
        os.remove(self.image_paths[0])

        # Build again with force_rebuild
        csv_path = build_inference_csv_from_config(config)
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 2)

    def test_build_csv_skips_invalid_images(self):
        # Create an invalid image (zero dimensions or unreadable)
        invalid_path = os.path.join(self.images_dir, "invalid_image.tif")
        with open(invalid_path, "w") as f:
            f.write("definitely not a tif")

        builder = InferenceCSVBuilder(images_folder=self.images_dir)
        csv_path = os.path.join(self.tmp_dir, "skips_invalid.csv")
        df = builder.build_csv(csv_path)

        # Should have 3 valid images and skip the invalid one
        self.assertEqual(len(df), 3)
        self.assertNotIn("invalid_image.tif", df["image"].values)


if __name__ == "__main__":
    unittest.main()
