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

from pytorch_segmentation_models_trainer.tools.evaluation.csv_builder import (
    DatasetCSVBuilder,
)


class TestDatasetCSVBuilder(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.images_dir = os.path.join(self.tmp_dir, "images")
        self.masks_dir = os.path.join(self.tmp_dir, "masks")
        os.makedirs(self.images_dir)
        os.makedirs(self.masks_dir)

        # Create dummy rasters
        self._create_dummy_raster(os.path.join(self.images_dir, "img_1.tif"))
        self._create_dummy_raster(os.path.join(self.images_dir, "img_2.tif"))
        self._create_dummy_raster(os.path.join(self.masks_dir, "img_1.tif"))
        self._create_dummy_raster(os.path.join(self.masks_dir, "img_2.tif"))

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
        config = OmegaConf.create(
            {"images_folder": self.images_dir, "masks_folder": self.masks_dir}
        )
        builder = DatasetCSVBuilder(config)
        self.assertEqual(builder.images_folder, Path(self.images_dir))

        # Folder not found
        config_fail = OmegaConf.create(
            {"images_folder": "/non/existent/path", "masks_folder": self.masks_dir}
        )
        with self.assertRaises(FileNotFoundError):
            DatasetCSVBuilder(config_fail)

        config_missing_masks = OmegaConf.create(
            {"images_folder": self.images_dir, "masks_folder": "/non/existent/masks"}
        )
        with self.assertRaises(FileNotFoundError):
            DatasetCSVBuilder(config_missing_masks)

    def test_build_csv_same_basename(self):
        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "matching_strategy": "same_basename",
            }
        )
        builder = DatasetCSVBuilder(config)
        csv_path = os.path.join(self.tmp_dir, "out.csv")
        df = builder.build_csv(csv_path)

        self.assertEqual(len(df), 2)
        self.assertTrue(os.path.exists(csv_path))
        self.assertEqual(df["width"].iloc[0], 10)

    def test_match_prefix_suffix(self):
        # Setup specific files
        os.remove(os.path.join(self.images_dir, "img_1.tif"))
        os.remove(os.path.join(self.masks_dir, "img_1.tif"))
        self._create_dummy_raster(
            os.path.join(self.images_dir, "image_abc_001_raw.tif")
        )
        self._create_dummy_raster(
            os.path.join(self.masks_dir, "mask_abc_001_label.tif")
        )

        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "matching_strategy": "prefix_suffix",
                "image_prefix": "image_",
                "image_suffix": "_raw",
                "mask_prefix": "mask_",
                "mask_suffix": "_label",
            }
        )
        builder = DatasetCSVBuilder(config)
        image_path = Path(self.images_dir) / "image_abc_001_raw.tif"
        mask_path = builder.match_image_to_mask(image_path)

        self.assertIsNotNone(mask_path)
        self.assertEqual(mask_path.name, "mask_abc_001_label.tif")

    def test_match_custom_regex(self):
        self._create_dummy_raster(os.path.join(self.images_dir, "img_xyz_789.tif"))
        self._create_dummy_raster(os.path.join(self.masks_dir, "mask_abc_789.tif"))

        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "matching_strategy": "custom_regex",
                "regex_pattern": r"(?P<id>\d+)",
            }
        )
        builder = DatasetCSVBuilder(config)
        image_path = Path(self.images_dir) / "img_xyz_789.tif"
        mask_path = builder.match_image_to_mask(image_path)

        self.assertIsNotNone(mask_path)
        self.assertEqual(mask_path.name, "mask_abc_789.tif")

    def test_match_custom_regex_missing_pattern(self):
        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "matching_strategy": "custom_regex",
            }
        )
        builder = DatasetCSVBuilder(config)
        with self.assertRaises(ValueError):
            builder.match_image_to_mask(Path("any.tif"))

    def test_recursive_search(self):
        sub_dir = os.path.join(self.images_dir, "sub")
        os.makedirs(sub_dir)
        self._create_dummy_raster(os.path.join(sub_dir, "img_sub.tif"))
        self._create_dummy_raster(os.path.join(self.masks_dir, "img_sub.tif"))

        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "recursive": True,
            }
        )
        builder = DatasetCSVBuilder(config)
        image_files = builder._find_files(builder.images_folder, "*.tif")
        self.assertEqual(len(image_files), 3)

    def test_validate_dataset_errors(self):
        config = OmegaConf.create(
            {"images_folder": self.images_dir, "masks_folder": self.masks_dir}
        )
        builder = DatasetCSVBuilder(config)

        # Missing image file
        df_missing_img = pd.DataFrame(
            {
                "image": ["/non/existent.tif"],
                "mask": [os.path.join(self.masks_dir, "img_1.tif")],
                "width": [10],
                "height": [10],
            }
        )
        with self.assertRaises(ValueError):
            builder.validate_dataset(df_missing_img)

        # Missing mask file
        df_missing_mask = pd.DataFrame(
            {
                "image": [os.path.join(self.images_dir, "img_1.tif")],
                "mask": ["/non/existent_mask.tif"],
                "width": [10],
                "height": [10],
            }
        )
        with self.assertRaises(ValueError):
            builder.validate_dataset(df_missing_mask)

    def test_invalid_matching_strategy(self):
        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "matching_strategy": "invalid",
            }
        )
        builder = DatasetCSVBuilder(config)
        with self.assertRaises(ValueError):
            builder.match_image_to_mask(Path("any.tif"))

    def test_build_csv_no_images(self):
        empty_dir = os.path.join(self.tmp_dir, "empty")
        os.makedirs(empty_dir)
        config = OmegaConf.create(
            {"images_folder": empty_dir, "masks_folder": self.masks_dir}
        )
        builder = DatasetCSVBuilder(config)
        with self.assertRaises(ValueError):
            builder.build_csv(os.path.join(self.tmp_dir, "fail.csv"))

    def test_build_csv_no_pairs(self):
        os.remove(os.path.join(self.masks_dir, "img_1.tif"))
        os.remove(os.path.join(self.masks_dir, "img_2.tif"))
        config = OmegaConf.create(
            {"images_folder": self.images_dir, "masks_folder": self.masks_dir}
        )
        builder = DatasetCSVBuilder(config)
        with self.assertRaises(ValueError):
            builder.build_csv(os.path.join(self.tmp_dir, "fail_no_pairs.csv"))

    def test_get_dimensions_error(self):
        invalid_img = os.path.join(self.images_dir, "invalid.tif")
        with open(invalid_img, "w") as f:
            f.write("not a tif")

        config = OmegaConf.create(
            {"images_folder": self.images_dir, "masks_folder": self.masks_dir}
        )
        builder = DatasetCSVBuilder(config)
        w, h = builder._get_image_dimensions(Path(invalid_img))
        self.assertEqual(w, 0)
        self.assertEqual(h, 0)

    def test_match_custom_regex_no_match(self):
        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "matching_strategy": "custom_regex",
                "regex_pattern": r"(?P<id>ABC\d+)",
            }
        )
        builder = DatasetCSVBuilder(config)
        # image name "img_1" won't match "ABC\d+"
        mask = builder.match_image_to_mask(Path(self.images_dir) / "img_1.tif")
        self.assertIsNone(mask)

    def test_match_custom_regex_no_mask_with_same_id(self):
        self._create_dummy_raster(os.path.join(self.images_dir, "img_777.tif"))

        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "matching_strategy": "custom_regex",
                "regex_pattern": r"(?P<id>777)",
            }
        )
        builder = DatasetCSVBuilder(config)

        self.assertIsNone(
            builder.match_image_to_mask(Path(self.images_dir) / "img_777.tif")
        )

    def test_validate_dataset_duplicates_and_invalid_dims(self):
        config = OmegaConf.create(
            {"images_folder": self.images_dir, "masks_folder": self.masks_dir}
        )
        builder = DatasetCSVBuilder(config)

        df = pd.DataFrame(
            {
                "image": [
                    os.path.join(self.images_dir, "img_1.tif"),
                    os.path.join(self.images_dir, "img_1.tif"),
                ],
                "mask": [
                    os.path.join(self.masks_dir, "img_1.tif"),
                    os.path.join(self.masks_dir, "img_1.tif"),
                ],
                "width": [0, 0],
                "height": [0, 0],
            }
        )
        # Should log warnings but return True
        self.assertTrue(builder.validate_dataset(df))

    def test_match_same_basename_with_glob_fallback(self):
        # Create a mask with same stem but different extension than pattern
        mask_path = os.path.join(self.masks_dir, "img_unique.png")
        self._create_dummy_raster(mask_path)

        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "image_pattern": "*.tif",
                "mask_pattern": "*.tif",  # pattern says tif, but file is png
            }
        )
        builder = DatasetCSVBuilder(config)
        mask = builder.match_image_to_mask(Path(self.images_dir) / "img_unique.tif")
        self.assertIsNotNone(mask)
        self.assertEqual(mask.suffix, ".png")

    def test_match_same_basename_exact_path(self):
        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "mask_pattern": "mask_*.tif",
            }
        )
        builder = DatasetCSVBuilder(config)

        mask = builder.match_image_to_mask(Path(self.images_dir) / "img_1.tif")

        self.assertEqual(mask, Path(self.masks_dir) / "img_1.tif")

    def test_match_prefix_suffix_fallback(self):
        # Trigger the fallback glob in _match_prefix_suffix
        self._create_dummy_raster(os.path.join(self.images_dir, "p_stem_s.tif"))
        self._create_dummy_raster(
            os.path.join(self.masks_dir, "mp_stem_ms.png")
        )  # different extension

        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "matching_strategy": "prefix_suffix",
                "image_prefix": "p_",
                "image_suffix": "_s",
                "mask_prefix": "mp_",
                "mask_suffix": "_ms",
                "mask_pattern": "*.tif",  # doesn't match png
            }
        )
        builder = DatasetCSVBuilder(config)
        mask = builder.match_image_to_mask(Path(self.images_dir) / "p_stem_s.tif")
        self.assertIsNotNone(mask)
        self.assertEqual(mask.name, "mp_stem_ms.png")

    def test_match_prefix_suffix_returns_none_when_no_candidate_exists(self):
        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "matching_strategy": "prefix_suffix",
                "mask_prefix": "missing_",
            }
        )
        builder = DatasetCSVBuilder(config)

        self.assertIsNone(
            builder.match_image_to_mask(Path(self.images_dir) / "img_1.tif")
        )

    def test_match_prefix_suffix_no_pattern_extension(self):
        # Trigger 'if not mask_extension' in _match_prefix_suffix
        self._create_dummy_raster(os.path.join(self.images_dir, "p2_stem_s2.tif"))
        self._create_dummy_raster(os.path.join(self.masks_dir, "mp2_stem_ms2.tif"))

        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "matching_strategy": "prefix_suffix",
                "image_prefix": "p2_",
                "image_suffix": "_s2",
                "mask_prefix": "mp2_",
                "mask_suffix": "_ms2",
                "mask_pattern": "*",  # no extension
            }
        )
        builder = DatasetCSVBuilder(config)
        mask = builder.match_image_to_mask(Path(self.images_dir) / "p2_stem_s2.tif")
        self.assertIsNotNone(mask)
        self.assertEqual(mask.name, "mp2_stem_ms2.tif")

    def test_build_csv_with_many_unmatched_images(self):
        # Create 10 images without masks to trigger the long warning list
        for i in range(10, 20):
            self._create_dummy_raster(os.path.join(self.images_dir, f"no_mask_{i}.tif"))

        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "matching_strategy": "same_basename",
            }
        )
        builder = DatasetCSVBuilder(config)
        df = builder.build_csv(os.path.join(self.tmp_dir, "many_unmatched.csv"))
        # Should still have 2 pairs, and log many warnings
        self.assertEqual(len(df), 2)

    def test_match_custom_regex_multiple_candidates(self):
        # Regex matching with multiple possible mask candidates
        self._create_dummy_raster(os.path.join(self.images_dir, "multi_999.tif"))
        self._create_dummy_raster(os.path.join(self.masks_dir, "mask_A_999.tif"))
        self._create_dummy_raster(os.path.join(self.masks_dir, "mask_B_999.tif"))

        config = OmegaConf.create(
            {
                "images_folder": self.images_dir,
                "masks_folder": self.masks_dir,
                "matching_strategy": "custom_regex",
                "regex_pattern": r"(?P<id>\d+)",
            }
        )
        builder = DatasetCSVBuilder(config)
        mask = builder.match_image_to_mask(Path(self.images_dir) / "multi_999.tif")
        # Should return one of them
        self.assertIn(mask.name, ["mask_A_999.tif", "mask_B_999.tif"])


if __name__ == "__main__":
    unittest.main()
