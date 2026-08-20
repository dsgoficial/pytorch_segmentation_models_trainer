# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import tempfile
import numpy as np
import rasterio
from pathlib import Path
from rasterio.transform import from_origin
from pytorch_segmentation_models_trainer.tools.evaluation.direct_folder_evaluator import (
    DirectFolderEvaluator,
    prepare_evaluation_csv_from_folders,
)


class TestDirectFolderEvaluator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.gt_dir = os.path.join(self.test_dir, "gt")
        self.pred_dir = os.path.join(self.test_dir, "pred")
        os.makedirs(self.gt_dir)
        os.makedirs(self.pred_dir)

        # Create some dummy files
        Path(os.path.join(self.gt_dir, "img1.tif")).touch()
        Path(os.path.join(self.gt_dir, "img2.tif")).touch()
        Path(os.path.join(self.pred_dir, "img1_output.tif")).touch()
        Path(os.path.join(self.pred_dir, "seg_img2.tif")).touch()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_init_raises_if_not_exists(self):
        with self.assertRaises(FileNotFoundError):
            DirectFolderEvaluator("/invalid/path", self.pred_dir)
        with self.assertRaises(FileNotFoundError):
            DirectFolderEvaluator(self.gt_dir, "/invalid/path")

    def test_generate_stem_variants(self):
        evaluator = DirectFolderEvaluator(self.gt_dir, self.pred_dir)
        variants = evaluator._generate_stem_variants("img1")
        self.assertIn("img1", variants)

        variants = evaluator._generate_stem_variants("seg_img1")
        self.assertIn("img1", variants)

        variants = evaluator._generate_stem_variants("img1_output")
        self.assertIn("img1", variants)

        variants = evaluator._generate_stem_variants("seg_img1_output")
        self.assertIn("img1", variants)

    def test_build_matching_pairs(self):
        evaluator = DirectFolderEvaluator(self.gt_dir, self.pred_dir)
        pairs = evaluator.build_matching_pairs()

        self.assertEqual(len(pairs), 2)
        stems = [p["name"] for p in pairs]
        self.assertIn("img1", stems)
        self.assertIn("img2", stems)

    def test_build_matching_pairs_empty_and_missing_predictions(self):
        empty_gt_dir = os.path.join(self.test_dir, "empty_gt")
        os.makedirs(empty_gt_dir)
        evaluator = DirectFolderEvaluator(empty_gt_dir, self.pred_dir)
        with self.assertRaises(ValueError):
            evaluator.build_matching_pairs()

        unmatched_gt_dir = os.path.join(self.test_dir, "unmatched_gt")
        os.makedirs(unmatched_gt_dir)
        for idx in range(6):
            Path(os.path.join(unmatched_gt_dir, f"missing_{idx}.tif")).touch()

        evaluator = DirectFolderEvaluator(unmatched_gt_dir, self.pred_dir)
        with self.assertRaises(ValueError):
            evaluator.build_matching_pairs()

    @patch("rasterio.open")
    def test_create_evaluation_csv(self, mock_rasterio):
        # Mock rasterio source
        mock_src = MagicMock()
        mock_src.width = 256
        mock_src.height = 256
        mock_rasterio.return_value.__enter__.return_value = mock_src

        evaluator = DirectFolderEvaluator(self.gt_dir, self.pred_dir)
        csv_path = os.path.join(self.test_dir, "out.csv")
        df = evaluator.create_evaluation_csv(csv_path)

        self.assertTrue(os.path.exists(csv_path))
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["width"], 256)

    def test_load_pair_reads_masks(self):
        gt_path = os.path.join(self.test_dir, "gt_real.tif")
        pred_path = os.path.join(self.test_dir, "pred_real.tif")
        transform = from_origin(0, 2, 1, 1)
        data = np.arange(4, dtype=np.uint8).reshape(1, 2, 2)
        for path in [gt_path, pred_path]:
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(data)

        evaluator = DirectFolderEvaluator(self.gt_dir, self.pred_dir)
        gt_mask, pred_mask = evaluator.load_pair(
            {"gt_path": gt_path, "pred_path": pred_path}
        )

        self.assertEqual(gt_mask.shape, (2, 2))
        self.assertTrue(np.array_equal(gt_mask, pred_mask))

    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.direct_folder_evaluator.DirectFolderEvaluator.create_evaluation_csv"
    )
    def test_prepare_helper(self, mock_create):
        csv_path = "/tmp/fake.csv"
        prepare_evaluation_csv_from_folders(self.gt_dir, self.pred_dir, csv_path)
        self.assertTrue(mock_create.called)

    def test_module_main_guard(self):
        import runpy
        import sys

        gt_dir = os.path.join(self.test_dir, "main_gt")
        pred_dir = os.path.join(self.test_dir, "main_pred")
        os.makedirs(gt_dir)
        os.makedirs(pred_dir)
        transform = from_origin(0, 2, 1, 1)
        data = np.ones((1, 2, 2), dtype=np.uint8)
        for path in [
            os.path.join(gt_dir, "tile.tif"),
            os.path.join(pred_dir, "tile.tif"),
        ]:
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(data)

        module = "pytorch_segmentation_models_trainer.tools.evaluation.direct_folder_evaluator"
        output_csv = os.path.join(self.test_dir, "main_out.csv")
        argv = [
            "direct_folder_evaluator.py",
            "--gt-folder",
            gt_dir,
            "--pred-folder",
            pred_dir,
            "--output-csv",
            output_csv,
        ]
        with patch.object(sys, "argv", argv):
            runpy.run_module(module, run_name="__main__")

        self.assertTrue(os.path.exists(output_csv))


if __name__ == "__main__":
    unittest.main()
