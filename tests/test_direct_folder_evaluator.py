# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
import tempfile
import pandas as pd
from pathlib import Path
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

    def test_generate_stem_variants(self):
        evaluator = DirectFolderEvaluator(self.gt_dir, self.pred_dir)
        variants = evaluator._generate_stem_variants("img1")
        self.assertIn("img1", variants)

        variants = evaluator._generate_stem_variants("seg_img1")
        self.assertIn("img1", variants)

        variants = evaluator._generate_stem_variants("img1_output")
        self.assertIn("img1", variants)

    def test_build_matching_pairs(self):
        evaluator = DirectFolderEvaluator(self.gt_dir, self.pred_dir)
        pairs = evaluator.build_matching_pairs()

        self.assertEqual(len(pairs), 2)
        stems = [p["name"] for p in pairs]
        self.assertIn("img1", stems)
        self.assertIn("img2", stems)

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

    @patch(
        "pytorch_segmentation_models_trainer.tools.evaluation.direct_folder_evaluator.DirectFolderEvaluator.create_evaluation_csv"
    )
    def test_prepare_helper(self, mock_create):
        csv_path = "/tmp/fake.csv"
        prepare_evaluation_csv_from_folders(self.gt_dir, self.pred_dir, csv_path)
        self.assertTrue(mock_create.called)


if __name__ == "__main__":
    unittest.main()
