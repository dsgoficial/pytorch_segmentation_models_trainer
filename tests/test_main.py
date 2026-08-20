# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch
from omegaconf import OmegaConf
from pytorch_segmentation_models_trainer.main import main


class TestMainRouting(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create({"mode": "train"})

    @patch("pytorch_segmentation_models_trainer.train.train")
    def test_routes_to_train(self, mock_train):
        self.cfg.mode = "train"
        main(self.cfg)
        self.assertTrue(mock_train.called)

    @patch("pytorch_segmentation_models_trainer.predict.predict")
    def test_routes_to_predict(self, mock_predict):
        self.cfg.mode = "predict"
        main(self.cfg)
        self.assertTrue(mock_predict.called)

    @patch("pytorch_segmentation_models_trainer.evaluate_experiments.evaluate")
    def test_routes_to_evaluate(self, mock_eval):
        self.cfg.mode = "evaluate-experiments"
        main(self.cfg)
        self.assertTrue(mock_eval.called)

    @patch("pytorch_segmentation_models_trainer.build_mask.build_masks")
    def test_routes_to_build_mask(self, mock_build):
        self.cfg.mode = "build-mask"
        main(self.cfg)
        self.assertTrue(mock_build.called)

    @patch("pytorch_segmentation_models_trainer.convert_ds.convert_dataset")
    def test_routes_to_convert_ds(self, mock_convert):
        self.cfg.mode = "convert-dataset"
        main(self.cfg)
        self.assertTrue(mock_convert.called)

    def test_raises_on_invalid_mode(self):
        self.cfg.mode = "invalid"
        with self.assertRaises(NotImplementedError):
            main(self.cfg)


if __name__ == "__main__":
    unittest.main()
