# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-03-01
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
from unittest.mock import MagicMock, patch

import hydra
import torch
from hydra import compose, initialize
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldModel,
)
from pytorch_segmentation_models_trainer.predict import (
    instantiate_inference_processor,
    instantiate_model_from_checkpoint,
    instantiate_polygonizer,
)
from pytorch_segmentation_models_trainer.tools.inference.inference_processors import (
    AbstractInferenceProcessor,
)
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import (
    TemplatePolygonizerProcessor,
)
from pytorch_segmentation_models_trainer.utils.os_utils import (
    create_folder,
)
from tests.mock_utils import create_dummy_checkpoint
from tests.utils import BasicTestCase

config_name_list = ["predict.yaml"]

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")

frame_field_root_dir = os.path.join(
    current_dir, "testing_data", "data", "frame_field_data"
)

device = "cpu"

pretrained_checkpoints_download_links = {
    "frame_field_resnet152_unet_200_epochs": "https://github.com/phborba/pytorch_smt_pretrained_weights/releases/download/v0.1/frame_field_resnet152_unet_200_epochs.ckpt"
}


class Test_Predict(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.output_dir = self.make_temp_dir()
        self.output_vector_file = os.path.join(self.output_dir, "output.geojson")
        self.output_file_name = "output.geojson"
        self.csv_ds_file = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        self.frame_field_ds = self.get_frame_field_ds()
        self.checkpoint_file_path = self.get_checkpoint_file(
            "frame_field_resnet152_unet_200_epochs.ckpt"
        )

    def get_frame_field_ds(self):
        with initialize(config_path="./test_configs", version_base=None):
            cfg = compose(
                config_name="frame_field_dataset.yaml",
                overrides=[
                    "input_csv_path=" + self.csv_ds_file,
                    "root_dir=" + frame_field_root_dir,
                ],
            )
            frame_field_ds = hydra.utils.instantiate(cfg, _recursive_=False)
        return frame_field_ds

    def get_checkpoint_file(self, file_name):
        checkpoint_folder = create_folder(os.path.join(root_dir, "data", "checkpoints"))
        ckeckpoint_file_path = os.path.join(checkpoint_folder, file_name)
        if not os.path.isfile(ckeckpoint_file_path):
            create_dummy_checkpoint(ckeckpoint_file_path)
        return ckeckpoint_file_path

    def make_inference(self, sample, frame_field_model):
        with torch.no_grad():
            out = frame_field_model(sample)
        self.assertEqual(
            out["seg"].shape,
            torch.Size([sample.shape[0], 2, sample.shape[-2], sample.shape[-1]]),
        )
        self.assertEqual(
            out["crossfield"].shape,
            torch.Size([sample.shape[0], 4, sample.shape[-2], sample.shape[-1]]),
        )

    def test_instantiate_model_from_checkpoint(self):
        with initialize(config_path="./test_configs", version_base=None):
            cfg = compose(
                config_name="predict.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + self.csv_ds_file,
                    "val_dataset.input_csv_path=" + self.csv_ds_file,
                    "checkpoint_path=" + self.checkpoint_file_path,
                ],
            )
        model = instantiate_model_from_checkpoint(cfg)
        self.assertIsInstance(model, FrameFieldModel)
        self.make_inference(torch.ones([2, 3, 224, 224]), model)

    def test_instantiate_polygonizer(self):
        with initialize(config_path="./test_configs", version_base=None):
            cfg = compose(
                config_name="predict.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + self.csv_ds_file,
                    "val_dataset.input_csv_path=" + self.csv_ds_file,
                    "checkpoint_path=" + self.checkpoint_file_path,
                ],
            )
        polygonizer = instantiate_polygonizer(cfg)
        self.assertIsInstance(polygonizer, TemplatePolygonizerProcessor)

    def test_instantiate_inference_processor(self):
        with initialize(config_path="./test_configs", version_base=None):
            cfg = compose(
                config_name="predict.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + self.csv_ds_file,
                    "val_dataset.input_csv_path=" + self.csv_ds_file,
                    "checkpoint_path=" + self.checkpoint_file_path,
                ],
            )
        inference_processor = instantiate_inference_processor(cfg)
        self.assertIsInstance(inference_processor, AbstractInferenceProcessor)

    @patch(
        "pytorch_segmentation_models_trainer.predict.instantiate_inference_processor"
    )
    @patch("pytorch_segmentation_models_trainer.predict.get_images")
    @patch("pytorch_segmentation_models_trainer.predict.tqdm")
    def test_predict_main_function(
        self, mock_tqdm, mock_get_images, mock_instantiate_proc
    ):
        with initialize(config_path="./test_configs", version_base=None):
            cfg = compose(
                config_name="predict.yaml",
                overrides=[
                    "train_dataset.input_csv_path=" + self.csv_ds_file,
                    "val_dataset.input_csv_path=" + self.csv_ds_file,
                    "checkpoint_path=" + self.checkpoint_file_path,
                ],
            )

        mock_processor = MagicMock()
        mock_instantiate_proc.return_value = mock_processor
        mock_get_images.return_value = ["img1.tif", "img2.tif"]
        mock_tqdm.side_effect = lambda x: x  # pass-through

        from pytorch_segmentation_models_trainer.predict import predict

        predict(cfg)

        self.assertEqual(mock_processor.process.call_count, 2)
        mock_processor.process.assert_any_call(
            "img1.tif", save_inference_output=True, inference_threshold=0.5
        )
