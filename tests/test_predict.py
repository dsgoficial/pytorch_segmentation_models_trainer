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
import subprocess
from pathlib import Path
import unittest
import warnings

import hydra
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from parameterized import parameterized
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldModel,
)
from pytorch_segmentation_models_trainer.predict import (
    instantiate_inference_processor,
    instantiate_model_from_checkpoint,
    instantiate_polygonizer,
    predict,
)
from pytorch_segmentation_models_trainer.tools.inference.inference_processors import (
    AbstractInferenceProcessor,
)
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import (
    TemplatePolygonizerProcessor,
)
from pytorch_segmentation_models_trainer.utils.os_utils import (
    create_folder,
    remove_folder,
)
from torchvision.datasets.utils import download_url


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


class Test_Predict(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self.output_dir = create_folder(os.path.join(root_dir, "test_output"))
        self.output_vector_file = os.path.join(self.output_dir, "output.geojson")
        self.output_file_name = "output.geojson"
        self.csv_ds_file = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        self.frame_field_ds = self.get_frame_field_ds()
        self.checkpoint_file_path = self.get_checkpoint_file(
            "frame_field_resnet152_unet_200_epochs.ckpt"
        )

    def tearDown(self):
        remove_folder(self.output_dir)

    def get_frame_field_ds(self):
        with initialize(config_path="./test_configs"):
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
            download_url(
                url=pretrained_checkpoints_download_links[file_name.split(".")[0]],
                root=checkpoint_folder,
                filename=file_name,
            )
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
        with initialize(config_path="./test_configs"):
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
        with initialize(config_path="./test_configs"):
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
        with initialize(config_path="./test_configs"):
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

    # @parameterized.expand(config_name_list)
    # def test_run_predict_from_object(self, config_name: str) -> None:
    #     with initialize(config_path="./test_configs"):
    #         cfg = compose(
    #             config_name=config_name,
    #             overrides=[
    #                 f"train_dataset.input_csv_path={self.csv_ds_file}",
    #                 f"val_dataset.input_csv_path={self.csv_ds_file}",
    #                 f"checkpoint_path={self.checkpoint_file_path}",
    #                 f"inference_image_reader.input_csv_path={self.csv_ds_file}",
    #                 f"inference_image_reader.root_dir={frame_field_root_dir}",
    #                 f"polygonizer.data_writer.output_file_folder={self.output_dir}",
    #                 f"polygonizer.data_writer.output_file_name={self.output_file_name}",
    #                 f"export_strategy.output_folder={self.output_dir}",
    #             ],
    #         )
    #         predict_obj = predict(cfg)
    #         assert os.path.isfile(self.output_vector_file)
    #         for i in range(cfg.inference_image_reader.n_first_rows_to_read):
    #             name = Path(self.frame_field_ds[i]["path"]).stem
    #             assert os.path.isfile(
    #                 os.path.join(self.output_dir, f"seg_{name}_inference.tif")
    #             )
    #             assert os.path.isfile(
    #                 os.path.join(self.output_dir, f"crossfield_{name}_inference.tif")
    #             )
