# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-02-25
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
import unittest
from pathlib import Path

# from unittest import IsolatedAsyncioTestCase

import albumentations as A
import hydra
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from fastapi.testclient import TestClient
from hydra import compose, initialize
from hydra.utils import instantiate
from parameterized import parameterized
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldSegmentationPLModel,
)
from pytorch_segmentation_models_trainer.server import app, get_inference_processor
from pytorch_segmentation_models_trainer.tools.data_handlers.data_writer import (
    VectorFileDataWriter,
)
from pytorch_segmentation_models_trainer.tools.inference.inference_processors import (
    SingleImageFromFrameFieldProcessor,
)
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import (
    ASMConfig,
    ASMPolygonizerProcessor,
)
from pytorch_segmentation_models_trainer.utils.os_utils import (
    create_folder,
    remove_folder,
)
from torchvision.datasets.utils import download_url

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")

frame_field_root_dir = os.path.join(
    current_dir, "testing_data", "data", "frame_field_data"
)

device = "cpu"

pretrained_checkpoints_download_links = {
    "frame_field_resnet152_unet_200_epochs": "https://github.com/phborba/pytorch_smt_pretrained_weights/releases/download/v0.1/frame_field_resnet152_unet_200_epochs.ckpt"
}
output_dir = create_folder(os.path.join(root_dir, "test_output"))


def get_asm_polygonizer():
    config = ASMConfig()
    asm_output_file_path = os.path.join(output_dir, "asm_polygonizer.geojson")
    data_writer = VectorFileDataWriter(output_file_path=asm_output_file_path)
    return ASMPolygonizerProcessor(data_writer=data_writer, config=config)


def get_model_for_eval():
    csv_path = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
    with initialize(config_path="./test_configs"):
        cfg = compose(
            config_name="frame_field_for_inference.yaml",
            overrides=[
                "train_dataset.input_csv_path=" + csv_path,
                "train_dataset.root_dir=" + frame_field_root_dir,
                "val_dataset.input_csv_path=" + csv_path,
                "val_dataset.root_dir=" + frame_field_root_dir,
            ],
        )
    checkpoint_file_path = get_checkpoint_file(
        "frame_field_resnet152_unet_200_epochs.ckpt"
    )
    pl_model = FrameFieldSegmentationPLModel.load_from_checkpoint(
        checkpoint_file_path, cfg=cfg
    )
    model = pl_model.model
    model.eval()
    return model


def get_checkpoint_file(file_name):
    checkpoint_folder = create_folder(os.path.join(root_dir, "data", "checkpoints"))
    checkpoint_file_path = os.path.join(checkpoint_folder, file_name)
    if not os.path.isfile(checkpoint_file_path):
        download_url(
            url=pretrained_checkpoints_download_links[file_name.split(".")[0]],
            root=checkpoint_folder,
            filename=file_name,
        )
    return checkpoint_file_path


def get_settings_override():
    inference_processor = SingleImageFromFrameFieldProcessor(
        model=get_model_for_eval(),
        device=device,
        batch_size=8,
        export_strategy=None,
        mask_bands=2,
        polygonizer=get_asm_polygonizer(),
    )
    inference_processor.polygonizer.data_writer = None
    return inference_processor


client = TestClient(app)
app.dependency_overrides[get_inference_processor] = get_settings_override


class Test_TestInferenceService(unittest.TestCase):
    def setUp(self):
        self.output_dir = create_folder(os.path.join(root_dir, "test_output"))
        self.frame_field_ds = self.get_frame_field_ds()

    def tearDown(self):
        remove_folder(output_dir)

    def get_frame_field_ds(self):
        csv_path = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="frame_field_dataset.yaml",
                overrides=[
                    "input_csv_path=" + csv_path,
                    "root_dir=" + frame_field_root_dir,
                ],
            )
            frame_field_ds = hydra.utils.instantiate(cfg, _recursive_=False)
        return frame_field_ds

    @parameterized.expand(
        [
            (None,),
            (
                {
                    "_target_": "pytorch_segmentation_models_trainer.tools.polygonization.polygonizer.SimplePolygonizerProcessor",
                    "config": {
                        "data_level": 0.9,
                        "tolerance": 1.0,
                        "seg_threshold": 0.9,
                        "min_area": 10,
                    },
                    "data_writer": None,
                },
            ),
        ]
    )
    def test_inference_from_service(self, polygonizer) -> None:
        file_path = self.frame_field_ds[0]["path"]
        response = client.get(f"/polygonize/?file_path={file_path}", json=polygonizer)
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.json()["features"]), 0)
