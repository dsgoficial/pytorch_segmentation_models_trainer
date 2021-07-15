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
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import FrameFieldModel
import unittest

import hydra
import numpy as np
import segmentation_models_pytorch as smp
import torch
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from parameterized import parameterized
from pytorch_segmentation_models_trainer.tools.inference.export_inference import (
    MultipleRasterExportInferenceStrategy, RasterExportInferenceStrategy)
from pytorch_segmentation_models_trainer.tools.inference.inference_processors import (
    SingleImageFromFrameFieldProcessor, SingleImageInfereceProcessor)
from pytorch_segmentation_models_trainer.utils.os_utils import (create_folder,
                                                                remove_folder)

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, 'testing_data')

frame_field_root_dir = os.path.join(
    current_dir, 'testing_data', 'data', 'frame_field_data')

device = 'cpu'

class Test_TestInference(unittest.TestCase):

    def setUp(self):
        self.output_dir = create_folder(os.path.join(root_dir, 'test_output'))

    def tearDown(self):
        remove_folder(self.output_dir)

    def test_create_inference_from_inference_processor(self) -> None:
        csv_path = os.path.join(frame_field_root_dir, 'dsg_dataset.csv')
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="frame_field_dataset.yaml",
                overrides=[
                    'input_csv_path='+csv_path,
                    'root_dir='+frame_field_root_dir
                ]
            )
            frame_field_ds = hydra.utils.instantiate(cfg)
        output_file_path = os.path.join(self.output_dir, 'output.tif')
        inference_processor = SingleImageInfereceProcessor(
            model=smp.Unet(),
            device=device,
            batch_size=8,
            export_strategy=RasterExportInferenceStrategy(
                input_raster_path=frame_field_ds[0]['path'],
                output_file_path=output_file_path
            )
        )
        inference_processor.process(
            image_path=frame_field_ds[0]['path']
        )
        assert os.path.isfile(output_file_path)
    
    def test_create_frame_field_inference_from_inference_processor(self) -> None:
        csv_path = os.path.join(frame_field_root_dir, 'dsg_dataset.csv')
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="frame_field_dataset.yaml",
                overrides=[
                    'input_csv_path='+csv_path,
                    'root_dir='+frame_field_root_dir
                ]
            )
            frame_field_ds = hydra.utils.instantiate(cfg)
        inference_processor = SingleImageFromFrameFieldProcessor(
            model=FrameFieldModel(
                segmentation_model=smp.Unet(),
                seg_params={
                    "compute_interior": True,
                    "compute_edge": True,
                    "compute_vertex": False
                }
            ),
            device=device,
            batch_size=8,
            export_strategy=MultipleRasterExportInferenceStrategy(
                input_raster_path=frame_field_ds[0]['path'],
                output_folder=self.output_dir,
                output_basename='output.tif'
            ),
            mask_bands=2
        )
        inference_processor.process(
            image_path=frame_field_ds[0]['path']
        )
        assert os.path.isfile(os.path.join(self.output_dir, 'seg_output.tif'))
        assert os.path.isfile(os.path.join(self.output_dir, 'crossfield_output.tif'))
