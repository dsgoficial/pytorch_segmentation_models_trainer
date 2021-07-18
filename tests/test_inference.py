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

import albumentations as A
import hydra
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from parameterized import parameterized
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import (
    FrameFieldModel, FrameFieldSegmentationPLModel)
from pytorch_segmentation_models_trainer.tools.data_handlers.data_writer import \
    VectorFileDataWriter
from pytorch_segmentation_models_trainer.tools.inference.export_inference import (
    MultipleRasterExportInferenceStrategy, RasterExportInferenceStrategy)
from pytorch_segmentation_models_trainer.tools.inference.inference_processors import (
    SingleImageFromFrameFieldProcessor, SingleImageInfereceProcessor)
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import (
    ACMConfig, ACMPolygonizerProcessor, ASMConfig, ASMPolygonizerProcessor,
    SimplePolConfig, SimplePolygonizerProcessor)
from pytorch_segmentation_models_trainer.utils.os_utils import (create_folder,
                                                                remove_folder)
from torchvision.datasets.utils import download_url

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, 'testing_data')

frame_field_root_dir = os.path.join(
    current_dir, 'testing_data', 'data', 'frame_field_data')

device = 'cpu'

pretrained_checkpoints_download_links = {
    'frame_field_resnet152_unet_200_epochs': 'https://github.com/phborba/pytorch_smt_pretrained_weights/releases/download/v0.1/frame_field_resnet152_unet_200_epochs.ckpt'
}

class Test_TestInference(unittest.TestCase):

    def setUp(self):
        self.output_dir = create_folder(os.path.join(root_dir, 'test_output'))
        self.frame_field_ds = self.get_frame_field_ds()
        with rasterio.open(self.frame_field_ds[0]['path'], 'r') as raster_ds:
            self.crs = raster_ds.crs
            self.profile = raster_ds.profile
            self.transform = raster_ds.transform
        self.polygonizers_dict = {
            'simple': self.get_simple_polygonizer(),
            'acm': self.get_acm_polygonizer(),
            'asm': self.get_asm_polygonizer()
        }
        self.center_crop = A.CenterCrop(512,512)

    def tearDown(self):
        remove_folder(self.output_dir)
    
    def get_frame_field_ds(self):
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
        return frame_field_ds
    
    def get_simple_polygonizer(self):
        config = SimplePolConfig()
        self.simple_output_file_path = os.path.join(self.output_dir, 'simple_polygonizer.geojson')
        data_writer = VectorFileDataWriter(
            output_file_path=self.simple_output_file_path,
            crs=self.crs
        )
        return SimplePolygonizerProcessor(
            crs=self.crs,
            transform=self.transform,
            data_writer=data_writer,
            config=config
        )
    
    def get_acm_polygonizer(self):
        config = ACMConfig()
        self.acm_output_file_path = os.path.join(self.output_dir, 'acm_polygonizer.geojson')
        data_writer = VectorFileDataWriter(
            output_file_path=self.acm_output_file_path,
            crs=self.crs
        )
        return ACMPolygonizerProcessor(
            crs=self.crs,
            transform=self.transform,
            data_writer=data_writer,
            config=config
        )

    def get_asm_polygonizer(self):
        config = ASMConfig()
        self.asm_output_file_path = os.path.join(self.output_dir, 'asm_polygonizer.geojson')
        data_writer = VectorFileDataWriter(
            output_file_path=self.asm_output_file_path,
            crs=self.crs
        )
        return ASMPolygonizerProcessor(
            crs=self.crs,
            transform=self.transform,
            data_writer=data_writer,
            config=config
        )

    def get_model_for_eval(self):
        csv_path = os.path.join(frame_field_root_dir, 'dsg_dataset.csv')
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="frame_field_for_inference.yaml",
                overrides=[
                    'train_dataset.input_csv_path='+csv_path,
                    'train_dataset.root_dir='+frame_field_root_dir,
                    'val_dataset.input_csv_path='+csv_path,
                    'val_dataset.root_dir='+frame_field_root_dir,
                ]
            )
        checkpoint_file_path = self.get_checkpoint_file('frame_field_resnet152_unet_200_epochs.ckpt')
        pl_model = FrameFieldSegmentationPLModel.load_from_checkpoint(
            checkpoint_file_path,
            cfg=cfg
        )
        model = pl_model.model
        model.eval()
        return model
    
    def get_checkpoint_file(self, file_name):
        checkpoint_folder = create_folder(os.path.join(root_dir, 'data', 'checkpoints'))
        ckeckpoint_file_path = os.path.join(checkpoint_folder, file_name)
        if not os.path.isfile(ckeckpoint_file_path):
            download_url(
                url=pretrained_checkpoints_download_links[file_name.split('.')[0]],
                root=checkpoint_folder,
                filename=file_name
            )
        return ckeckpoint_file_path

    def test_create_inference_from_inference_processor(self) -> None:
        
        output_file_path = os.path.join(self.output_dir, 'output.tif')
        inference_processor = SingleImageInfereceProcessor(
            model=smp.Unet(),
            device=device,
            batch_size=8,
            export_strategy=RasterExportInferenceStrategy(
                input_raster_path=self.frame_field_ds[0]['path'],
                output_file_path=output_file_path
            )
        )
        inference_processor.process(
            image_path=self.frame_field_ds[0]['path']
        )
        assert os.path.isfile(output_file_path)
    
    @parameterized.expand(
        [
            (False,),
            (True,),
        ]
    )
    def test_create_frame_field_inference_from_inference_processor(self, with_polygonizer) -> None:
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
                input_raster_path=self.frame_field_ds[0]['path'],
                output_folder=self.output_dir,
                output_basename='output.tif'
            ),
            mask_bands=2,
            polygonizer=self.polygonizers_dict['simple'] if with_polygonizer else None
        )
        inference_processor.process(
            image_path=self.frame_field_ds[0]['path']
        )
        assert os.path.isfile(os.path.join(self.output_dir, 'seg_output.tif'))
        assert os.path.isfile(os.path.join(self.output_dir, 'crossfield_output.tif'))

    @parameterized.expand(
        [
            ('simple',),
            ('acm',),
            ('asm',),
        ]
    )
    def test_create_frame_field_inference_from_pretrained_with_polygonize(self, polygonizer_key) -> None:
        inference_processor = SingleImageFromFrameFieldProcessor(
            model=self.get_model_for_eval(),
            device=device,
            batch_size=8,
            export_strategy=MultipleRasterExportInferenceStrategy(
                input_raster_path=self.frame_field_ds[0]['path'],
                output_folder=self.output_dir,
                output_basename=f'{polygonizer_key}_output.tif'
            ),
            mask_bands=2,
            polygonizer=self.polygonizers_dict[polygonizer_key]
        )
        inference_processor.process(
            image_path=self.frame_field_ds[0]['path'],
            threshold=0.5
        )
        assert os.path.isfile(os.path.join(self.output_dir, f'seg_{polygonizer_key}_output.tif'))
        assert os.path.isfile(os.path.join(self.output_dir, f'crossfield_{polygonizer_key}_output.tif'))
        assert os.path.isfile(getattr(self, f"{polygonizer_key}_output_file_path"))
