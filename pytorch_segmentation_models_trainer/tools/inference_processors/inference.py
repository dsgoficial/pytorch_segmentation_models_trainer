# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-07-14
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba - Cartographic Engineer
                                                            @ Brazilian Army
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
from abc import ABC, abstractmethod

import albumentations as A
import cv2
import numpy as np
import rasterio
import torch
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy
from rasterio.plot import reshape_as_image, reshape_as_raster
from torch.utils.data import DataLoader


class AbstractInferenceProcessor(ABC):
    """
    Abstract method to process inferences
    """
    def __init__(self, model, device, model_input_shape=None, step_shape=None, config=None):
        self.model = model
        self.device = device
        self.config = config
        self.model.to(device)
        self.model_input_shape = (448, 448) if model_input_shape is None else model_input_shape
        self.step_shape = (224, 224) if step_shape is None else step_shape
        self.normalize = A.Normalize()
    
    def process(self, image_path: str, output_folder: str) -> str:
        with rasterio.open(image_path) as src:
            profile = src.profile
        image = cv2.imread(image_path)
        inference = self.make_inference(image)
        output_path = os.path.join(
            output_folder,
            os.path.basename(image_path)
        )
        self.save_inference(inference, output_path, profile=profile)
        return output_path

    @abstractmethod
    def make_inference(self, image: np.array, threshold=0.5) -> torch.Tensor:
        """Makes model inference. Must be reimplemented in children classes

        Args:
            image (np.array): image to run inference

        Returns:
            torch.Tensor: model inference
        """
        pass

    @abstractmethod
    def save_inference(self, inference: torch.Tensor, output: str, **kwargs) -> None:
        """Saves the model inference. Must be reimplemented in children classes

        Args:
            inference (np.array): image to run inference
            output (str): output path

        Returns:
            torch.Tensor: model inference
        """
        pass

class SingleImageInfereceProcessor(AbstractInferenceProcessor):
    def __init__(self, model, device, config=None):
        super(SingleImageInfereceProcessor, self).__init__(model, device, config)

    def make_inference(self, image: np.array, threshold=0.5) -> torch.Tensor:
        """Makes model inference.

        Args:
            image (np.array): image to run inference

        Returns:
            torch.Tensor: model inference
        """
        tiler = ImageSlicer(
            image.shape,
            tile_size=self.model_input_shape,
            tile_step=self.step_shape
        )
        # image = self.normalize(image=reshape_as_image(image))['image']
        # HCW -> CHW. Optionally, do normalization here
        tiles = [tensor_from_rgb_image(tile)/255. for tile in tiler.split(image)]

        # Allocate a CUDA buffer for holding entire mask
        merger = TileMerger(tiler.target_shape, 1, tiler.weight, device=self.device)

        # Run predictions for tiles and accumulate them
        with torch.no_grad():
            for tiles_batch, coords_batch in DataLoader(
                list(zip(tiles, tiler.crops)), batch_size=8, pin_memory=True
            ):
                tiles_batch = tiles_batch.float().to(self.device)
                pred_batch = self.model(tiles_batch)
                merger.integrate_batch(pred_batch, coords_batch)

        # Normalize accumulated mask and convert back to numpy
        merged_mask = np.moveaxis(to_numpy(merger.merge()), 0, -1).astype(np.uint8)
        merged_mask = tiler.crop_to_orignal_size(merged_mask)

        return (merged_mask > threshold).astype(np.uint8)

    def save_inference(self, inference: torch.Tensor, output: str, **kwargs) -> None:
        profile = kwargs.get('profile', {})
        profile['count'] = inference.shape[-1]
        with rasterio.open(output, 'w', **profile) as out:
            out.write(reshape_as_raster(inference))
