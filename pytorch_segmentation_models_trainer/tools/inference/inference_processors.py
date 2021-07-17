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
from typing import Dict, List, Union

import albumentations as A
import cv2
import numpy as np
import torch
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image, to_numpy
from torch.utils.data import DataLoader

class AbstractInferenceProcessor(ABC):
    """
    Abstract method to process inferences
    """
    def __init__(self, model, device, batch_size, export_strategy, polygonizer=None, model_input_shape=None, step_shape=None, mask_bands=1, config=None):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.export_strategy = export_strategy
        self.polygonizer = polygonizer
        self.config = config
        self.model.to(device)
        self.model_input_shape = (448, 448) if model_input_shape is None else model_input_shape
        self.step_shape = (224, 224) if step_shape is None else step_shape
        self.mask_bands = mask_bands
        self.normalize = A.Normalize()
    
    def process(self, image_path: str, threshold: float=0.5) -> str:
        image = cv2.imread(image_path)
        inference = self.make_inference(image)
        if self.polygonizer is not None:
            self.polygonizer.process(
                {
                    key: tensor_from_rgb_image(value).unsqueeze(0) for key, value in inference.items()
                }
            )
        inference['seg'] = (inference['seg'] > threshold).astype(np.uint8)
        return self.export_strategy.save_inference(inference)

    @abstractmethod
    def make_inference(self, image: np.array)  -> Union[np.array, dict]:
        """Makes model inference. Must be reimplemented in children classes

        Args:
            image (np.array): image to run inference

        Returns:
            torch.Tensor: model inference
        """
        pass

class SingleImageInfereceProcessor(AbstractInferenceProcessor):
    def __init__(self, model, device, batch_size, export_strategy, polygonizer=None, model_input_shape=None, step_shape=None, mask_bands=1, config=None):
        super(SingleImageInfereceProcessor, self).__init__(
            model, device, batch_size, export_strategy=export_strategy, polygonizer=polygonizer, model_input_shape=model_input_shape,\
            step_shape=step_shape, mask_bands=mask_bands, config=config
        )

    def make_inference(self, image: np.array) -> Union[np.array, dict]:
        """Makes model inference.

        Args:
            image (np.array): image to run inference

        Returns:
            Union[np.array, dict]: model inference
        """
        tiler = ImageSlicer(
            image.shape,
            tile_size=self.model_input_shape,
            tile_step=self.step_shape
        )
        normalized_image = self.normalize(image=image)['image']
        tiles = [tensor_from_rgb_image(tile) for tile in tiler.split(normalized_image)]
        merger_dict = self.get_merger_dict(tiler)
        self.predict_and_merge(tiles, tiler, merger_dict)
        return self.merge_masks(tiler, merger_dict)
    
    def get_merger_dict(self, tiler: ImageSlicer):
        return {
            'seg': TileMerger(tiler.target_shape, self.mask_bands, tiler.weight, device=self.device)
        }
    
    def predict_and_merge(self, tiles: List[np.array], tiler: ImageSlicer, merger_dict: Dict[str, TileMerger]):
        with torch.no_grad():
            for tiles_batch, coords_batch in DataLoader(
                list(zip(tiles, tiler.crops)), batch_size=self.batch_size, pin_memory=True
            ):
                tiles_batch = tiles_batch.float().to(self.device)
                pred_batch = self.model(tiles_batch)
                self.integrate_batch(pred_batch, coords_batch, merger_dict)
    
    def integrate_batch(self, pred_batch, coords_batch, merger_dict):
        merger_dict['seg'].integrate_batch(pred_batch, coords_batch)

    def merge_masks(self, tiler: ImageSlicer, merger_dict: Dict[str, TileMerger]):
        merged_mask = np.moveaxis(to_numpy(merger_dict['seg'].merge()), 0, -1)
        return {
            'seg': tiler.crop_to_orignal_size(merged_mask)
        }

class SingleImageFromFrameFieldProcessor(SingleImageInfereceProcessor):
    def __init__(self, model, device, batch_size, export_strategy, polygonizer=None, model_input_shape=None, step_shape=None, mask_bands=1, config=None):
        super(SingleImageFromFrameFieldProcessor, self).__init__(
            model, device, batch_size, export_strategy=export_strategy, polygonizer=polygonizer, model_input_shape=model_input_shape,\
            step_shape=step_shape, mask_bands=mask_bands, config=config
        )
    
    def get_merger_dict(self, tiler: ImageSlicer):
        return {
            'seg': TileMerger(tiler.target_shape, self.mask_bands, tiler.weight, device=self.device),
            'crossfield': TileMerger(tiler.target_shape, 4, tiler.weight, device=self.device),
        }
    
    def integrate_batch(self, pred_batch, coords_batch, merger_dict):
        for key in ['seg', 'crossfield']:
            merger_dict[key].integrate_batch(pred_batch[key], coords_batch)

    def merge_masks(self, tiler: ImageSlicer, merger_dict: Dict[str, TileMerger]):
        return {
            key: np.moveaxis(to_numpy(merger_dict[key].merge()), 0, -1) \
                for key in ['seg', 'crossfield']
        }
