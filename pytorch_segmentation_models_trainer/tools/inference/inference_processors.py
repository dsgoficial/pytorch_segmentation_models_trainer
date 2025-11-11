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
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
import os
import math
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image
from pytorch_segmentation_models_trainer.tools.detection.bbox_handler import (
    BboxTileMerger,
)
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import (
    PolygonRNNPolygonizerProcessor,
    TemplatePolygonizerProcessor,
)
from typing import Any, Dict, List, Optional, Union

import albumentations as A
import cv2
from torch.cuda.amp import autocast
import numpy as np
import rasterio
import torch
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from pytorch_toolbelt.utils.torch_utils import (
    image_to_tensor,
    tensor_from_rgb_image,
    to_numpy,
)
from torch.utils.data import DataLoader


class AbstractInferenceProcessor(ABC):
    """
    Abstract method to process inferences
    """

    def __init__(
        self,
        model,
        device,
        batch_size,
        export_strategy,
        polygonizer=None,
        model_input_shape=None,
        step_shape=None,
        mask_bands=1,
        normalize_mean=None,
        normalize_std=None,
        config=None,
        group_output_by_image_basename=False,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.export_strategy = export_strategy
        self.polygonizer = polygonizer
        self.config = config
        self.model.to(device)
        self.model_input_shape = (
            (448, 448) if model_input_shape is None else model_input_shape
        )
        self.step_shape = (224, 224) if step_shape is None else step_shape
        self.mask_bands = mask_bands
        self.group_output_by_image_basename = group_output_by_image_basename
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def read_image_and_profile(self, image_path, restore_geo_transform=True):
        with rasterio.open(image_path, "r") as raster_ds:
            image = raster_ds.read()
            image = np.moveaxis(image, 0, -1)
            profile = raster_ds.profile
        if not restore_geo_transform:
            profile["crs"] = None
        return image, profile
    
    def get_normalization_function(self):
        if self.normalize_mean is not None and self.normalize_std is not None:
            func = A.Normalize(mean=self.normalize_mean, std=self.normalize_std, p=1.0)
        else:
            # Padrão ImageNet RGB
            func = A.Normalize()
        return func

    def process(
        self,
        image_path: str,
        threshold: float = 0.5,
        save_inference_output: bool = True,
        polygonizer: Optional[TemplatePolygonizerProcessor] = None,
        restore_geo_transform: bool = True,
        **kwargs: Optional[Any],
    ) -> Dict[str, Any]:
        image, profile = self.read_image_and_profile(
            image_path, restore_geo_transform=restore_geo_transform
        )
        inference = self.make_inference(image, **kwargs)
        output_dict = defaultdict(list)
        polygonizer = self.polygonizer if polygonizer is None else polygonizer
        if polygonizer is not None:
            output_dict["polygons"] += self.process_polygonizer(
                polygonizer,
                inference,
                profile,
                image_name=Path(image_path).stem
                if self.group_output_by_image_basename
                else None,
            )
        if save_inference_output:
            self.save_inference(image_path, threshold, profile, inference, output_dict)
        if "output_inferences" in kwargs and kwargs["output_inferences"]:
            output_dict["inference_output"].append(inference)
        return output_dict

    def process_polygonizer(self, polygonizer, inference, profile, image_name):
        return polygonizer.process(
            {
                key: image_to_tensor(value).unsqueeze(0)
                for key, value in inference.items()
            },
            profile,
            parent_dir_name=image_name,
        )

    def save_inference(self, image_path, threshold, profile, inference, output_dict, apply_threshold=True):
        inference["seg"] = (inference["seg"] > threshold).astype(np.uint8) if apply_threshold \
            else inference["seg"].astype(np.uint8)
        profile["input_name"] = Path(image_path).stem
        profile["suffix"] = Path(image_path).suffix
        if self.export_strategy is not None:
            output_dict["inference"].append(
                self.export_strategy.save_inference(inference, profile)
            )

    @abstractmethod
    def make_inference(
        self, image: np.ndarray, **kwargs: Optional[Any]
    ) -> Union[np.ndarray, dict]:
        """Makes model inference. Must be reimplemented in children classes

        Args:
            image (np.array): image to run inference

        Returns:
            torch.Tensor: model inference
        """
        pass


class SingleImageInfereceProcessor(AbstractInferenceProcessor):
    def __init__(
        self,
        model,
        device,
        batch_size,
        export_strategy,
        polygonizer=None,
        model_input_shape=None,
        step_shape=None,
        mask_bands=1,
        normalize_mean=None,
        normalize_std=None,
        config=None,
        group_output_by_image_basename=False,
    ):
        super(SingleImageInfereceProcessor, self).__init__(
            model,
            device,
            batch_size,
            export_strategy=export_strategy,
            polygonizer=polygonizer,
            model_input_shape=model_input_shape,
            step_shape=step_shape,
            mask_bands=mask_bands,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            config=config,
            group_output_by_image_basename=group_output_by_image_basename,
        )

    def make_inference(
        self, image: np.ndarray, **kwargs: Optional[Any]
    ) -> Union[np.ndarray, dict]:
        """Makes model inference.

        Args:
            image (np.array): image to run inference

        Returns:
            Union[np.array, dict]: model inference
        """
        norm_func = self.get_normalization_function()
        normalized_image = norm_func(image=image)["image"]
        pad_func = A.PadIfNeeded(
            math.ceil(image.shape[0] / self.model_input_shape[0])
            * self.model_input_shape[0],
            math.ceil(image.shape[1] / self.model_input_shape[1])
            * self.model_input_shape[1],
            border_mode=cv2.BORDER_REFLECT_101,
            value=None,
        )
        center_crop_func = A.CenterCrop(*image.shape[0:2])
        normalized_image = pad_func(image=normalized_image)["image"]
        tiler = ImageSlicer(
            normalized_image.shape,
            tile_size=self.model_input_shape,
            tile_step=self.step_shape,
        )
        tiles = [image_to_tensor(tile) for tile in tiler.split(normalized_image)]
        merger_dict = self.get_merger_dict(tiler)
        self.predict_and_merge(tiles, tiler, merger_dict)
        merged_masks_dict = self.merge_masks(tiler, merger_dict)
        return {
            key: center_crop_func(image=value)["image"]
            for key, value in merged_masks_dict.items()
        }

    def get_merger_dict(self, tiler: ImageSlicer):
        return {
            "seg": TileMerger(
                tiler.target_shape, self.mask_bands, tiler.weight, device=self.device
            )
        }

    def predict_and_merge(
        self,
        tiles: List[np.array],
        tiler: ImageSlicer,
        merger_dict: Dict[str, TileMerger],
    ):
        with torch.no_grad():
            for tiles_batch, coords_batch in DataLoader(
                list(zip(tiles, tiler.crops)),
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=4,
                persistent_workers=True,
                prefetch_factor=2,
            ):
                tiles_batch = tiles_batch.float().to(self.device)
                with autocast():
                    pred_batch = self.model(tiles_batch)
                self.integrate_batch(pred_batch, coords_batch, merger_dict)

    def integrate_batch(self, pred_batch, coords_batch, merger_dict):
        merger_dict["seg"].integrate_batch(pred_batch, coords_batch)

    def merge_masks(self, tiler: ImageSlicer, merger_dict: Dict[str, TileMerger]):
        merged_mask = np.moveaxis(to_numpy(merger_dict["seg"].merge()), 0, -1)
        return {"seg": tiler.crop_to_orignal_size(merged_mask)}

class MultiClassInferenceProcessor(SingleImageInfereceProcessor):
    """
    Inference processor para segmentação multi-class.
    Modelo deve retornar logits/probabilidades: [B, num_classes, H, W]
    Resultado final será uma imagem de 1 banda com índices de classe: [H, W]
    """
    
    def __init__(
        self,
        model,
        device,
        batch_size,
        export_strategy,
        polygonizer=None,
        model_input_shape=None,
        step_shape=None,
        num_classes=2,  # ✅ Novo parâmetro!
        normalize_mean=None,
        normalize_std=None,
        config=None,
        group_output_by_image_basename=False,
    ):
        # ✅ Passa num_classes como mask_bands para TileMerger
        super().__init__(
            model=model,
            device=device,
            batch_size=batch_size,
            export_strategy=export_strategy,
            polygonizer=polygonizer,
            model_input_shape=model_input_shape,
            step_shape=step_shape,
            mask_bands=num_classes,  # ✅ TileMerger precisa de todos os canais
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            config=config,
            group_output_by_image_basename=group_output_by_image_basename,
        )
        self.num_classes = num_classes
    
    def merge_masks(self, tiler: ImageSlicer, merger_dict: Dict[str, TileMerger]):
        """
        ✅ Sobrescreve para fazer argmax após merge!
        """
        # Merge ponderado das probabilidades (mantém todas as classes)
        merged_probs = to_numpy(merger_dict["seg"].merge())  # [num_classes, H, W]
        
        # ✅ ARGMAX: converte para índices de classe
        class_indices = np.argmax(merged_probs, axis=0)  # [H, W]
        
        # Crop para tamanho original
        class_indices = tiler.crop_to_orignal_size(class_indices)
        if class_indices.ndim == 2:
            class_indices = class_indices[..., np.newaxis]
        
        # ✅ Retorna 1 banda com índices (0, 1, 2, ..., num_classes-1)
        return {"seg": class_indices}
    
    def save_inference(self, image_path, threshold, profile, inference, output_dict, apply_threshold=True):
        """
        ✅ Salva imagem de 1 banda com índices de classe (sem threshold!)
        """
        # inference["seg"] já é [H, W] com valores 0, 1, 2, ..., num_classes-1
        
        # ✅ Atualiza profile para 1 banda, dtype uint8
        profile["count"] = 1
        profile["dtype"] = 'uint8'
        profile["input_name"] = Path(image_path).stem
        profile["suffix"] = Path(image_path).suffix
        
        if self.export_strategy is not None:
            output_dict["inference"].append(
                self.export_strategy.save_inference(inference, profile)
            )


class SingleImageFromFrameFieldProcessor(SingleImageInfereceProcessor):
    def __init__(
        self,
        model,
        device,
        batch_size,
        export_strategy,
        polygonizer=None,
        model_input_shape=None,
        step_shape=None,
        mask_bands=1,
        normalize_mean=None,
        normalize_std=None,
        config=None,
        group_output_by_image_basename=False,
    ):
        super(SingleImageFromFrameFieldProcessor, self).__init__(
            model,
            device,
            batch_size,
            export_strategy=export_strategy,
            polygonizer=polygonizer,
            model_input_shape=model_input_shape,
            step_shape=step_shape,
            mask_bands=mask_bands,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            config=config,
            group_output_by_image_basename=group_output_by_image_basename,
        )

    def get_merger_dict(self, tiler: ImageSlicer):
        return {
            "seg": TileMerger(
                tiler.target_shape, self.mask_bands, tiler.weight, device=self.device
            ),
            "crossfield": TileMerger(
                tiler.target_shape, 4, tiler.weight, device=self.device
            ),
        }

    def integrate_batch(self, pred_batch, coords_batch, merger_dict):
        for key in ["seg", "crossfield"]:
            merger_dict[key].integrate_batch(pred_batch[key], coords_batch)

    def merge_masks(self, tiler: ImageSlicer, merger_dict: Dict[str, TileMerger]):
        return {
            key: np.moveaxis(to_numpy(merger_dict[key].merge()), 0, -1)
            for key in ["seg", "crossfield"]
        }


class ObjectDetectionInferenceProcessor(AbstractInferenceProcessor):
    def __init__(
        self,
        model,
        device,
        batch_size,
        export_strategy,
        model_input_shape=None,
        step_shape=None,
        mask_bands=1,
        post_process_method=None,
        min_visibility=0.3,
        normalize_mean=None,
        normalize_std=None,
        config=None,
        group_output_by_image_basename=False,
    ):
        super(ObjectDetectionInferenceProcessor, self).__init__(
            model,
            device,
            batch_size,
            export_strategy=export_strategy,
            polygonizer=None,
            model_input_shape=model_input_shape,
            step_shape=step_shape,
            mask_bands=mask_bands,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            config=config,
            group_output_by_image_basename=group_output_by_image_basename,
        )
        self.post_process_method = (
            post_process_method if post_process_method is not None else "union"
        )
        self.min_visibility = min_visibility

    def process(
        self,
        image_path: str,
        threshold: float = 0.5,
        save_inference_output: bool = True,
        polygonizer: Optional[TemplatePolygonizerProcessor] = None,
        restore_geo_transform: bool = True,
        **kwargs: Optional[Any],
    ) -> Dict[str, Any]:
        kwargs.update({"output_inferences": True})
        return super(ObjectDetectionInferenceProcessor, self).process(
            image_path,
            threshold,
            save_inference_output,
            polygonizer,
            restore_geo_transform,
            **kwargs,
        )

    def make_inference(
        self, image: np.ndarray, **kwargs: Optional[Any]
    ) -> List[Dict[str, torch.Tensor]]:
        norm_func = self.get_normalization_function()
        normalized_image = norm_func(image=image)["image"]
        pad_func = A.PadIfNeeded(
            math.ceil(image.shape[0] / self.model_input_shape[0])
            * self.model_input_shape[0],
            math.ceil(image.shape[1] / self.model_input_shape[1])
            * self.model_input_shape[1],
        )
        normalized_image = pad_func(image=normalized_image)["image"]
        tiler = ImageSlicer(
            normalized_image.shape,
            tile_size=self.model_input_shape,
            tile_step=self.step_shape,
        )
        tiles = [image_to_tensor(tile) for tile in tiler.split(normalized_image)]
        merger = BboxTileMerger(
            image_shape=image.shape,
            post_process_method=self.post_process_method,
            device=self.device,
        )
        self.predict_and_integrate_batches(tiles, tiler, merger)
        bboxes = merger.merge()
        return bboxes

    def predict_and_integrate_batches(
        self, tiles: List[np.array], tiler: ImageSlicer, merger: BboxTileMerger
    ):
        with torch.no_grad():
            for tiles_batch, crop_coords_batch in DataLoader(
                list(zip(tiles, tiler.crops)),
                batch_size=self.batch_size,
                pin_memory=True,
            ):
                tiles_batch = tiles_batch.float().to(self.device)
                pred_batch = self.model(tiles_batch)
                merger.integrate_boxes(pred_batch, crop_coords_batch)

    def save_inference(self, image_path, threshold, profile, inference, output_dict, apply_threshold=True):
        inference = [
            {k: v.cpu().tolist() for k, v in item.items()} for item in inference
        ]
        profile["input_name"] = Path(image_path).stem
        profile["suffix"] = Path(image_path).suffix
        output_dict["inference"].append(
            self.export_strategy.save_inference(inference, profile)
        )


class PolygonRNNInferenceProcessor(AbstractInferenceProcessor):
    def __init__(
        self,
        model,
        device,
        batch_size,
        polygonizer=None,
        normalize_mean=None,
        normalize_std=None,
        config=None,
        sequence_length=60,
        group_output_by_image_basename=False,
    ):
        super(PolygonRNNInferenceProcessor, self).__init__(
            model,
            device,
            batch_size,
            export_strategy=None,
            polygonizer=polygonizer,
            model_input_shape=(224, 224),
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            config=config,
            group_output_by_image_basename=group_output_by_image_basename,
        )
        self.image_size = 224
        self.sequence_length = sequence_length

    def process(
        self,
        image_path: str,
        threshold: float = 0.5,
        save_inference_output: bool = True,
        polygonizer: Optional[TemplatePolygonizerProcessor] = None,
        restore_geo_transform: bool = True,
        **kwargs: Optional[Any],
    ) -> Dict[str, Any]:
        save_inference_output = False
        return super(PolygonRNNInferenceProcessor, self).process(
            image_path,
            threshold,
            save_inference_output,
            polygonizer,
            restore_geo_transform,
            **kwargs,
        )

    def make_inference(
        self, image: np.ndarray, bboxes: List[np.ndarray], **kwargs: Optional[Any]
    ) -> Dict[str, np.ndarray]:
        img = Image.fromarray(image)
        image_tensor_list_dict: List[
            Dict[str, torch.Tensor]
        ] = self.crop_and_resize_image_to_bboxes(img, bboxes)
        output_list = []
        self.model.eval()
        with torch.no_grad():
            for batch in DataLoader(
                image_tensor_list_dict, batch_size=self.batch_size, pin_memory=True
            ):
                output_batch_polygons = self.model.test(
                    batch["croped_images"].to(self.device), self.sequence_length
                )
                batch.pop("croped_images")
                batch["output_batch_polygons"] = output_batch_polygons
                output_list.append(batch)
        return {
            key: torch.concat([batch[key] for batch in output_list])
            for key in output_list[0].keys()
        }

    def crop_and_resize_image_to_bboxes(
        self, image: Image, bboxes: List[np.ndarray]
    ) -> List[Dict[str, torch.Tensor]]:
        output_list = []
        norm_func = self.get_normalization_function()
        for box in bboxes:
            croped_image = image.crop(box=box)
            image_tile = croped_image.resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
            scale_h, scale_w = self._get_scales(*box)
            output_list.append(
                {
                    "croped_images": torch.from_numpy(
                        norm_func(image=np.array(image_tile))["image"]
                    ),
                    "scale_h": torch.tensor(scale_h),
                    "scale_w": torch.tensor(scale_w),
                    "min_row": torch.tensor(box[0]),
                    "min_col": torch.tensor(box[1]),
                }
            )
        return output_list

    def _get_scales(
        self, min_row: int, min_col: int, max_row: int, max_col: int
    ) -> tuple:
        """
        Gets scales for the image.

        Args:
            min_row (int): min row
            min_col (int): min col
            max_row (int): max row
            max_col (int): max col

        Returns:
            tuple: scale_h, scale_w
        """
        object_h = max_row - min_row
        object_w = max_col - min_col
        scale_h = self.image_size / object_h
        scale_w = self.image_size / object_w
        return scale_h, scale_w

    def process_polygonizer(self, polygonizer, inference, profile, image_name=None):
        return self.polygonizer.process(inference, profile)
