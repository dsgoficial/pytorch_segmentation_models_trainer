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
import logging
import os
import math
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image
from pytorch_segmentation_models_trainer.tools.detection.bbox_handler import (
    BboxTileMerger,
)
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import (
    TemplatePolygonizerProcessor,
)
from pytorch_segmentation_models_trainer.tools.tta.tta import (
    ROTATION_AUGMENTATIONS as _DEFAULT_TTA_AUGMENTATIONS,
    apply_tta,
)
from typing import Any, Dict, FrozenSet, List, Optional, Union

import albumentations as A
import cv2
from torch.cuda.amp import autocast
import numpy as np
import rasterio
import rasterio.windows
import torch
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from pytorch_toolbelt.utils.torch_utils import (
    image_to_tensor,
    to_numpy,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


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
        normalize_max_value: Optional[float] = None,
        config=None,
        group_output_by_image_basename=False,
        use_tta: bool = False,
        tta_augmentations: Optional[List[str]] = None,
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
        self.normalize_max_value = normalize_max_value
        self.use_tta = use_tta
        self.tta_augmentations = (
            tta_augmentations
            if tta_augmentations is not None
            else list(_DEFAULT_TTA_AUGMENTATIONS)
        )
        # Keys in model output that cannot be spatially de-augmented.
        # Subclasses override this when needed (e.g. frame-field crossfield).
        self._tta_skip_keys: FrozenSet[str] = frozenset()

    def read_image_and_profile(self, image_path, restore_geo_transform=True):
        with rasterio.open(image_path, "r") as raster_ds:
            image = raster_ds.read()
            image = np.moveaxis(image, 0, -1)
            profile = raster_ds.profile
        if not restore_geo_transform:
            profile["crs"] = None
        return image, profile

    def get_normalization_function(self):
        kwargs = {}
        if self.normalize_mean is not None and self.normalize_std is not None:
            kwargs["mean"] = self.normalize_mean
            kwargs["std"] = self.normalize_std
        if self.normalize_max_value is not None:
            kwargs["max_pixel_value"] = self.normalize_max_value
        return A.Normalize(p=1.0, **kwargs)

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

    def save_inference(
        self,
        image_path,
        threshold,
        profile,
        inference,
        output_dict,
        apply_threshold=True,
    ):
        inference["seg"] = (
            (inference["seg"] > threshold).astype(np.uint8)
            if apply_threshold
            else inference["seg"].astype(np.uint8)
        )
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
    # Map tta_mode strings to tta_augmentations lists (same interface as MultiClass)
    _TTA_MODE_AUGMENTATIONS = {
        "d4": [
            "rot0",
            "rot90",
            "rot180",
            "rot270",
            "flip_h",
            "flip_v",
            "flip_h_rot90",
            "flip_v_rot90",
        ],
        "flip": ["rot0", "flip_h", "rot180", "flip_v"],
    }

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
        normalize_max_value: Optional[float] = None,
        config=None,
        group_output_by_image_basename=False,
        use_tta: bool = False,
        tta_augmentations: Optional[List[str]] = None,
        tile_weight="mean",
        tta_mode: Optional[str] = None,
    ):
        # tta_mode is a convenience alias: "d4" or "flip" automatically sets
        # use_tta=True and selects the matching tta_augmentations preset.
        if tta_mode is not None:
            if tta_mode not in self._TTA_MODE_AUGMENTATIONS:
                raise ValueError(
                    f"tta_mode must be None, 'd4', or 'flip', got '{tta_mode}'"
                )
            use_tta = True
            tta_augmentations = self._TTA_MODE_AUGMENTATIONS[tta_mode]
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
            normalize_max_value=normalize_max_value,
            config=config,
            group_output_by_image_basename=group_output_by_image_basename,
            use_tta=use_tta,
            tta_augmentations=tta_augmentations,
        )
        self._tile_weight = self._resolve_tile_weight(tile_weight)

    @staticmethod
    def _gaussian_importance_map(patch_size, sigma_scale=0.125):
        """nnU-Net-style 2D Gaussian importance map.

        Pixels near the tile center get weight ~1.0, edges get near-zero.
        This reduces border artifacts from sliding window inference because
        CNN predictions degrade near borders due to padding/context truncation.

        Args:
            patch_size: (H, W) tuple.
            sigma_scale: fraction of patch size used as sigma (default 1/8).
        """
        center = [(s - 1) / 2.0 for s in patch_size]
        sigma = [max(s * sigma_scale, 1.0) for s in patch_size]

        axes = [np.arange(s, dtype=np.float32) for s in patch_size]
        grid = np.meshgrid(*axes, indexing="ij")

        gaussian = np.ones(patch_size, dtype=np.float32)
        for ax, c, s in zip(grid, center, sigma):
            gaussian *= np.exp(-0.5 * ((ax - c) / s) ** 2)

        # Numerical stability — floor at 1e-4 of peak
        gaussian = np.maximum(gaussian, gaussian.max() * 1e-4)
        return gaussian

    def _resolve_tile_weight(self, tile_weight):
        """Resolve tile_weight to a value accepted by ImageSlicer.

        Args:
            tile_weight: "mean" (uniform), "pyramid" (distance-based),
                "gaussian" (nnU-Net-style Gaussian), or a numpy array.

        Returns:
            str or np.ndarray passed to ImageSlicer(weight=...).
        """
        if isinstance(tile_weight, np.ndarray):
            return tile_weight
        if tile_weight in ("mean", "pyramid"):
            return tile_weight
        if tile_weight == "gaussian":
            return self._gaussian_importance_map(tuple(self.model_input_shape))
        raise ValueError(
            f"tile_weight must be 'mean', 'pyramid', 'gaussian', or "
            f"numpy array, got '{tile_weight}'"
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
            weight=self._tile_weight,
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
                num_workers=0,
            ):
                tiles_batch = tiles_batch.float().to(self.device)
                if self.use_tta:

                    def _model_fn(x):
                        with autocast():
                            return self.model(x)

                    pred_batch = apply_tta(
                        model_fn=_model_fn,
                        batch=tiles_batch,
                        augmentations=self.tta_augmentations,
                        skip_keys=self._tta_skip_keys,
                    )
                else:
                    with autocast():
                        pred_batch = self.model(tiles_batch)
                self.integrate_batch(pred_batch, coords_batch, merger_dict)
                del tiles_batch, pred_batch

    def integrate_batch(self, pred_batch, coords_batch, merger_dict):
        merger_dict["seg"].integrate_batch(pred_batch, coords_batch)

    def merge_masks(self, tiler: ImageSlicer, merger_dict: Dict[str, TileMerger]):
        merged_mask = np.moveaxis(to_numpy(merger_dict["seg"].merge()), 0, -1)
        return {"seg": tiler.crop_to_orignal_size(merged_mask)}


class MultiClassInferenceProcessor(SingleImageInfereceProcessor):
    """
    Inference processor for multi-class segmentation.
    Model must return logits/probabilities: [B, num_classes, H, W]
    Final result is a single-band image with class indices: [H, W]

    TTA (Test-Time Augmentation) modes:
        None  — no TTA (default, backward compatible)
        "d4"  — all 8 dihedral symmetries (4 rotations x 2 flips)
        "flip" — horizontal + vertical flip (4 passes)
    """

    # D4 group: (num_rot90, flip_horizontal)
    _D4_TRANSFORMS = [
        (0, False),
        (1, False),
        (2, False),
        (3, False),
        (0, True),
        (1, True),
        (2, True),
        (3, True),
    ]
    _FLIP_TRANSFORMS = [
        (0, False),
        (0, True),  # identity, hflip
        (2, False),
        (2, True),  # rot180, rot180+hflip (= vflip)
    ]

    def __init__(
        self,
        model,
        device,
        batch_size,
        export_strategy,
        polygonizer=None,
        model_input_shape=None,
        step_shape=None,
        num_classes=2,
        normalize_mean=None,
        normalize_std=None,
        normalize_max_value: Optional[float] = None,
        config=None,
        group_output_by_image_basename=False,
        tta_mode=None,
        tile_weight="mean",
        confidence_mode=None,
        confidence_export_strategy=None,
        striped_threshold_pixels=50_000_000,
        stripe_height=4096,
        output_probs_dir=None,
    ):
        super().__init__(
            model=model,
            device=device,
            batch_size=batch_size,
            export_strategy=export_strategy,
            polygonizer=polygonizer,
            model_input_shape=model_input_shape,
            step_shape=step_shape,
            mask_bands=num_classes,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            normalize_max_value=normalize_max_value,
            config=config,
            group_output_by_image_basename=group_output_by_image_basename,
            tile_weight=tile_weight,
        )
        self.num_classes = num_classes
        if tta_mode is not None and tta_mode not in ("d4", "flip"):
            raise ValueError(
                f"tta_mode must be None, 'd4', or 'flip', got '{tta_mode}'"
            )
        self.tta_mode = tta_mode
        self.striped_threshold_pixels = striped_threshold_pixels
        self.stripe_height = stripe_height
        self.output_probs_dir = output_probs_dir
        if output_probs_dir is not None:
            os.makedirs(output_probs_dir, exist_ok=True)
        if confidence_mode is not None and confidence_mode not in ("basic", "full"):
            raise ValueError(
                f"confidence_mode must be None, 'basic', or 'full', got '{confidence_mode}'"
            )
        self.confidence_mode = confidence_mode
        self.confidence_export_strategy = confidence_export_strategy

    # ---- TTA helpers (operate on torch tensors [B, C, H, W]) ----

    @staticmethod
    def _apply_transform(tensor, num_rot90, flip_h):
        """Apply a D4 transform to a batch tensor [B, C, H, W]."""
        if flip_h:
            tensor = torch.flip(tensor, dims=[-1])
        if num_rot90:
            tensor = torch.rot90(tensor, k=num_rot90, dims=[-2, -1])
        return tensor

    @staticmethod
    def _invert_transform(tensor, num_rot90, flip_h):
        """Invert a D4 transform on a batch tensor [B, C, H, W]."""
        if num_rot90:
            tensor = torch.rot90(tensor, k=-num_rot90, dims=[-2, -1])
        if flip_h:
            tensor = torch.flip(tensor, dims=[-1])
        return tensor

    def _get_tta_transforms(self):
        if self.tta_mode == "d4":
            return self._D4_TRANSFORMS
        elif self.tta_mode == "flip":
            return self._FLIP_TRANSFORMS
        return [(0, False)]

    def predict_and_merge(
        self,
        tiles: List[np.array],
        tiler: ImageSlicer,
        merger_dict: Dict[str, TileMerger],
    ):
        """Predict with optional TTA — average logits across transforms."""
        transforms = self._get_tta_transforms()
        n_tta = len(transforms)

        with torch.no_grad():
            for tiles_batch, coords_batch in DataLoader(
                list(zip(tiles, tiler.crops)),
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=0,
            ):
                tiles_batch = tiles_batch.float().to(self.device)

                if n_tta <= 1:
                    # No TTA — fast path (original behavior)
                    with autocast():
                        pred_batch = self.model(tiles_batch)
                else:
                    # TTA: accumulate logits from all transforms
                    accum = None
                    for rot, flip in transforms:
                        aug = self._apply_transform(tiles_batch, rot, flip)
                        with autocast():
                            pred = self.model(aug)  # [B, C, H, W]
                        del aug
                        pred = pred.float()
                        pred = self._invert_transform(pred, rot, flip)
                        accum = pred if accum is None else accum + pred
                        del pred
                    pred_batch = accum / n_tta
                    del accum

                self.integrate_batch(pred_batch, coords_batch, merger_dict)
                del tiles_batch, pred_batch

    def merge_masks(self, tiler: ImageSlicer, merger_dict: Dict[str, TileMerger]):
        """Argmax after weighted tile merge."""
        merged_probs = to_numpy(merger_dict["seg"].merge())  # [num_classes, H, W]
        class_indices = np.argmax(merged_probs, axis=0)  # [H, W]
        class_indices = tiler.crop_to_orignal_size(class_indices)
        if class_indices.ndim == 2:
            class_indices = class_indices[..., np.newaxis]
        return {"seg": class_indices}

    def make_inference_with_probs(self, image):
        """Run inference returning class predictions, softmax probabilities,
        and per-pixel confidence.

        Unlike ``make_inference()``, this returns the full probability maps so
        that callers can apply confidence-based filtering (e.g. pseudo-labeling).

        Args:
            image: np.ndarray [H, W, C] — input image (uint8 or float).

        Returns:
            dict with keys:
                ``seg``        : np.ndarray [H, W]     uint8   — class indices
                ``probs``      : np.ndarray [C, H, W]  float32 — softmax probs
                ``confidence`` : np.ndarray [H, W]     float32 — max prob / pixel
        """
        norm_func = self.get_normalization_function()
        normalized_image = norm_func(image=image)["image"]
        orig_h, orig_w = image.shape[:2]

        pad_func = A.PadIfNeeded(
            math.ceil(orig_h / self.model_input_shape[0]) * self.model_input_shape[0],
            math.ceil(orig_w / self.model_input_shape[1]) * self.model_input_shape[1],
            border_mode=cv2.BORDER_REFLECT_101,
            value=None,
        )
        center_crop = A.CenterCrop(orig_h, orig_w)
        normalized_image = pad_func(image=normalized_image)["image"]

        tiler = ImageSlicer(
            normalized_image.shape,
            tile_size=self.model_input_shape,
            tile_step=self.step_shape,
            weight=self._tile_weight,
        )
        tiles = [image_to_tensor(tile) for tile in tiler.split(normalized_image)]
        merger_dict = self.get_merger_dict(tiler)
        self.predict_and_merge(tiles, tiler, merger_dict)

        # Merged logits (before argmax)
        merged_logits = to_numpy(merger_dict["seg"].merge())  # [C, H_pad, W_pad]

        # Numerically stable softmax — in-place to save memory
        merged_logits -= merged_logits.max(axis=0, keepdims=True)
        np.exp(merged_logits, out=merged_logits)
        merged_logits /= merged_logits.sum(axis=0, keepdims=True)
        # merged_logits is now probs [C, H_pad, W_pad]

        class_indices = np.argmax(merged_logits, axis=0)  # [H_pad, W_pad]
        confidence = merged_logits.max(axis=0)  # [H_pad, W_pad]

        # CenterCrop back to original dimensions (mirrors parent make_inference)
        class_indices = center_crop(image=class_indices[..., np.newaxis])[
            "image"
        ].squeeze(-1)
        confidence = center_crop(image=confidence[..., np.newaxis])["image"].squeeze(-1)
        probs_hwc = np.moveaxis(merged_logits, 0, -1)  # [H_pad, W_pad, C]
        probs_hwc = center_crop(image=probs_hwc)["image"]

        return {
            "seg": class_indices.astype(np.uint8),
            "probs": np.moveaxis(probs_hwc, -1, 0).astype(np.float32),
            "confidence": confidence.astype(np.float32),
        }

    @staticmethod
    def _compute_confidence_maps(probs):
        """Compute confidence metrics from softmax probabilities.

        Args:
            probs: np.ndarray [C, H, W] — softmax probabilities.

        Returns:
            dict with keys ``max_prob``, ``entropy``, ``margin``
            (each np.ndarray [H, W] float32, range [0, 1]).
        """
        max_prob = probs.max(axis=0)  # [H, W]

        # Normalized entropy (0 = full certainty, 1 = max uncertainty)
        eps = 1e-10
        log_probs = np.log(probs + eps)
        entropy = -(probs * log_probs).sum(axis=0)  # [H, W]
        max_entropy = np.log(probs.shape[0])  # log(num_classes)
        entropy_norm = entropy / max_entropy  # [0, 1]

        # Margin between top-2 classes (0 = undecided, 1 = certain)
        top2 = np.partition(probs, -2, axis=0)[-2:]  # top-2 along class axis
        margin = top2.max(axis=0) - top2.min(axis=0)  # [H, W]

        return {
            "max_prob": max_prob.astype(np.float32),
            "entropy": entropy_norm.astype(np.float32),
            "margin": margin.astype(np.float32),
        }

    def _is_large_image(self, image_path):
        """Check if image exceeds striped_threshold_pixels."""
        with rasterio.open(image_path) as ds:
            return ds.height * ds.width > self.striped_threshold_pixels

    def process(
        self,
        image_path,
        threshold=0.5,
        save_inference_output=True,
        polygonizer=None,
        restore_geo_transform=True,
        **kwargs,
    ):
        """Process a single image, optionally computing confidence maps.

        Automatically uses striped inference for large images to avoid OOM.
        """
        # Auto-detect large images and use striped inference
        if self._is_large_image(image_path):
            output_dir = (
                self.export_strategy.output_file_path if self.export_strategy else "."
            )
            os.makedirs(output_dir, exist_ok=True)
            stem = Path(image_path).stem
            output_seg = os.path.join(output_dir, f"{stem}.tif")
            output_probs = None
            if self.output_probs_dir is not None:
                output_probs = os.path.join(self.output_probs_dir, f"{stem}.tif")
            elif self.confidence_mode is not None:
                probs_dir = os.path.join(output_dir, "probs")
                os.makedirs(probs_dir, exist_ok=True)
                output_probs = os.path.join(probs_dir, f"{stem}.tif")
            logger.info(f"Large image detected, using striped inference: {image_path}")
            self.make_inference_striped(
                image_path,
                output_seg,
                output_probs_path=output_probs,
                stripe_height=self.stripe_height,
            )
            return defaultdict(list)

        image, profile = self.read_image_and_profile(
            image_path, restore_geo_transform=restore_geo_transform
        )

        if self.confidence_mode is not None:
            result = self.make_inference_with_probs(image)
            inference = {"seg": result["seg"]}
            confidence_maps = self._compute_confidence_maps(result["probs"])
        else:
            inference = self.make_inference(image, **kwargs)
            confidence_maps = None

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
            if confidence_maps is not None:
                self._save_confidence_maps(image_path, profile, confidence_maps)
        if "output_inferences" in kwargs and kwargs["output_inferences"]:
            output_dict["inference_output"].append(inference)
        return output_dict

    def _save_confidence_maps(self, image_path, profile, confidence_maps):
        """Save confidence metrics as individual float32 GeoTIFFs."""
        if self.confidence_export_strategy is None:
            return

        conf_profile = dict(profile)
        conf_profile["dtype"] = "float32"
        conf_profile["count"] = 1
        conf_profile["input_name"] = Path(image_path).stem
        conf_profile["suffix"] = Path(image_path).suffix

        if self.confidence_mode == "basic":
            maps_to_save = {"max_prob": confidence_maps["max_prob"]}
        else:  # "full"
            maps_to_save = confidence_maps

        for metric_name, data in maps_to_save.items():
            self.confidence_export_strategy.save_confidence(
                metric_name, data[..., np.newaxis], dict(conf_profile)
            )

    # ---- Striped inference for large images (BigTIFF) ----

    def _predict_batch_to_merger(
        self, batch_tiles, batch_coords, merger, transforms, n_tta
    ):
        """Predict a batch with optional TTA and integrate into merger.

        Both the TTA accumulation and the merger integration happen on the
        same device as the merger (GPU when possible) to avoid costly
        GPU-to-CPU transfers every batch.
        """
        tiles_tensor = torch.stack(batch_tiles).float().to(self.device)
        coords_tensor = torch.from_numpy(np.array(batch_coords))
        merger_on_cpu = merger.weight.device.type == "cpu"

        if n_tta <= 1:
            with autocast():
                pred = self.model(tiles_tensor)
            pred = pred.float().cpu() if merger_on_cpu else pred.float()
        else:
            accum = None
            for rot, flip in transforms:
                aug = self._apply_transform(tiles_tensor, rot, flip)
                with autocast():
                    p = self.model(aug)
                del aug
                p = p.float().cpu() if merger_on_cpu else p.float()
                p = self._invert_transform(p, rot, flip)
                accum = p if accum is None else accum + p
                del p
            pred = accum / n_tta
            del accum

        del tiles_tensor
        merger.integrate_batch(pred, coords_tensor)
        del pred

    def _infer_stripe(self, stripe_normalized):
        """Process a single normalized stripe, returning merged logits on CPU.

        Uses iter_split() (generator) to avoid materializing all tiles.
        TileMerger runs on GPU to keep predictions on-device and avoid
        per-batch GPU-to-CPU transfers.  Falls back to CPU only when
        self.device is "cpu".

        Args:
            stripe_normalized: np.ndarray [h, W, C] — already normalized.

        Returns:
            np.ndarray [num_classes, h, W] — merged logits (CPU, float32).
        """
        h, w = stripe_normalized.shape[:2]

        pad_func = A.PadIfNeeded(
            math.ceil(h / self.model_input_shape[0]) * self.model_input_shape[0],
            math.ceil(w / self.model_input_shape[1]) * self.model_input_shape[1],
            border_mode=cv2.BORDER_REFLECT_101,
            position="top_left",
        )
        padded = pad_func(image=stripe_normalized)["image"]

        tiler = ImageSlicer(
            padded.shape,
            tile_size=self.model_input_shape,
            tile_step=self.step_shape,
            weight=self._tile_weight,
        )

        transforms = self._get_tta_transforms()
        n_tta = len(transforms)

        merger = TileMerger(
            tiler.target_shape, self.num_classes, tiler.weight, device=self.device
        )

        batch_tiles = []
        batch_coords = []

        with torch.no_grad():
            for tile, coords in tiler.iter_split(padded):
                batch_tiles.append(image_to_tensor(tile))
                batch_coords.append(coords)

                if len(batch_tiles) == self.batch_size:
                    self._predict_batch_to_merger(
                        batch_tiles, batch_coords, merger, transforms, n_tta
                    )
                    batch_tiles.clear()
                    batch_coords.clear()

            if batch_tiles:
                self._predict_batch_to_merger(
                    batch_tiles, batch_coords, merger, transforms, n_tta
                )

        # merged has shape [C, target_H, target_W] (includes ImageSlicer margins)
        merged = to_numpy(merger.merge())
        del merger
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Remove ImageSlicer margins, then PadIfNeeded padding (top_left → just crop to h,w)
        mt, ml = tiler.margin_top, tiler.margin_left
        cropped = merged[
            :,
            mt : mt + tiler.image_height,
            ml : ml + tiler.image_width,
        ]
        return cropped[:, :h, :w]

    def make_inference_striped(
        self,
        image_path,
        output_seg_path,
        output_probs_path=None,
        stripe_height=4096,
    ):
        """Streaming stripe-based inference for large/BigTIFF images.

        Reads the image in horizontal stripes via rasterio windowed reads,
        runs inference per stripe with TileMerger on CPU, and writes results
        directly to disk via rasterio windowed writes. Never loads the full
        image into RAM.

        Args:
            image_path: path to input GeoTIFF.
            output_seg_path: path for output segmentation GeoTIFF (uint8).
            output_probs_path: optional path for softmax probabilities
                (num_classes bands, float32). If None, probabilities are not saved.
            stripe_height: height in pixels of each processing stripe (default 4096).
        """
        src_ds = rasterio.open(image_path)
        H, W = src_ds.height, src_ds.width
        profile = src_ds.profile.copy()

        overlap = self.model_input_shape[0]
        n_stripes = math.ceil(H / stripe_height)
        logger.info(
            f"Striped inference: {H}x{W} image, {n_stripes} stripes "
            f"(height={stripe_height}, overlap={overlap})"
        )

        seg_profile = {
            **profile,
            "dtype": "uint8",
            "count": 1,
            "compress": "deflate",
            "nodata": 255,
            "BIGTIFF": "YES",
        }
        dst_seg = rasterio.open(output_seg_path, "w", **seg_profile)

        dst_probs = None
        if output_probs_path:
            probs_profile = {
                **profile,
                "dtype": "float32",
                "count": self.num_classes,
                "compress": "deflate",
                "BIGTIFF": "YES",
            }
            dst_probs = rasterio.open(output_probs_path, "w", **probs_profile)

        norm_func = self.get_normalization_function()

        try:
            y = 0
            stripe_idx = 0
            while y < H:
                y_end = min(y + stripe_height, H)
                read_y0 = max(0, y - overlap)
                read_y1 = min(H, y_end + overlap)

                logger.info(
                    f"  Stripe {stripe_idx + 1}/{n_stripes}: "
                    f"rows [{y}:{y_end}] (read [{read_y0}:{read_y1}])"
                )

                window_read = rasterio.windows.Window(
                    col_off=0, row_off=read_y0, width=W, height=read_y1 - read_y0
                )
                stripe = src_ds.read(window=window_read)
                stripe = np.moveaxis(stripe, 0, -1)

                stripe_norm = norm_func(image=stripe)["image"]
                del stripe

                merged_logits = self._infer_stripe(stripe_norm)
                del stripe_norm

                crop_top = y - read_y0
                logits_cropped = merged_logits[
                    :, crop_top : crop_top + (y_end - y), :
                ].copy()
                del merged_logits

                window_write = rasterio.windows.Window(
                    col_off=0, row_off=y, width=W, height=y_end - y
                )

                if dst_probs is not None:
                    # Softmax needed for probability output
                    logits_cropped -= logits_cropped.max(axis=0, keepdims=True)
                    np.exp(logits_cropped, out=logits_cropped)
                    logits_cropped /= logits_cropped.sum(axis=0, keepdims=True)
                    dst_probs.write(logits_cropped, window=window_write)

                seg = np.argmax(logits_cropped, axis=0).astype(np.uint8)
                dst_seg.write(seg[np.newaxis], window=window_write)
                del seg, logits_cropped

                y = y_end
                stripe_idx += 1
        finally:
            src_ds.close()
            dst_seg.close()
            if dst_probs is not None:
                dst_probs.close()

        logger.info(f"Striped inference complete: {output_seg_path}")

    def save_inference(
        self,
        image_path,
        threshold,
        profile,
        inference,
        output_dict,
        apply_threshold=True,
    ):
        """
        Save single-band image with class indices (no threshold!)
        """
        # inference["seg"] is already [H, W] with values 0, 1, 2, ..., num_classes-1

        # Update profile to 1 band, dtype uint8
        profile["count"] = 1
        profile["dtype"] = "uint8"
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
        normalize_max_value: Optional[float] = None,
        config=None,
        group_output_by_image_basename=False,
        use_tta: bool = False,
        tta_augmentations: Optional[List[str]] = None,
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
            normalize_max_value=normalize_max_value,
            config=config,
            group_output_by_image_basename=group_output_by_image_basename,
            use_tta=use_tta,
            tta_augmentations=tta_augmentations,
        )
        # The crossfield tensor encodes tangent-field angles whose values must
        # also be rotated when the spatial axes are rotated.  That transform is
        # non-trivial, so we skip crossfield de-augmentation and take it from
        # the identity (rot0) pass instead.
        self._tta_skip_keys: FrozenSet[str] = frozenset({"crossfield"})

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
        normalize_max_value: Optional[float] = None,
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
            normalize_max_value=normalize_max_value,
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

    def save_inference(
        self,
        image_path,
        threshold,
        profile,
        inference,
        output_dict,
        apply_threshold=True,
    ):
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
        normalize_max_value: Optional[float] = None,
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
            normalize_max_value=normalize_max_value,
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
