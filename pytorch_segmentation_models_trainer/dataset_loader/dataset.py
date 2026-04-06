# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-02-25
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
import contextlib
import itertools
import json
import logging
import math
import os
import rasterio
from collections import OrderedDict
from copy import deepcopy
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
import shapely.wkt
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from albumentations.pytorch import ToTensorV2
from pytorch_segmentation_models_trainer.utils import polygonrnn_utils
from pytorch_segmentation_models_trainer.utils.object_detection_utils import (
    bbox_xywh_to_xyxy,
    bbox_xyxy_to_xywh,
)
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from torch.utils.data import Dataset
import copy

from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from pytorch_toolbelt.utils.torch_utils import (
    image_to_tensor,
    tensor_from_rgb_image,
    to_numpy,
)
import kornia as K


def _sanitize_aug_config(cfg):
    """Handle always_apply deprecation in albumentations >= 2.0.

    Converts always_apply=True to p=1.0 and removes always_apply=False,
    since albumentations 2.0+ no longer accepts this parameter.
    """
    try:
        container = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        container = dict(cfg) if hasattr(cfg, "items") else cfg
    if isinstance(container, dict) and "always_apply" in container:
        always_apply = container.pop("always_apply")
        if always_apply and "p" not in container:
            container["p"] = 1.0
    return container


def load_augmentation_object(input_list, bbox_params=None):
    if isinstance(input_list, A.Compose):
        return input_list
    try:
        aug_list = [
            instantiate(_sanitize_aug_config(i), _recursive_=False)
            for i in input_list
        ]
    except:
        aug_list = input_list
    if bbox_params is None:
        return A.Compose(aug_list)
    if isinstance(bbox_params, A.BboxParams):
        return A.Compose(aug_list, bbox_params=bbox_params)
    return A.Compose(aug_list, bbox_params=OmegaConf.to_container(bbox_params))


class AbstractDataset(Dataset):
    def __init__(
        self,
        input_csv_path: Path = None,
        df: pd.DataFrame = None,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        mask_key=None,
        n_first_rows_to_read=None,
    ) -> None:
        if input_csv_path is None and df is None:
            raise ValueError("Must provide either input_csv_path or df")
        self.input_csv_path = input_csv_path
        self.root_dir = root_dir
        if df is not None:
            self.df = df
        else:
            self.df = (
                pd.read_csv(input_csv_path)
                if n_first_rows_to_read is None
                else pd.read_csv(input_csv_path, nrows=n_first_rows_to_read)
            )
        self.len = len(self.df)
        self.transform = (
            None
            if augmentation_list is None
            else load_augmentation_object(augmentation_list)
        )
        self.data_loader = data_loader
        self.image_key = image_key if image_key is not None else "image"
        self.mask_key = mask_key if mask_key is not None else "mask"

    def update_df(self, new_df):
        self.df = new_df
        self.len = len(self.df)

    def __len__(self) -> int:
        return self.len

    @abstractmethod
    def __getitem__(
        self, idx: int
    ) -> Union[Dict[str, Any], Tuple[torch.Tensor, Any], Tuple[torch.Tensor, Any, Any]]:
        """Item getter. Must be reimplemented in each subclass.

        Args:
            idx (int): index of the item to be returned

        Returns:
            Dict[str, Any]: Loaded item.
        """
        pass

    def get_path(self, idx: int, key: str = None, add_root_dir: bool = True):
        key = self.image_key if key is None else key
        image_path = str(self.df.iloc[idx][key])
        if self.root_dir is not None and add_root_dir:
            return self._add_root_dir_to_path(image_path)
        return image_path

    def _add_root_dir_to_path(self, image_path):
        return os.path.join(
            self.root_dir,
            image_path if not image_path.startswith(os.path.sep) else image_path[1::],
        )

    def load_image(self, idx, key=None, is_mask=False, force_rgb=False, is_binary_mask=True):
        key = self.image_key if key is None else key
        image_path = self.get_path(idx, key=key)
        return self.load_image_from_path(
            image_path, is_mask=is_mask, force_rgb=force_rgb, is_binary_mask=is_binary_mask
        )

    def load_image_from_path(self, image_path, is_mask=False, force_rgb=False, is_binary_mask=True):
        try:
            image = (
                Image.open(image_path)
                if not is_mask
                else Image.open(image_path).convert("L")
            )
            if force_rgb:
                image = image.convert("RGB")
            image = np.array(image)
        except Exception:
            # Fallback to rasterio for multi-band GeoTIFFs not supported by PIL
            with rasterio.open(image_path) as src:
                if is_mask:
                    image = src.read(1)
                else:
                    image = src.read()  # (C, H, W)
                    image = np.transpose(image, (1, 2, 0))  # (H, W, C)
        if not is_mask:
            return image
        if is_binary_mask:
            return (image > 0).astype(np.uint8)
        return image.astype(np.uint8)

    def to_tensor(self, x):
        return x if isinstance(x, torch.Tensor) else torch.from_numpy(x)


class ImageDataset(AbstractDataset):
    def __init__(
        self,
        input_csv_path: Path = None,
        df=None,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        n_first_rows_to_read=None,
    ) -> None:
        super(ImageDataset, self).__init__(
            input_csv_path=input_csv_path,
            df=df,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            n_first_rows_to_read=n_first_rows_to_read,
        )

    @classmethod
    def get_grouped_datasets(
        cls,
        df,
        group_by_keys: List[str],
        root_dir=None,
        augmentation_list=None,
        image_key=None,
        n_first_rows_to_read=None,
        **kwargs,
    ):
        return {
            k: cls(
                df=pd.DataFrame(df.iloc[v]).reset_index(),
                root_dir=root_dir,
                augmentation_list=augmentation_list,
                image_key=image_key,
                n_first_rows_to_read=n_first_rows_to_read,
                **kwargs,
            )
            for k, v in df.groupby(group_by_keys).groups.items()
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % self.len

        image = self.load_image(idx, key=self.image_key)
        result = (
            {"image": image} if self.transform is None else self.transform(image=image)
        )
        result.update({"path": self.get_path(idx, key=self.image_key)})
        return result


class TiledInferenceImageDataset(ImageDataset):
    def __init__(
        self,
        input_csv_path: Path = None,
        df=None,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        normalize_output=True,
        n_first_rows_to_read=None,
        pad_if_needed=False,
        model_input_shape=None,
        step_shape=None,
    ) -> None:
        super(TiledInferenceImageDataset, self).__init__(
            input_csv_path=input_csv_path,
            df=df,
            root_dir=root_dir,
            augmentation_list=None,
            data_loader=data_loader,
            image_key=image_key,
            n_first_rows_to_read=n_first_rows_to_read,
        )
        if pad_if_needed and model_input_shape is None:
            raise ValueError("Must provide model_input_shape if pad_if_needed is True")
        self.pad_if_needed = pad_if_needed
        self.model_input_shape = model_input_shape
        self.step_shape = (224, 224) if step_shape is None else step_shape
        self.transform = A.Normalize() if normalize_output else None
        self.tiler_dict = dict()
        self.shape_dict = dict()
        self.pad_func = A.PadIfNeeded(
            math.ceil(self.df["width"].max() / self.model_input_shape[0])
            * self.model_input_shape[0],
            math.ceil(self.df["height"].max() / self.model_input_shape[1])
            * self.model_input_shape[1],
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % self.len

        image = self.load_image(idx, key=self.image_key)
        self.shape_dict[idx] = image.shape[0:2]
        result = (
            {"image": image} if self.transform is None else self.transform(image=image)
        )
        if self.pad_if_needed:
            result = self.pad_func(**result)
        result.update({"path": self.get_path(idx, key=self.image_key)})
        tiler = ImageSlicer(
            result["image"].shape,
            tile_size=self.model_input_shape,
            tile_step=self.step_shape,
        )
        tiles = [image_to_tensor(tile) for tile in tiler.split(result.pop("image"))]
        result.update({"tiles": torch.stack(tiles)})
        result.update(
            {"tile_image_idx": idx * torch.ones(len(tiles), dtype=torch.int64)}
        )
        result.update({"tiler_object": tiler})
        result.update({"original_shape": tuple(self.df[["width", "height"]].iloc[idx])})
        return result

    @staticmethod
    def collate_fn(batch: List) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        paths = [item["path"] for item in batch]
        tiles = torch.cat([item["tiles"] for item in batch], dim=0)
        indexes = torch.cat([item["tile_image_idx"] for item in batch], dim=0)
        tiler_object_list = [item["tiler_object"] for item in batch]
        original_shape_list = [item["original_shape"] for item in batch]
        return {
            "path": paths,
            "tiles": tiles,
            "tile_image_idx": indexes,
            "tiler_object_list": tiler_object_list,
            "original_shape": original_shape_list,
        }

class SegmentationDataset(AbstractDataset):
    """Segmentation dataset with band selection support.

    Args:
        input_csv_path (Path): Path to the CSV file with the data
        root_dir: Root directory of the data
        augmentation_list: List of Albumentations augmentations
        data_loader: DataLoader configuration
        image_key: Column name with image paths
        mask_key: Column name with mask paths
        n_first_rows_to_read: Number of rows to read from the CSV
        n_classes: Number of segmentation classes
        selected_bands (Optional[List[int]]): List of band indices to load
            (1-based). E.g.: [1, 2, 3] loads the first 3 bands.
            If None, loads all available bands.
        use_rasterio (bool): If True, uses rasterio to load images (recommended
            for multispectral images). If False, uses PIL (default for RGB).
    """

    def __init__(
        self,
        input_csv_path: Path,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        mask_key=None,
        n_first_rows_to_read=None,
        n_classes=2,
        selected_bands: Optional[List[int]] = None,
        use_rasterio: bool = False,
        reset_augmentation_function: bool = False,
    ) -> None:
        super(SegmentationDataset, self).__init__(
            input_csv_path=input_csv_path,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            mask_key=mask_key,
            n_first_rows_to_read=n_first_rows_to_read,
        )
        self.n_classes = n_classes
        self.selected_bands = selected_bands
        self.use_rasterio = use_rasterio
        self.reset_augmentation_function = reset_augmentation_function
        
        # Validate selected bands
        if self.selected_bands is not None:
            if not all(isinstance(b, int) and b > 0 for b in self.selected_bands):
                raise ValueError("selected_bands must contain only positive integers (1-based)")

    def load_image(self, idx: int, key: str = None, is_mask: bool = False, 
                   force_rgb: bool = False, is_binary_mask: bool = True) -> np.ndarray:
        """Load image with band selection support.

        Args:
            idx: Image index in the dataset
            key: Column key in the DataFrame
            is_mask: If True, loads as a mask
            force_rgb: If True, forces conversion to RGB
            is_binary_mask: If True, converts mask to binary

        Returns:
            np.ndarray: Loaded image with shape (H, W, C) or (H, W) for masks
        """
        key = self.image_key if key is None else key
        image_path = self.get_path(idx, key=key)
        
        # Masks always use the original method
        if is_mask:
            return self.load_image_from_path(
                image_path, 
                is_mask=True, 
                force_rgb=force_rgb,
                is_binary_mask=is_binary_mask
            )
        
        # If using rasterio OR if band selection is specified, use rasterio method
        if self.use_rasterio or self.selected_bands is not None:
            return self._load_image_with_rasterio(image_path)
        else:
            # Default method for RGB images
            return self.load_image_from_path(image_path, is_mask=False, force_rgb=force_rgb)

    def _load_image_with_rasterio(self, image_path: str) -> np.ndarray:
        with rasterio.open(image_path, 'r') as src:
            data = src.read() if self.selected_bands is None \
                else src.read(self.selected_bands)
            image = np.transpose(data, (1, 2, 0)).copy()
            del data
            src.close()
        return np.array(image, dtype=np.uint8)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns a dataset item with custom band support."""
        idx = idx % self.len

        image = self.load_image(idx, key=self.image_key)
        mask = self.load_image(
            idx, 
            key=self.mask_key, 
            is_mask=True, 
            is_binary_mask=(self.n_classes == 2)
        )
        if self.transform is None:
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)  # (H,W,C) -> (C,H,W) for PyTorch
            image = image / 255.0
            mask = torch.from_numpy(mask).long()
            return {
                "image": image,
                "mask": mask
            }
        # This copy is to avoid data leaks from albumentations.
        # Albumentations stores caches and sometimes do not purge them from the memory, 
        # inducing crashes on the server while training, due to lack of available memory.
        # We added a deepcopy to copy the function, perform the transformation, copy the 
        # result and delete any albumentations related object before returning the evaluated data.
        if not self.reset_augmentation_function:
            return self.transform(image=image, mask=mask)
        transform_func = deepcopy(self.transform)
        output = transform_func(image=image, mask=mask)
        output_dict = {
            "image": output["image"].clone().detach() if torch.is_tensor(output["image"]) else output["image"].copy(),
            "mask": output["mask"].clone().detach() if torch.is_tensor(output["mask"]) else output["mask"].copy(),
        }
        del output
        del transform_func
        return output_dict


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_gdal_crs_warnings():
    """Temporarily set GTIFF_SRS_SOURCE=GEOKEYS to suppress GDAL CRS mismatch warnings."""
    prev = os.environ.get("GTIFF_SRS_SOURCE")
    os.environ["GTIFF_SRS_SOURCE"] = "GEOKEYS"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("GTIFF_SRS_SOURCE", None)
        else:
            os.environ["GTIFF_SRS_SOURCE"] = prev


def _worker_init_fn(worker_id):
    """Seed numpy random per DataLoader worker to avoid duplicate crops.

    PyTorch seeds torch.manual_seed per worker but NOT numpy. Without this,
    all workers using np.random produce identical crop sequences when using
    spawn (Windows) or fork start methods.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


class _RasterioLRUCache:
    """LRU cache for rasterio file handles with explicit close on eviction.

    Unlike functools.lru_cache, this explicitly closes evicted rasterio
    DatasetReader handles, ensuring OS file descriptors are released
    promptly instead of relying on garbage collection.

    Args:
        maxsize: Maximum number of open file handles. When exceeded,
            the least recently used handle is closed and removed.
    """

    def __init__(self, maxsize: int = 64):
        self._cache = OrderedDict()
        self._maxsize = maxsize

    def get(self, path: str):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        if len(self._cache) >= self._maxsize:
            _, old_src = self._cache.popitem(last=False)
            try:
                old_src.close()
            except Exception:
                pass
        src = rasterio.open(path)
        self._cache[path] = src
        return src

    def close_all(self):
        for src in self._cache.values():
            try:
                src.close()
            except Exception:
                pass
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


class RandomCropSegmentationDataset(AbstractDataset):
    """Dataset that performs random crops on-the-fly from full-size images.

    Instead of pre-generating tiles on disk, reads random windows directly
    from large images using rasterio windowed read. Each __getitem__
    selects a random image (weighted by area) and extracts a fixed-size crop.

    Rasterio file handles are kept in an LRU cache to avoid the overhead
    of opening/closing files on every crop, limiting the number of
    simultaneously open file descriptors. The cache is created lazily
    in each DataLoader worker and cleaned up automatically.

    The input CSV must list images and masks at their original size
    (not pre-cut tiles).

    Supports an integrated class balancing system:
    - class_balanced_sampling: samples crops biased toward rare classes
    - cutmix_prob/cutmix_alpha: class-aware CutMix (second crop seeks a
      different class from the first to maximize per-sample diversity)

    Args:
        input_csv_path: CSV with 'image' and 'mask' columns (full-size paths)
        crop_size: Crop size [H, W] (default: [256, 256])
        samples_per_epoch: Number of random crops per epoch.
            If 0, automatically calculated based on the number and size
            of images (~95% total area coverage, 3x multiplier).
        n_classes: Number of segmentation classes
        selected_bands: Band indices to load (1-based). None = all.
        use_rasterio: If True, uses rasterio for images (recommended for multi-band)
        reset_augmentation_function: Deepcopy of transform to avoid memory leak
        min_valid_ratio: Minimum fraction of valid pixels (>0) in the crop
        max_retries: Maximum attempts to find a valid crop
        file_cache_maxsize: Maximum rasterio file handles open per worker.
            If 0, automatically calculated as 2 * number_of_images
            (to cache image + mask for all). Default 0 (auto).
            On servers with many images, adjust ulimit -n 65536.
        class_balanced_sampling: If True, samples crops biased toward
            under-represented classes using a pre-computed spatial index.
        class_sampling_stride: Stride of the mask scan grid used to
            build the per-class position index. If 0, uses
            crop_size[0] // 2. Default 0 (auto).
        cutmix_prob: Probability of applying CutMix per sample.
            The second crop seeks a different class from the dominant
            one in the first (class-aware CutMix). 0 disables. Default 0.
        cutmix_alpha: Parameter of the Beta(alpha, alpha) distribution for
            the CutMix lambda. 1.0 = uniform in [0,1]. Default 1.0.
        cutmix_rare_classes: List of rare class indices to be forced as
            the target of the second CutMix crop. When active, with
            probability cutmix_rare_prob the second crop specifically
            seeks one of these classes. Useful for extremely
            under-represented classes. Default None.
        cutmix_rare_weights: Sampling weights for each class in
            cutmix_rare_classes. Must have the same size as
            cutmix_rare_classes. If None, uses uniform weights.
            E.g.: [0.3, 0.7] for 30% class1 + 70% class2. Default None.
        cutmix_rare_prob: Probability of forcing a rare class in the second
            CutMix crop (when cutmix_rare_classes is set).
            Default 0.5 (50% of CutMix operations seek a rare class).
        classmix: If True, uses ClassMix (copies class-shaped regions)
            instead of rectangular CutMix. Preserves semantic boundaries.
            Ref: Olsson et al., WACV 2021. Default False.
        pre_mix_color_aug: Optional list of albumentations color augmentation
            configs applied independently to each crop BEFORE ClassMix/CutMix.
            Simulates different radiometric conditions between source images,
            which is critical for remote sensing where pasted regions should
            have independent spectral variation. Only applied when mixing
            is triggered (cutmix_prob). Example:
            [RandomBrightnessContrast(0.15), RandomGamma([80,120])].
            Default None (disabled).
        soft_labels: If True, masks are multi-band float32 GeoTIFFs
            (C bands, one per class) with probability distributions
            instead of integer indices. Ignored pixels have all
            bands == 0. Default False.
        hard_mask_csv: Path to CSV with hard masks (1-band uint8).
            When provided together with soft_labels=True, the spatial
            class index uses the hard masks (much faster to read).
            The CSV must have the same structure and order as input_csv_path.
            Default None (uses the dataset's own masks).
        return_hard_mask: If True and soft_labels=True and hard_mask_csv
            is provided, __getitem__ returns a "hard_mask" key in addition
            to "image" and "mask". The hard mask is read at the same crop
            position and receives the same geometric augmentations.
            Used for dual-head training. Default False.
        grid_mode: If True, uses deterministic grid-based tiling instead
            of random crops. Tile positions are pre-computed based on
            crop_size and overlap, and invalid tiles (below min_valid_ratio)
            are filtered out during initialization. Randomness is limited
            to the DataLoader shuffle order per epoch. samples_per_epoch
            is ignored; __len__ returns the number of valid grid tiles.
            Default False (random crop mode).
        overlap_x: Horizontal overlap between adjacent tiles as a fraction
            [0.0, 1.0). 0.5 means 50% overlap. Only used when grid_mode=True.
            Default 0.5.
        overlap_y: Vertical overlap between adjacent tiles as a fraction
            [0.0, 1.0). 0.5 means 50% overlap. Only used when grid_mode=True.
            Default 0.5.
    """

    def __init__(
        self,
        input_csv_path: Path,
        crop_size: Optional[List[int]] = None,
        samples_per_epoch: int = 10000,
        n_classes: int = 2,
        selected_bands: Optional[List[int]] = None,
        use_rasterio: bool = True,
        reset_augmentation_function: bool = False,
        min_valid_ratio: float = 0.1,
        max_retries: int = 10,
        file_cache_maxsize: int = 0,
        class_balanced_sampling: bool = False,
        class_sampling_stride: int = 0,
        cutmix_prob: float = 0.0,
        cutmix_alpha: float = 1.0,
        cutmix_rare_classes: Optional[List[int]] = None,
        cutmix_rare_weights: Optional[List[float]] = None,
        cutmix_rare_prob: float = 0.5,
        classmix: bool = False,
        soft_labels: bool = False,
        hard_mask_csv: str = None,
        return_hard_mask: bool = False,
        pre_mix_color_aug: Optional[list] = None,
        class_presence_cache: str = None,
        grid_cache: str = None,
        grid_mode: bool = False,
        overlap_x: float = 0.5,
        overlap_y: float = 0.5,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        mask_key=None,
        n_first_rows_to_read=None,
    ) -> None:
        super(RandomCropSegmentationDataset, self).__init__(
            input_csv_path=input_csv_path,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            mask_key=mask_key,
            n_first_rows_to_read=n_first_rows_to_read,
        )
        self.crop_size = crop_size if crop_size is not None else [256, 256]
        self.n_classes = n_classes
        self.selected_bands = selected_bands
        self.use_rasterio = use_rasterio
        self.reset_augmentation_function = reset_augmentation_function
        self.min_valid_ratio = min_valid_ratio
        self.max_retries = max_retries
        self.class_balanced_sampling = class_balanced_sampling
        self.class_sampling_stride = class_sampling_stride
        self.cutmix_prob = cutmix_prob
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_rare_classes = cutmix_rare_classes or []
        # Normalize weights to sum to 1.0
        if cutmix_rare_weights and self.cutmix_rare_classes:
            w = np.array(cutmix_rare_weights, dtype=np.float64)
            self.cutmix_rare_weights = w / w.sum()
        else:
            self.cutmix_rare_weights = None  # uniform
        self.cutmix_rare_prob = cutmix_rare_prob
        self.classmix = classmix
        self.soft_labels = soft_labels
        self.hard_mask_csv = hard_mask_csv
        self.return_hard_mask = return_hard_mask and soft_labels
        # Cached transform with additional_targets for hard_mask (built lazily)
        self._transform_with_hard_mask = None
        # Pre-mix color augmentation: applied independently to each crop
        # before ClassMix/CutMix, simulating different radiometric conditions
        # between source images (critical for remote sensing).
        self._pre_mix_transform = (
            load_augmentation_object(pre_mix_color_aug)
            if pre_mix_color_aug
            else None
        )
        self.class_presence_cache = class_presence_cache
        self.grid_cache = grid_cache
        self.grid_mode = grid_mode
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y
        if grid_mode:
            for name, val in [("overlap_x", overlap_x), ("overlap_y", overlap_y)]:
                if not (0.0 <= val < 1.0):
                    raise ValueError(f"{name} must be in [0.0, 1.0), got {val}")
        self._hard_mask_paths = None
        if hard_mask_csv and soft_labels:
            import pandas as pd
            hard_df = pd.read_csv(hard_mask_csv)
            mask_col = mask_key if mask_key is not None else "mask"
            self._hard_mask_paths = hard_df[mask_col].tolist()
            logger.info(
                f"RandomCropSegmentationDataset: using hard masks from "
                f"{hard_mask_csv} for spatial index ({len(self._hard_mask_paths)} entries)"
            )

        if self.selected_bands is not None:
            if not all(isinstance(b, int) and b > 0 for b in self.selected_bands):
                raise ValueError(
                    "selected_bands must contain only positive integers (1-based)"
                )

        # Automatically calculate file_cache_maxsize if 0:
        # 2 * N images (image + mask) to cache everything.
        # Each open handle consumes ~10KB of GDAL metadata,
        # so 1000 handles = ~10MB per worker — negligible.
        # The real limit is the OS ulimit (default 1024 on Linux).
        n_files = len(self.df) if hasattr(self, 'df') else 500
        if file_cache_maxsize <= 0:
            self.file_cache_maxsize = 2 * n_files + 16  # margin
            logger.info(
                f"RandomCropSegmentationDataset: file_cache_maxsize auto = "
                f"{self.file_cache_maxsize} ({n_files} images x 2 + margin). "
                f"Make sure ulimit -n >= {self.file_cache_maxsize + 256}"
            )
        else:
            self.file_cache_maxsize = file_cache_maxsize

        # LRU cache of rasterio file handles (lazy, created per worker)
        # Cannot be initialized here with open files because
        # rasterio datasets are not picklable and DataLoader uses spawn
        # (Windows) or fork for workers.
        self._file_cache = _RasterioLRUCache(maxsize=self.file_cache_maxsize)

        # Cache of image dimensions (reads header only, fast)
        self.image_dims = []  # list of (width, height)
        self._valid_indices = []  # indices of images large enough
        crop_h, crop_w = self.crop_size
        with _suppress_gdal_crs_warnings():
            for i in range(len(self.df)):
                image_path = self.get_path(i, key=self.image_key)
                with rasterio.open(image_path) as src:
                    w, h = src.width, src.height
                self.image_dims.append((w, h))
                if w >= crop_w and h >= crop_h:
                    self._valid_indices.append(i)

        if not self._valid_indices:
            raise ValueError(
                f"No image large enough for crop {self.crop_size}. "
                f"Dimensions found: {self.image_dims}"
            )

        # Weights proportional to the number of possible crop positions
        positions = []
        for idx in self._valid_indices:
            w, h = self.image_dims[idx]
            positions.append((w - crop_w + 1) * (h - crop_h + 1))
        total_positions = sum(positions)
        self.image_weights = np.array([p / total_positions for p in positions])

        # Spatial class index for class-balanced sampling.
        # Built once in init (main process), serialized to
        # workers via pickle. All numpy -> picklable (Windows spawn OK).
        self._class_positions = {}  # {class_id: np.ndarray de (img_idx, col, row)}
        self._class_sampling_weights = np.array([], dtype=np.float64)
        self._class_to_images = {}  # {class_id: np.ndarray of img_idx}

        if self.grid_mode:
            # Grid mode: deterministic tile positions, shuffle only per epoch.
            # samples_per_epoch is ignored — __len__ returns grid size.
            self._build_grid_positions()
            self.samples_per_epoch = len(self._grid_positions)
            if self.class_balanced_sampling:
                logger.info(
                    "Grid mode active: class_balanced_sampling only affects "
                    "CutMix/ClassMix second crops, not main sampling"
                )
        else:
            # Random crop mode (original behavior)
            # Automatic calculation of samples_per_epoch
            # With random crops, 1x (total_area / crop_area) covers only ~63%
            # of the area per epoch (1 - e^-1). A 3x multiplier ensures ~95%
            # coverage: 1 - e^-3 ~ 0.95.
            if samples_per_epoch <= 0:
                coverage_multiplier = 3
                crop_area = crop_h * crop_w
                total_pixel_area = sum(
                    self.image_dims[idx][0] * self.image_dims[idx][1]
                    for idx in self._valid_indices
                )
                self.samples_per_epoch = int(
                    coverage_multiplier * total_pixel_area / crop_area
                )
                logger.info(
                    f"RandomCropSegmentationDataset: samples_per_epoch automatically "
                    f"calculated = {self.samples_per_epoch:,} "
                    f"({len(self._valid_indices)} images, "
                    f"total area = {total_pixel_area:,} pixels, "
                    f"crop = {crop_h}x{crop_w}, "
                    f"multiplier = {coverage_multiplier}x for ~95% coverage)"
                )
            else:
                self.samples_per_epoch = samples_per_epoch

            if self.class_balanced_sampling:
                self._build_class_position_index()
            elif self.cutmix_rare_classes:
                self._build_class_presence_index()

    def __getstate__(self):
        """Remove file cache before pickle (for spawn workers on Windows)."""
        state = self.__dict__.copy()
        state["_file_cache"] = _RasterioLRUCache(maxsize=self.file_cache_maxsize)
        return state

    def __setstate__(self, state):
        """Restore state with empty cache (will be lazily populated in the worker)."""
        self.__dict__.update(state)

    def __del__(self):
        """Close all cached file handles."""
        self._close_cache()

    def _close_cache(self):
        """Close all file handles in the cache."""
        if not hasattr(self, "_file_cache"):
            return
        if isinstance(self._file_cache, _RasterioLRUCache):
            self._file_cache.close_all()
        elif isinstance(self._file_cache, dict):
            for src in self._file_cache.values():
                try:
                    src.close()
                except Exception:
                    pass
            self._file_cache.clear()

    def _get_src(self, path: str):
        """Return cached rasterio dataset via LRU. Opens lazily on first access."""
        if isinstance(self._file_cache, _RasterioLRUCache):
            return self._file_cache.get(path)
        # Fallback for plain dict (backwards compatibility with running processes)
        if path not in self._file_cache or self._file_cache[path] is None:
            self._file_cache[path] = rasterio.open(path)
        return self._file_cache[path]

    def get_path(self, idx: int, key: str = None, add_root_dir: bool = True):
        """Override for RandomCropSegmentationDataset.

        Since crops are random and do not correspond 1:1 to CSV indices
        (samples_per_epoch >> len(df)), wraps the index with modulo to avoid
        IndexError. The returned path serves as a visual reference (title
        in the visualization callback), not as an exact crop identifier.
        """
        wrapped_idx = idx % len(self.df)
        return super().get_path(wrapped_idx, key=key, add_root_dir=add_root_dir)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _read_crop(self, image_path: str, x: int, y: int, is_mask: bool = False):
        """Read a fixed-size crop using rasterio windowed read with cache."""
        crop_h, crop_w = self.crop_size
        window = rasterio.windows.Window(
            col_off=x, row_off=y, width=crop_w, height=crop_h
        )
        src = self._get_src(image_path)
        if is_mask:
            if self.soft_labels:
                # Soft labels: C-band float32 GeoTIFF → (H, W, C)
                data = src.read(window=window)  # (C, H, W) float32
                return np.transpose(data, (1, 2, 0)).copy()  # (H, W, C)
            else:
                data = src.read(1, window=window)  # (H, W)
                return data.astype(np.uint8)
        else:
            if self.selected_bands is not None:
                data = src.read(self.selected_bands, window=window)
            else:
                data = src.read(window=window)
            # (C, H, W) -> (H, W, C)
            image = np.transpose(data, (1, 2, 0)).copy()
            return image.astype(np.uint8)

    def _read_full_mask(self, idx: int) -> np.ndarray:
        """Read full mask for image idx as a single-band hard-label array.

        Handles hard_mask_csv and soft_labels (argmax).
        Ignore pixels are set to 255 for soft labels.

        Returns:
            np.ndarray: (H, W) uint8 mask.
        """
        if self._hard_mask_paths is not None:
            with rasterio.open(self._hard_mask_paths[idx]) as src:
                return src.read(1)

        mask_path = self.get_path(idx, key=self.mask_key)
        with rasterio.open(mask_path) as src:
            if self.soft_labels:
                raw = src.read()  # (C, H, W) float32
                ignore = raw.sum(axis=0) == 0
                hard = np.argmax(raw, axis=0).astype(np.uint8)
                hard[ignore] = 255
                return hard
            else:
                return src.read(1)

    def _finalize_class_position_index(self, class_positions: dict, label: str = "built"):
        """Convert class_positions dict to numpy arrays and compute sampling weights.

        Args:
            class_positions: {class_id: list of (img_idx, col, row)} tuples.
            label: Label for log messages.
        """
        for c, pos_list in class_positions.items():
            if pos_list:
                self._class_positions[c] = np.array(pos_list, dtype=np.int32)

        counts = np.array([
            len(self._class_positions.get(c, []))
            for c in range(self.n_classes)
        ], dtype=np.float64)
        weights = np.zeros(self.n_classes, dtype=np.float64)
        mask = counts > 0
        weights[mask] = 1.0 / counts[mask]
        total = weights.sum()
        if total > 0:
            weights /= total
        self._class_sampling_weights = weights

        logger.info(f"Per-class position index {label}:")
        for c in range(self.n_classes):
            n_pos = len(self._class_positions.get(c, []))
            logger.info(
                f"  Class {c}: {n_pos:,} positions, "
                f"sampling weight={self._class_sampling_weights[c]:.4f}"
            )

    def _build_grid_positions(self):
        """Build deterministic grid of tile positions and pre-filter invalid ones.

        Reads each mask once per image to check min_valid_ratio for all
        candidate positions. If class_balanced_sampling or cutmix_rare_classes
        are active, class presence/position data is collected in the same pass.

        If grid_cache is set and the file exists, loads pre-computed positions
        directly. Otherwise computes and saves to grid_cache for next time.

        Populates self._grid_positions (Nx3 int32 array of img_idx, col, row).
        """
        if self.grid_cache and os.path.exists(self.grid_cache):
            return self._load_grid_cache()

        crop_h, crop_w = self.crop_size
        stride_x = max(1, int(crop_w * (1 - self.overlap_x)))
        stride_y = max(1, int(crop_h * (1 - self.overlap_y)))
        total_pixels = crop_h * crop_w

        build_class_positions = self.class_balanced_sampling
        build_class_tiles = bool(self.cutmix_rare_classes) or bool(getattr(self, 'cutmix_prob', 0))

        if build_class_positions:
            class_positions = {c: [] for c in range(self.n_classes)}
        class_tiles = {c: [] for c in range(self.n_classes)} if build_class_tiles else None

        logger.info(
            f"Building grid positions (grid_mode=True, "
            f"crop={crop_h}x{crop_w}, stride={stride_y}x{stride_x}, "
            f"overlap={self.overlap_y:.0%}x{self.overlap_x:.0%}, "
            f"{len(self._valid_indices)} images)..."
        )

        valid_positions = []
        total_candidates = 0

        with _suppress_gdal_crs_warnings():
            for idx in self._valid_indices:
                w, h = self.image_dims[idx]

                # Flush edges: append clamped position if last stride doesn't reach the edge
                cols = list(range(0, w - crop_w + 1, stride_x))
                if cols and cols[-1] < w - crop_w:
                    cols.append(w - crop_w)
                elif not cols and w >= crop_w:
                    cols = [0]

                rows = list(range(0, h - crop_h + 1, stride_y))
                if rows and rows[-1] < h - crop_h:
                    rows.append(h - crop_h)
                elif not rows and h >= crop_h:
                    rows = [0]

                total_candidates += len(cols) * len(rows)
                full_mask = self._read_full_mask(idx)

                for row in rows:
                    for col in cols:
                        crop_region = full_mask[row:row + crop_h, col:col + crop_w]
                        valid_pixels = np.sum(crop_region < self.n_classes)

                        if valid_pixels / total_pixels < self.min_valid_ratio:
                            continue

                        tile_idx = len(valid_positions)
                        valid_positions.append((idx, col, row))

                        if build_class_positions or build_class_tiles:
                            bins = np.bincount(
                                crop_region[crop_region < self.n_classes].ravel(),
                                minlength=self.n_classes,
                            )
                            for c in np.nonzero(bins)[0]:
                                if build_class_positions:
                                    class_positions[c].append((idx, col, row))
                                if build_class_tiles:
                                    class_tiles[c].append(tile_idx)

                del full_mask

        if not valid_positions:
            raise ValueError(
                f"No valid grid tiles found with min_valid_ratio={self.min_valid_ratio}. "
                f"Consider lowering min_valid_ratio or checking your masks."
            )

        self._grid_positions = np.array(valid_positions, dtype=np.int32)

        rejected = total_candidates - len(valid_positions)
        reject_pct = (rejected / total_candidates * 100) if total_candidates > 0 else 0
        logger.info(
            f"Grid positions built: {len(valid_positions):,} valid tiles "
            f"from {total_candidates:,} candidates "
            f"({rejected:,} rejected, {reject_pct:.1f}% rejection rate)"
        )

        if build_class_positions:
            self._finalize_class_position_index(class_positions, label="(from grid pass)")

        if build_class_tiles:
            self._class_to_tiles = {
                c: np.array(tiles, dtype=np.int32)
                for c, tiles in class_tiles.items()
                if tiles
            }
            logger.info("Class-to-tiles index (from grid pass):")
            for c in range(self.n_classes):
                n_tiles = len(self._class_to_tiles.get(c, []))
                logger.info(f"  Class {c}: {n_tiles} tiles")

        # Save cache for next time
        if self.grid_cache:
            self._save_grid_cache()

    def _save_grid_cache(self):
        """Save grid positions and class indices to JSON for fast reload."""
        import json
        cache = {
            "grid_positions": self._grid_positions.tolist(),
            "class_to_tiles": {
                str(c): tiles.tolist()
                for c, tiles in getattr(self, "_class_to_tiles", {}).items()
            },
            "class_to_images": {
                str(c): imgs.tolist()
                for c, imgs in getattr(self, "_class_to_images", {}).items()
            },
            "class_positions": {
                str(c): pos.tolist()
                for c, pos in getattr(self, "_class_positions", {}).items()
            },
            "config": {
                "crop_size": list(self.crop_size),
                "overlap_x": float(self.overlap_x),
                "overlap_y": float(self.overlap_y),
                "min_valid_ratio": float(self.min_valid_ratio),
                "n_images": len(self._valid_indices),
            },
        }
        os.makedirs(os.path.dirname(self.grid_cache) or ".", exist_ok=True)
        with open(self.grid_cache, "w") as f:
            json.dump(cache, f)
        logger.info(f"Grid cache saved: {self.grid_cache} ({len(self._grid_positions)} tiles)")

    def _load_grid_cache(self):
        """Load pre-computed grid positions from JSON cache."""
        import json
        logger.info(f"Loading grid cache from {self.grid_cache}...")
        with open(self.grid_cache) as f:
            cache = json.load(f)

        # Validate config matches
        cfg = cache.get("config", {})
        if cfg.get("crop_size") != self.crop_size:
            logger.warning(
                f"Grid cache crop_size mismatch: cache={cfg.get('crop_size')}, "
                f"current={self.crop_size}. Rebuilding."
            )
            return self._build_grid_positions_fresh()
        if cfg.get("n_images") != len(self._valid_indices):
            logger.warning(
                f"Grid cache n_images mismatch: cache={cfg.get('n_images')}, "
                f"current={len(self._valid_indices)}. Rebuilding."
            )
            return self._build_grid_positions_fresh()
        for param in ("overlap_x", "overlap_y", "min_valid_ratio"):
            if cfg.get(param) != getattr(self, param):
                logger.warning(
                    f"Grid cache {param} mismatch: cache={cfg.get(param)}, "
                    f"current={getattr(self, param)}. Rebuilding."
                )
                return self._build_grid_positions_fresh()

        self._grid_positions = np.array(cache["grid_positions"], dtype=np.int32)

        # Restore class-to-tiles index
        if cache.get("class_to_tiles"):
            self._class_to_tiles = {
                int(c): np.array(tiles, dtype=np.int32)
                for c, tiles in cache["class_to_tiles"].items()
                if tiles
            }

        # Restore class indices if present
        if cache.get("class_to_images"):
            self._class_to_images = {
                int(c): np.array(imgs, dtype=np.int32)
                for c, imgs in cache["class_to_images"].items()
                if imgs
            }
        if cache.get("class_positions"):
            self._class_positions = {
                int(c): np.array(pos, dtype=np.int32)
                for c, pos in cache["class_positions"].items()
                if pos
            }
            # Rebuild sampling weights
            counts = np.array([
                len(self._class_positions.get(c, []))
                for c in range(self.n_classes)
            ], dtype=np.float64)
            weights = np.zeros(self.n_classes, dtype=np.float64)
            mask = counts > 0
            weights[mask] = 1.0 / counts[mask]
            total = weights.sum()
            if total > 0:
                weights /= total
            self._class_sampling_weights = weights

        logger.info(f"Grid cache loaded: {len(self._grid_positions)} tiles")

    def _build_grid_positions_fresh(self):
        """Rebuild grid without cache (called when cache is invalid)."""
        old_cache = self.grid_cache
        self.grid_cache = None  # Prevent infinite recursion
        self._build_grid_positions()
        self.grid_cache = old_cache
        self._save_grid_cache()

    def _build_class_position_index(self):
        """Build spatial class index for class-balanced sampling.

        For each valid image, reads the entire mask and detects which classes
        are present at each crop position on a grid. The result is a dictionary
        mapping each class to an array of positions (img_idx, col, row).

        Sampling weights are the inverse of the number of positions per
        class, normalized. Classes with more positions (abundant) receive
        a lower probability of being sampled.
        """
        crop_h, crop_w = self.crop_size
        stride = self.class_sampling_stride
        if stride <= 0:
            stride = crop_h // 2

        class_positions = {c: [] for c in range(self.n_classes)}

        logger.info(
            f"Building per-class position index "
            f"(stride={stride}, {len(self._valid_indices)} images)..."
        )

        if self._hard_mask_paths is not None:
            logger.info(
                "  Using hard masks for index construction (fast, 1-band uint8)"
            )

        with _suppress_gdal_crs_warnings():
            for idx in self._valid_indices:
                full_mask = self._read_full_mask(idx)
                w, h = self.image_dims[idx]

                for row in range(0, h - crop_h + 1, stride):
                    for col in range(0, w - crop_w + 1, stride):
                        crop_region = full_mask[row:row + crop_h, col:col + crop_w]
                        classes_present = np.unique(crop_region)
                        for c in classes_present:
                            if 0 <= c < self.n_classes:
                                class_positions[c].append((idx, col, row))

        self._finalize_class_position_index(class_positions)

    def _build_class_presence_index(self):
        """Build lightweight class-to-images index for CutMix rare targeting.

        Unlike _build_class_position_index (which stores per-crop positions
        on a grid), this only records WHICH images contain WHICH classes
        using np.unique on each mask. Much faster and lighter:
        ~30-60s for 1000 images vs ~3-6 min for the full position index.

        Used when class_balanced_sampling=False but cutmix_rare_classes
        is set — allows _get_random_crop to target images that contain
        the desired rare class without biasing ALL sampling.

        If class_presence_cache is set and the JSON exists, loads the
        pre-computed index directly (instant). The JSON maps image
        filenames to lists of class indices present.
        """
        if self.class_presence_cache and os.path.exists(self.class_presence_cache):
            return self._load_class_presence_cache()
        return self._build_class_presence_index_fresh()

    def _build_class_presence_index_fresh(self):
        """Build class-presence index from scratch by reading all masks."""
        logger.info(
            f"Building class-presence index "
            f"({len(self._valid_indices)} images, "
            f"{'hard masks' if self._hard_mask_paths else 'soft masks'})..."
        )

        class_images = {c: [] for c in range(self.n_classes)}

        with _suppress_gdal_crs_warnings():
            for idx in self._valid_indices:
                full_mask = self._read_full_mask(idx)
                classes_present = np.unique(full_mask)
                for c in classes_present:
                    if 0 <= c < self.n_classes:
                        class_images[c].append(idx)

        self._class_to_images = {
            c: np.array(imgs, dtype=np.int32)
            for c, imgs in class_images.items()
            if imgs
        }

        logger.info("Class-presence index built:")
        for c in range(self.n_classes):
            n_imgs = len(self._class_to_images.get(c, []))
            logger.info(f"  Class {c}: present in {n_imgs} images")

        # Auto-save (or overwrite stale) cache for next time
        if self.class_presence_cache:
            self._save_class_presence_cache()

    def _save_class_presence_cache(self):
        """Save class-presence index to JSON for fast reload."""
        import json
        # Build inverse lookup: idx -> set of classes (avoids O(N) numpy search)
        idx_to_classes = {}
        for c, imgs in self._class_to_images.items():
            for idx in imgs:
                idx_to_classes.setdefault(int(idx), []).append(int(c))
        cache = {
            "_config": {
                "n_images": len(self._valid_indices),
                "n_classes": self.n_classes,
            },
        }
        for idx in self._valid_indices:
            mask_path = self.get_path(idx, key=self.mask_key)
            basename = os.path.basename(mask_path)
            cache[basename] = sorted(idx_to_classes.get(idx, []))
        os.makedirs(os.path.dirname(self.class_presence_cache) or ".", exist_ok=True)
        with open(self.class_presence_cache, "w") as f:
            json.dump(cache, f, indent=2)
        logger.info(f"Class-presence cache saved: {self.class_presence_cache}")

    def _load_class_presence_cache(self):
        """Load pre-computed class-presence index from JSON.

        The JSON maps mask filenames (or image filenames) to lists of
        class indices present. The method matches filenames to dataset
        indices via the mask column in the CSV.

        Expected JSON format:
            {
                "treino/masks/mask_IMG_0.tif": [0, 1, 4, 5, 6],
                "treino/masks/mask_IMG_1.tif": [0, 2, 3, 5],
                ...
            }
        """
        import json

        logger.info(
            f"Loading class-presence cache from {self.class_presence_cache}..."
        )
        with open(self.class_presence_cache) as f:
            cache = json.load(f)

        # Validate config if present
        cfg = cache.pop("_config", {})
        if cfg:
            if cfg.get("n_images") != len(self._valid_indices):
                logger.warning(
                    f"Class-presence cache n_images mismatch: "
                    f"cache={cfg.get('n_images')}, "
                    f"current={len(self._valid_indices)}. Rebuilding."
                )
                return self._build_class_presence_index_fresh()
            if cfg.get("n_classes") != self.n_classes:
                logger.warning(
                    f"Class-presence cache n_classes mismatch: "
                    f"cache={cfg.get('n_classes')}, "
                    f"current={self.n_classes}. Rebuilding."
                )
                return self._build_class_presence_index_fresh()

        # Build lookup: mask_path -> list of classes
        # The cache keys may be relative or absolute; try matching by basename
        cache_by_basename = {}
        for path, classes in cache.items():
            basename = os.path.basename(path)
            cache_by_basename[basename] = classes

        class_images = {c: [] for c in range(self.n_classes)}

        matched = 0
        for idx in self._valid_indices:
            mask_path = self.get_path(idx, key=self.mask_key)
            basename = os.path.basename(mask_path)
            classes = cache_by_basename.get(basename)
            if classes is None:
                continue
            matched += 1
            for c in classes:
                if 0 <= c < self.n_classes:
                    class_images[c].append(idx)

        self._class_to_images = {
            c: np.array(imgs, dtype=np.int32)
            for c, imgs in class_images.items()
            if imgs
        }

        n_total = len(self._valid_indices)
        unmatched = n_total - matched
        if unmatched > 0:
            ratio = matched / n_total if n_total else 0
            logger.warning(
                f"Class-presence cache: {unmatched}/{n_total} images NOT matched. "
                f"Cache may be stale ({self.class_presence_cache})."
            )
            if ratio < 0.9:
                logger.warning(
                    f"Match ratio {ratio:.1%} < 90%. Rebuilding from scratch."
                )
                return self._build_class_presence_index_fresh()
        logger.info(
            f"Class-presence cache loaded: {matched}/{n_total} "
            f"images matched"
        )
        for c in range(self.n_classes):
            n_imgs = len(self._class_to_images.get(c, []))
            logger.info(f"  Class {c}: present in {n_imgs} images")

    def _sample_image_with_class(self, target_class):
        """Sample a random crop from an image that contains target_class.

        Used by CutMix rare targeting when class_balanced_sampling is off.
        Picks a random image known to contain the target class, then
        samples a random crop position within it.

        Args:
            target_class: Class index to target.

        Returns:
            Tuple[int, int, int]: (img_idx, x, y) position for crop.
        """
        crop_h, crop_w = self.crop_size
        images = self._class_to_images.get(target_class)

        if images is None or len(images) == 0:
            choice = np.random.choice(
                len(self._valid_indices), p=self.image_weights
            )
            img_idx = self._valid_indices[choice]
        else:
            img_idx = images[np.random.randint(len(images))]

        w, h = self.image_dims[img_idx]
        x = np.random.randint(0, max(1, w - crop_w + 1))
        y = np.random.randint(0, max(1, h - crop_h + 1))
        return img_idx, int(x), int(y)

    def _sample_class_aware_position(
        self, exclude_class: int = -1, target_class: int = -1
    ):
        """Sample crop position biased toward under-represented classes.

        Args:
            exclude_class: Class to avoid as target (used by class-aware
                CutMix to ensure diversity). -1 = no exclusion.
            target_class: If >= 0, forces sampling of this specific class
                (used by CutMix rare boost). -1 = sample normally.

        Returns:
            Tuple[int, int, int]: (img_idx, x, y) position for crop.
        """
        crop_h, crop_w = self.crop_size

        if target_class >= 0 and target_class in self._class_positions:
            # Force specific class
            pass
        else:
            # Copy weights and zero out excluded class
            weights = self._class_sampling_weights.copy()
            if 0 <= exclude_class < len(weights):
                weights[exclude_class] = 0.0
                total = weights.sum()
                if total > 0:
                    weights /= total
                else:
                    # Fallback: if there was only one class, use original weights
                    weights = self._class_sampling_weights

            # Sample target class
            target_class = np.random.choice(self.n_classes, p=weights)

        # Sample position for this class
        positions = self._class_positions[target_class]
        pos_idx = np.random.randint(len(positions))
        img_idx, col, row = positions[pos_idx]

        # Jitter to avoid grid artifacts (+-crop_size//4)
        jitter_range = max(1, crop_h // 4)
        w, h = self.image_dims[img_idx]
        col += np.random.randint(-jitter_range, jitter_range + 1)
        row += np.random.randint(-jitter_range, jitter_range + 1)
        col = np.clip(col, 0, w - crop_w)
        row = np.clip(row, 0, h - crop_h)

        return img_idx, int(col), int(row)

    def _get_dominant_class(self, mask: np.ndarray) -> int:
        """Return the class with the most pixels in the crop (excluding ignore).

        Args:
            mask: Crop mask (H, W) integers, or (H, W, C) soft labels.

        Returns:
            Index of the dominant class, or -1 if there are no valid pixels.
        """
        if self.soft_labels:
            # Soft labels: (H, W, C) float — ignore pixels have sum == 0
            pixel_sums = mask.sum(axis=-1)  # (H, W)
            valid = pixel_sums > 0
            if not valid.any():
                return -1
            hard = np.argmax(mask, axis=-1)  # (H, W)
            counts = np.bincount(hard[valid], minlength=self.n_classes)
            return int(np.argmax(counts))
        else:
            valid = mask[mask < self.n_classes]
            if valid.size == 0:
                return -1
            counts = np.bincount(valid, minlength=self.n_classes)
            return int(np.argmax(counts))

    def _apply_cutmix(self, image, mask, image2, mask2):
        """Apply CutMix: paste a rectangular region from crop2 onto crop1.

        The cut size is sampled from Beta(alpha, alpha). The center position
        is random. Both image and mask are mixed.

        Args:
            image: Main crop (H, W, C).
            mask: Main mask (H, W) or (H, W, C) soft.
            image2: Second crop (H, W, C).
            mask2: Second mask (H, W) or (H, W, C) soft.

        Returns:
            Tuple[np.ndarray, np.ndarray]: mixed (image, mask).
        """
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        h, w = image.shape[:2]

        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = max(1, int(h * cut_ratio))
        cut_w = max(1, int(w * cut_ratio))

        # Random center
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        image = image.copy()
        mask = mask.copy()
        image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
        mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]

        self._last_cutmix_box = (y1, y2, x1, x2)
        return image, mask

    def _apply_classmix(self, image, mask, image2, mask2):
        """ClassMix: copy class-shaped regions from crop2 onto crop1.

        Instead of rectangles (CutMix), copies pixels whose class in mask2
        belongs to a randomly sampled subset. Preserves semantic boundaries
        and is more coherent for segmentation.

        Ref: Olsson et al., "ClassMix", WACV 2021.

        Args:
            image: Main crop (H, W, C).
            mask: Main mask (H, W) or (H, W, C) soft.
            image2: Second crop (H, W, C).
            mask2: Second mask (H, W) or (H, W, C) soft.

        Returns:
            Tuple[np.ndarray, np.ndarray]: mixed (image, mask).
        """
        # Convert to hard labels for class selection
        if self.soft_labels:
            pixel_sums = mask2.sum(axis=-1)
            valid2 = pixel_sums > 0
            hard2 = np.argmax(mask2, axis=-1)
        else:
            hard2 = mask2
            valid2 = mask2 < self.n_classes

        unique_classes = np.unique(hard2[valid2])
        if len(unique_classes) < 2:
            # Fallback to rectangular CutMix if only 1 class
            return self._apply_cutmix(image, mask, image2, mask2)

        # Select ~half of the classes to copy
        n_select = max(1, len(unique_classes) // 2)
        selected = np.random.choice(unique_classes, n_select, replace=False)

        # Binary mask of pixels to copy
        copy_mask = np.zeros(hard2.shape, dtype=bool)
        for c in selected:
            copy_mask |= (hard2 == c) & valid2

        if not copy_mask.any():
            return image, mask

        image = image.copy()
        mask = mask.copy()
        image[copy_mask] = image2[copy_mask]
        mask[copy_mask] = mask2[copy_mask]

        return image, mask

    def _is_crop_valid(
        self, image_data: np.ndarray, mask_data: np.ndarray = None
    ) -> bool:
        """Check if the crop has enough valid pixels.

        Args:
            image_data: Image crop (H, W, C) or (H, W).
            mask_data: Mask crop (H, W) or (H, W, C) for soft labels.
        """
        if image_data.size == 0:
            return False
        if len(image_data.shape) == 3:
            valid_pixels = np.sum(np.any(image_data > 0, axis=2))
        else:
            valid_pixels = np.sum(image_data > 0)
        total_pixels = image_data.shape[0] * image_data.shape[1]
        if (valid_pixels / total_pixels) < self.min_valid_ratio:
            return False

        # For soft labels, check mask validity (ignore pixels have sum == 0)
        if self.soft_labels and mask_data is not None:
            valid_mask_pixels = np.sum(mask_data.sum(axis=-1) > 0)
            if valid_mask_pixels / total_pixels < self.min_valid_ratio:
                return False

        return True

    def _get_random_crop(self, exclude_class: int = -1, target_class: int = -1):
        """Get a validated crop (image + mask).

        If class_balanced_sampling is active, samples biased toward
        under-represented classes. Otherwise, uses area-based sampling.

        Args:
            exclude_class: Class to avoid as target in class-aware
                sampling (used by CutMix). -1 = no exclusion.
            target_class: If >= 0, forces sampling of this specific
                class (used by CutMix rare boost). -1 = sample normally.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (image, mask), with
                image shape (H, W, C) and mask shape (H, W).
        """
        crop_h, crop_w = self.crop_size
        image = mask = None

        for _ in range(self.max_retries):
            if getattr(self, 'class_balanced_sampling', False) and getattr(self, '_class_positions', {}):
                img_idx, x, y = self._sample_class_aware_position(
                    exclude_class, target_class=target_class
                )
            elif target_class >= 0 and getattr(self, '_class_to_images', {}):
                # Lightweight targeting: pick image containing target class
                img_idx, x, y = self._sample_image_with_class(target_class)
            else:
                # Original area-based sampling
                choice = np.random.choice(
                    len(self._valid_indices), p=self.image_weights
                )
                img_idx = self._valid_indices[choice]
                w, h = self.image_dims[img_idx]
                x = np.random.randint(0, w - crop_w + 1)
                y = np.random.randint(0, h - crop_h + 1)

            image_path = self.get_path(img_idx, key=self.image_key)
            image = self._read_crop(image_path, x, y, is_mask=False)

            mask_path = self.get_path(img_idx, key=self.mask_key)
            mask = self._read_crop(mask_path, x, y, is_mask=True)

            if self._is_crop_valid(image, mask_data=mask):
                break
        # If all retries fail, use the last crop anyway

        if self.n_classes == 2 and not self.soft_labels:
            mask = (mask > 0).astype(np.uint8)

        self._last_crop_position = (img_idx, x, y)
        return image, mask

    def _get_random_grid_tile(self, target_class: int = -1, exclude_class: int = -1):
        """Get a random grid tile, optionally targeting or excluding a class.

        Uses _class_to_tiles index for efficient class-targeted sampling.
        Falls back to random tile selection if no index is available.

        Args:
            target_class: If >= 0, sample a tile containing this class.
            exclude_class: If >= 0, prefer tiles NOT dominated by this class.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (image, mask).
        """
        c2t = getattr(self, '_class_to_tiles', {})

        if target_class >= 0 and target_class in c2t and len(c2t[target_class]) > 0:
            tile_idx = int(np.random.choice(c2t[target_class]))
        elif exclude_class >= 0:
            # Try to find a tile not dominated by exclude_class
            # Use all tiles as candidates (fast path)
            tile_idx = int(np.random.randint(len(self._grid_positions)))
        else:
            tile_idx = int(np.random.randint(len(self._grid_positions)))

        return self._get_grid_crop(tile_idx)

    def _get_grid_crop(self, idx: int):
        """Get a pre-computed grid tile by index. No retries needed.

        Args:
            idx: Index into self._grid_positions.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (image, mask).
        """
        img_idx, x, y = self._grid_positions[idx]
        img_idx, x, y = int(img_idx), int(x), int(y)

        image_path = self.get_path(img_idx, key=self.image_key)
        image = self._read_crop(image_path, x, y, is_mask=False)

        mask_path = self.get_path(img_idx, key=self.mask_key)
        mask = self._read_crop(mask_path, x, y, is_mask=True)

        if self.n_classes == 2 and not self.soft_labels:
            mask = (mask > 0).astype(np.uint8)

        self._last_crop_position = (img_idx, x, y)
        return image, mask

    def _read_hard_mask_crop(self):
        """Read hard mask crop at the same position as the last soft mask crop.

        Uses _last_crop_position set by _get_random_crop / _get_grid_crop
        and _hard_mask_paths for the file path.  Returns (H, W) uint8.
        Delegates to _read_crop with soft_labels temporarily disabled.
        """
        if not hasattr(self, '_last_crop_position') or self._hard_mask_paths is None:
            return None
        img_idx, x, y = self._last_crop_position
        hard_path = self._hard_mask_paths[img_idx]
        # Temporarily disable soft_labels so _read_crop reads single-band uint8
        orig = self.soft_labels
        self.soft_labels = False
        result = self._read_crop(hard_path, x, y, is_mask=True)
        self.soft_labels = orig
        return result

    @staticmethod
    def _to_hard_mask_tensor(hard_mask):
        """Convert hard mask to long tensor, handling both numpy and torch."""
        if isinstance(hard_mask, np.ndarray):
            return torch.from_numpy(hard_mask.copy()).long()
        return hard_mask.clone().detach().long()

    def _get_transform_with_hard_mask(self):
        """Get (or build once) transform with additional_targets for hard_mask."""
        if self._transform_with_hard_mask is None and self.transform is not None:
            self._transform_with_hard_mask = A.Compose(
                self.transform.transforms,
                additional_targets={"hard_mask": "mask"},
            )
        return self._transform_with_hard_mask

    def _ensure_soft_mask_tensor(self, mask):
        """Ensure soft mask is a float tensor (C, H, W).

        After albumentations, mask can be np.ndarray (H,W,C) or
        torch.Tensor in different layouts. This method normalizes
        to (C, H, W) float32.
        """
        if isinstance(mask, np.ndarray):
            # (H, W, C) numpy → (C, H, W) tensor
            return torch.from_numpy(mask.copy()).float().permute(2, 0, 1)
        elif isinstance(mask, torch.Tensor):
            if mask.dim() == 3 and mask.shape[-1] == self.n_classes:
                # (H, W, C) → (C, H, W)
                return mask.permute(2, 0, 1).float()
            elif mask.dim() == 3 and mask.shape[0] == self.n_classes:
                # Already (C, H, W)
                return mask.float()
            else:
                return mask.float()
        return mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Get main crop: grid (deterministic) or random
        if self.grid_mode:
            image, mask = self._get_grid_crop(idx)
        else:
            image, mask = self._get_random_crop()

        # Read hard mask at same crop position (for dual-head training)
        hard_mask = None
        if self.return_hard_mask and self._hard_mask_paths is not None:
            hard_mask = self._read_hard_mask_crop()

        # Integrated CutMix/ClassMix: second crop seeks a DIFFERENT class from
        # the dominant one in the first, maximizing per-sample diversity.
        # ClassMix copies class-shaped regions (semantic) instead of
        # rectangles (CutMix). Ref: Olsson et al., WACV 2021.
        #
        # cutmix_rare_classes: with prob cutmix_rare_prob, forces the second crop
        # to contain a specific rare class (e.g., bare_soil), ensuring that
        # extremely under-represented classes appear frequently via
        # CutMix/ClassMix even when not sampled normally.
        # Apply color augmentation before mixing (and before main augmentation).
        # When mixing is active, each crop gets independent color aug to simulate
        # different radiometric conditions. When not mixing, the single crop
        # still gets color aug once — ensuring all samples see it exactly once.
        if getattr(self, '_pre_mix_transform', None) is not None:
            r1 = self._pre_mix_transform(image=image, mask=mask)
            image, mask = r1["image"], r1["mask"]

        if getattr(self, 'cutmix_prob', 0.0) > 0 and np.random.random() < self.cutmix_prob:
            main_class = self._get_dominant_class(mask)
            rare_classes = getattr(self, 'cutmix_rare_classes', [])
            rare_prob = getattr(self, 'cutmix_rare_prob', 0.5)
            if rare_classes and np.random.random() < rare_prob:
                rare_w = getattr(self, 'cutmix_rare_weights', None)
                target = int(np.random.choice(rare_classes, p=rare_w))
                if self.grid_mode:
                    image2, mask2 = self._get_random_grid_tile(target_class=target)
                else:
                    image2, mask2 = self._get_random_crop(target_class=target)
            else:
                if self.grid_mode:
                    image2, mask2 = self._get_random_grid_tile(exclude_class=main_class)
                else:
                    image2, mask2 = self._get_random_crop(exclude_class=main_class)

            # Read hard mask for second crop (CutMix/ClassMix)
            hard_mask2 = None
            if self.return_hard_mask and self._hard_mask_paths is not None:
                hard_mask2 = self._read_hard_mask_crop()

            # Independent color aug on second crop
            if getattr(self, '_pre_mix_transform', None) is not None:
                r2 = self._pre_mix_transform(image=image2, mask=mask2)
                image2, mask2 = r2["image"], r2["mask"]
            if getattr(self, 'classmix', False):
                image, mask = self._apply_classmix(image, mask, image2, mask2)
                # Apply same ClassMix mask to hard_mask
                if hard_mask is not None and hard_mask2 is not None:
                    # ClassMix uses dominant class region from mask2
                    dom = self._get_dominant_class(mask2)
                    if self.soft_labels:
                        cls_mask = np.argmax(mask2, axis=-1) == dom
                    else:
                        cls_mask = mask2 == dom
                    hard_mask = np.where(cls_mask, hard_mask2, hard_mask)
            else:
                image, mask = self._apply_cutmix(image, mask, image2, mask2)
                # Apply same CutMix box to hard_mask
                if hard_mask is not None and hard_mask2 is not None:
                    # _apply_cutmix stores the box in self._last_cutmix_box
                    if hasattr(self, '_last_cutmix_box'):
                        y1, y2, x1, x2 = self._last_cutmix_box
                        hard_mask[y1:y2, x1:x2] = hard_mask2[y1:y2, x1:x2]

        if self.transform is None:
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            image = image / 255.0
            if self.soft_labels:
                mask = torch.from_numpy(mask.copy()).float()
                mask = mask.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            else:
                mask = torch.from_numpy(mask).long()
            result = {"image": image, "mask": mask}
            if hard_mask is not None:
                result["hard_mask"] = self._to_hard_mask_tensor(hard_mask)
            return result

        # Apply augmentation (with or without hard_mask as additional target)
        if not self.reset_augmentation_function:
            if hard_mask is not None:
                result = self._get_transform_with_hard_mask()(
                    image=image, mask=mask, hard_mask=hard_mask)
                result["hard_mask"] = self._to_hard_mask_tensor(result["hard_mask"])
            else:
                result = self.transform(image=image, mask=mask)
            if self.soft_labels:
                result["mask"] = self._ensure_soft_mask_tensor(result["mask"])
            return result

        transform_func = deepcopy(self.transform)
        if hard_mask is not None:
            transform_func = A.Compose(
                transform_func.transforms,
                additional_targets={"hard_mask": "mask"},
            )
            output = transform_func(image=image, mask=mask, hard_mask=hard_mask)
        else:
            output = transform_func(image=image, mask=mask)
        output_dict = {
            "image": output["image"].clone().detach()
            if torch.is_tensor(output["image"])
            else output["image"].copy(),
            "mask": output["mask"].clone().detach()
            if torch.is_tensor(output["mask"])
            else output["mask"].copy(),
        }
        if hard_mask is not None:
            output_dict["hard_mask"] = self._to_hard_mask_tensor(output["hard_mask"])
        if self.soft_labels:
            output_dict["mask"] = self._ensure_soft_mask_tensor(output_dict["mask"])
        del output
        del transform_func
        return output_dict


class FrameFieldSegmentationDataset(SegmentationDataset):
    def __init__(
        self,
        input_csv_path: Path,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        mask_key=None,
        multi_band_mask=False,
        boundary_mask_key=None,
        return_boundary_mask=True,
        vertex_mask_key=None,
        return_vertex_mask=True,
        n_first_rows_to_read=None,
        return_crossfield_mask=True,
        crossfield_mask_key=None,
        return_distance_mask=True,
        distance_mask_key=None,
        return_size_mask=True,
        size_mask_key=None,
        image_width=224,
        image_height=224,
        gpu_augmentation_list=None,
    ) -> None:
        mask_key = "polygon_mask" if mask_key is None else mask_key
        super(FrameFieldSegmentationDataset, self).__init__(
            input_csv_path,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            mask_key=mask_key,
            n_first_rows_to_read=n_first_rows_to_read,
        )
        self.multi_band_mask = multi_band_mask
        self.boundary_mask_key = (
            boundary_mask_key if boundary_mask_key is not None else "boundary_mask"
        )
        self.vertex_mask_key = (
            vertex_mask_key if vertex_mask_key is not None else "vertex_mask"
        )
        self.return_crossfield_mask = return_crossfield_mask
        self.return_distance_mask = return_distance_mask
        self.return_size_mask = return_size_mask
        self.crossfield_mask_key = (
            crossfield_mask_key
            if crossfield_mask_key is not None
            else "crossfield_mask"
        )
        self.distance_mask_key = (
            distance_mask_key if distance_mask_key is not None else "distance_mask"
        )
        self.size_mask_key = size_mask_key if size_mask_key is not None else "size_mask"
        self.alternative_transform = A.Compose(
            [A.Resize(image_height, image_width), A.Normalize(), ToTensorV2()]
        )
        self.masks_to_load_dict = {
            mask_key: True,
            self.boundary_mask_key: return_boundary_mask,
            self.vertex_mask_key: return_vertex_mask,
        }

    def load_masks(self, idx):
        if self.multi_band_mask:
            multi_band_mask = self.load_image(idx, key=self.mask_key, is_mask=True)
            mask_dict = {
                self.mask_key: multi_band_mask[:, :, 0],
                self.boundary_mask_key: multi_band_mask[:, :, 1],
                self.vertex_mask_key: multi_band_mask[:, :, 2],
            }
        else:
            mask_dict = {
                mask_key: self.load_image(idx, key=mask_key, is_mask=True)
                for mask_key, load_mask in self.masks_to_load_dict.items()
                if load_mask
            }
        if self.return_crossfield_mask:
            mask_dict[self.crossfield_mask_key] = self.load_image(
                idx, key=self.crossfield_mask_key, is_mask=False
            )
        if self.return_distance_mask:
            mask_dict[self.distance_mask_key] = self.load_image(
                idx, key=self.distance_mask_key, is_mask=False
            )
        if self.return_size_mask:
            mask_dict[self.size_mask_key] = self.load_image(
                idx, key=self.size_mask_key, is_mask=False
            )
        return mask_dict

    def compute_class_freq(self, gt_polygons_image):
        pass

    def get_mean_axis(self, mask):
        if len(mask.shape) > 2:
            return tuple(
                [
                    idx
                    for idx, shape in enumerate(mask.shape)
                    if shape != min(mask.shape)
                ]
            )
        elif len(mask.shape) == 2:
            return (0, 1)
        else:
            return 0

    def is_valid_crop(self, ds_item_dict):
        gt_polygons_mask = (0 < ds_item_dict["gt_polygons_image"]).float()
        background_freq = 1 - torch.sum(ds_item_dict["class_freq"], dim=0)
        pixel_class_freq = (
            gt_polygons_mask * ds_item_dict["class_freq"][:, None, None]
            + (1 - gt_polygons_mask) * background_freq[None, None, None]
        )
        if pixel_class_freq.min() == 0 or (
            "sizes" in ds_item_dict and ds_item_dict["sizes"].min() == 0
        ):
            return False
        return True

    def build_ds_item_dict(self, idx, transformed):
        gt_polygons_image = self.to_tensor(
            np.stack(
                [
                    transformed["masks"][0],
                    transformed["masks"][1],
                    transformed["masks"][2],
                ],
                axis=0,
            )
        ).float()
        ds_item_dict = {
            "idx": idx,
            "path": self.get_path(idx),
            "image": self.to_tensor(transformed["image"]),
            "gt_polygons_image": gt_polygons_image,
            "class_freq": torch.mean(gt_polygons_image, axis=(1, 2)),
        }
        return ds_item_dict

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.multi_band_mask:
            return super().__getitem__(idx)
        image = self.load_image(idx, force_rgb=True)
        mask_dict = self.load_masks(idx)
        if self.transform is None:
            ds_item_dict = {
                "idx": idx,
                "path": self.get_path(idx),
                "image": self.to_tensor(image),
                "gt_polygons_image": self.to_tensor(
                    np.stack(
                        [
                            mask_dict[key]
                            for key, load_key in self.masks_to_load_dict.items()
                            if load_key
                        ],
                        axis=-1,
                    )
                ),
                "class_freq": self.to_tensor(
                    np.mean(
                        mask_dict[self.mask_key],
                        axis=self.get_mean_axis(mask_dict[self.mask_key]),
                    )
                    / 255
                    if "class_freq" not in self.df.columns
                    else self.to_tensor(
                        np.fromstring(
                            self.df.iloc[idx]["class_freq"]
                            .replace("[", "")
                            .replace("]", ""),
                            sep=" ",
                        )
                    )
                ),
            }
            if self.return_crossfield_mask:
                ds_item_dict["gt_crossfield_angle"] = (
                    self.to_tensor(mask_dict[self.crossfield_mask_key])
                    .float()
                    .unsqueeze(0)
                )
            if self.return_distance_mask:
                ds_item_dict["distances"] = (
                    self.to_tensor(mask_dict[self.distance_mask_key])
                    .float()
                    .unsqueeze(0)
                )
            if self.return_size_mask:
                ds_item_dict["sizes"] = (
                    self.to_tensor(mask_dict[self.size_mask_key]).float().unsqueeze(0)
                )
            return ds_item_dict
        transformed = self.transform(image=image, masks=list(mask_dict.values()))
        ds_item_dict = self.build_ds_item_dict(idx, transformed)
        if not self.is_valid_crop(ds_item_dict):
            transformed = self.alternative_transform(
                image=image, masks=list(mask_dict.values())
            )
            ds_item_dict = self.build_ds_item_dict(idx, transformed)

        mask_idx = sum(self.masks_to_load_dict.values())
        if self.return_crossfield_mask:
            ds_item_dict["gt_crossfield_angle"] = (
                self.to_tensor(transformed["masks"][mask_idx]).float().unsqueeze(0)
            )
            mask_idx += 1
        if self.return_distance_mask:
            ds_item_dict["distances"] = (
                self.to_tensor(transformed["masks"][mask_idx]).float().unsqueeze(0)
            )
            mask_idx += 1
        if self.return_size_mask:
            ds_item_dict["sizes"] = (
                self.to_tensor(transformed["masks"][mask_idx]).float().unsqueeze(0)
            )
        return ds_item_dict


class PolygonRNNDataset(AbstractDataset):
    def __init__(
        self,
        input_csv_path: Path,
        sequence_length: int = 60,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key: str = None,
        mask_key: str = None,
        image_width_key: str = None,
        image_height_key: str = None,
        scale_h_key: str = None,
        scale_w_key: str = None,
        min_col_key: str = None,
        min_row_key: str = None,
        original_image_path_key: str = None,
        original_polygon_key: str = None,
        n_first_rows_to_read: str = None,
        dataset_type: str = "train",
        grid_size=28,
    ) -> None:
        super(PolygonRNNDataset, self).__init__(
            input_csv_path=input_csv_path,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            mask_key=mask_key,
            n_first_rows_to_read=n_first_rows_to_read,
        )
        self.sequence_length = sequence_length
        self.scale_h_key = scale_h_key if scale_h_key is not None else "scale_h"
        self.scale_w_key = scale_w_key if scale_w_key is not None else "scale_w"
        self.min_col_key = min_col_key if min_col_key is not None else "min_col"
        self.min_row_key = min_row_key if min_row_key is not None else "min_row"
        self.original_image_path_key = (
            original_image_path_key
            if original_image_path_key is not None
            else "original_image_path"
        )
        self.original_polygon_key = (
            original_polygon_key
            if original_polygon_key is not None
            else "original_polygon_wkt"
        )
        if dataset_type not in ["train", "val"]:
            raise NotImplemented
        self.dataset_type = dataset_type
        if dataset_type == "val":
            self.unique_image_path_list = self.df[self.original_image_path_key].unique()
        self.grid_size = grid_size

    def load_polygon(self, idx: int) -> Tuple[np.ndarray, int]:
        mask_name = os.path.join(self.root_dir, self.df.iloc[idx][self.mask_key])
        return self._load_polygon_from_path(mask_name)

    def _load_polygon_from_path(self, mask_name: str) -> Tuple[np.ndarray, int]:
        with open(mask_name, "r") as f:
            json_file = json.load(f)
        polygon = np.array(json_file["polygon"])
        point_num = len(json_file["polygon"])
        return polygon, point_num

    def __getitem__(
        self,
        index: int,
        load_images: bool = True,
        get_bbox: bool = False,
        original_image_bounds: Optional[Tuple] = None,
    ) -> Dict[str, Any]:
        polygon, num_vertexes = self.load_polygon(index)
        label_array, label_index_array = polygonrnn_utils.build_arrays(
            polygon, num_vertexes, self.sequence_length, grid_size=self.grid_size
        )
        output_dict: Dict[str, Any] = {
            "x1": self.to_tensor(label_array[2]).float(),
            "x2": self.to_tensor(label_array[:-2]).float(),
            "x3": self.to_tensor(label_array[1:-1]).float(),
            "ta": self.to_tensor(label_index_array[2:]).long(),
        }
        if get_bbox:
            img_bounds = (
                (300, 300) if original_image_bounds is None else original_image_bounds
            )
            bbox = polygonrnn_utils.get_extended_bounds_from_np_array_polygon(
                polygon, img_bounds, extend_factor=0.1
            )
            output_dict.update({"bbox": torch.floor(torch.tensor(bbox))})
        if self.dataset_type == "val":
            output_dict.update(
                {
                    "polygon_wkt": self.df.iloc[index][self.original_polygon_key],
                    "scale_h": self.df.iloc[index][self.scale_h_key],
                    "scale_w": self.df.iloc[index][self.scale_w_key],
                    "min_col": self.df.iloc[index][self.min_col_key],
                    "min_row": self.df.iloc[index][self.min_row_key],
                    "original_image_path": self.get_path(
                        index, key=self.original_image_path_key
                    ),
                }
            )
        if load_images:
            image = self.load_image(
                index, key=self.image_key, is_mask=False, force_rgb=True
            )
            if self.transform is not None:
                augmented = self.transform(image=image)
                image = augmented["image"]
            output_dict.update({"image": self.to_tensor(image).float()})
        return output_dict

    def get_images_from_image_path(self, image_path: str) -> Dict[str, Any]:
        entries, items = self.get_training_images_from_image_path(image_path)
        return {
            "original_image": self.load_image(
                entries.index.tolist()[0], key=self.original_image_path_key
            ),
            "shapely_polygon_list": [
                shapely.wkt.loads(item["polygon_wkt"]) for item in items
            ],
            "croped_images": torch.stack([item["image"] for item in items]),
            "scale_h": torch.tensor([item["scale_h"] for item in items]),
            "scale_w": torch.tensor([item["scale_w"] for item in items]),
            "min_col": torch.tensor([item["min_col"] for item in items]),
            "min_row": torch.tensor([item["min_row"] for item in items]),
            "ta": torch.stack([item["ta"] for item in items]),
        }

    def get_entries_from_image_path(self, image_path: str) -> pd.DataFrame:
        return self.df.loc[self.df[self.original_image_path_key] == image_path]

    def get_training_images_from_image_path(
        self, image_path: str
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        entries = self.get_entries_from_image_path(image_path)
        items = [self.__getitem__(idx) for idx in entries.index.tolist()]
        return entries, items

    def get_n_image_path_lists(self, n: int) -> List:
        return list(
            itertools.chain.from_iterable(
                self.get_images_from_image_path(image_path)
                for image_path in self.unique_image_path_list[:n]
            )
        )

    def get_n_image_path_dict_list(self, n: int) -> Dict[str, Any]:
        output_dict = {
            image_path: self.get_images_from_image_path(image_path)
            for image_path in self.unique_image_path_list[:n]
        }
        return output_dict


class ObjectDetectionDataset(AbstractDataset):
    def __init__(
        self,
        input_csv_path: Path,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        mask_key=None,
        bounding_box_key=None,
        n_first_rows_to_read=None,
        bbox_format="xywh",
        bbox_output_format="xyxy",
        bbox_params=None,
    ) -> None:
        super(ObjectDetectionDataset, self).__init__(
            input_csv_path=input_csv_path,
            root_dir=root_dir,
            augmentation_list=None,
            data_loader=data_loader,
            image_key=image_key,
            mask_key=mask_key,
            n_first_rows_to_read=n_first_rows_to_read,
        )
        self.transform = (
            None
            if augmentation_list is None
            else load_augmentation_object(augmentation_list, bbox_params=bbox_params)
        )
        self.bounding_box_key = (
            "bounding_boxes" if bounding_box_key is None else bounding_box_key
        )
        self.bbox_format = bbox_format
        self.bbox_output_format = bbox_output_format

    def convert_bbox(self, bbox: List) -> List:
        if self.bbox_format == self.bbox_output_format:
            return bbox
        elif self.bbox_format == "xywh" and self.bbox_output_format == "xyxy":
            return bbox_xywh_to_xyxy(bbox)
        elif self.bbox_format == "xyxy" and self.bbox_output_format == "xywh":
            return bbox_xyxy_to_xywh(bbox)
        else:
            raise NotImplementedError

    def load_bounding_boxes_and_labels(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bbox_path = self.get_path(idx, key=self.bounding_box_key)
        with open(bbox_path, "r") as f:
            json_file = json.load(f)
        bbox_list, label_list = [], []
        for box_item in json_file:
            bbox_list.append(box_item["bbox"])
            label_list.append(box_item["class"])
        return (
            torch.as_tensor(bbox_list, dtype=torch.float32),
            torch.as_tensor(label_list, dtype=torch.int64),
        )

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
        image = self.load_image(
            index, key=self.image_key, is_mask=False, force_rgb=True
        )
        bbox_list, label_list = self.load_bounding_boxes_and_labels(index)
        ds_item_dict = {"image": image, "bboxes": bbox_list, "labels": label_list}
        if self.transform is not None:
            ds_item_dict = self.transform(**ds_item_dict)
        image = ds_item_dict.pop("image")
        ds_item_dict["boxes"] = torch.as_tensor(
            [self.convert_bbox(bbox) for bbox in ds_item_dict.pop("bboxes")],
            dtype=torch.float32,
        )
        ds_item_dict["labels"] = torch.as_tensor(
            ds_item_dict["labels"], dtype=torch.int64
        )
        return image, ds_item_dict, index

    @staticmethod
    def collate_fn(
        batch: List,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images, targets, indexes = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        indexes = torch.tensor(indexes, dtype=torch.int64)
        return images, list(targets), indexes


class InstanceSegmentationDataset(ObjectDetectionDataset):
    def __init__(
        self,
        input_csv_path: Path,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        mask_key=None,
        bounding_box_key=None,
        keypoint_key=None,
        n_first_rows_to_read=None,
        bbox_format="xywh",
        bbox_output_format="xyxy",
        return_mask=True,
        return_keypoints=False,
        bbox_params=None,
    ) -> None:
        mask_key = "polygon_mask" if mask_key is None else mask_key
        super(InstanceSegmentationDataset, self).__init__(
            input_csv_path=input_csv_path,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            mask_key=mask_key,
            bounding_box_key=bounding_box_key,
            n_first_rows_to_read=n_first_rows_to_read,
            bbox_format=bbox_format,
            bbox_output_format=bbox_output_format,
            bbox_params=bbox_params,
        )
        self.return_mask = return_mask
        self.return_keypoints = return_keypoints
        self.keypoint_key = "keypoints" if keypoint_key is None else keypoint_key

    def load_keypoints(self, idx):
        keypoints_path = self.get_path(idx, key=self.keypoint_key)
        with open(keypoints_path, "r") as f:
            json_file = json.load(f)
        return torch.as_tensor(json_file[self.keypoint_key], dtype=torch.float32)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
        image = self.load_image(
            index, key=self.image_key, is_mask=False, force_rgb=True
        )
        bbox_list, label_list = self.load_bounding_boxes_and_labels(index)
        ds_item_dict = {"image": image, "bboxes": bbox_list, "labels": label_list}
        if self.return_mask:
            ds_item_dict["masks"] = [
                self.load_image(index, key=self.mask_key, is_mask=True)
            ]
        if self.return_keypoints:
            ds_item_dict["keypoints"] = self.load_keypoints(index)
        if self.transform is not None:
            ds_item_dict = self.transform(**ds_item_dict)
        image = ds_item_dict.pop("image")
        ds_item_dict["boxes"] = torch.as_tensor(
            [self.convert_bbox(bbox) for bbox in ds_item_dict.pop("bboxes")],
            dtype=torch.float32,
        )
        ds_item_dict["labels"] = torch.as_tensor(
            ds_item_dict["labels"], dtype=torch.int64
        )
        if self.return_mask:
            ds_item_dict["masks"] = torch.as_tensor(
                ds_item_dict["masks"], dtype=torch.uint8
            )
        return image, ds_item_dict, index


class NaiveModPolyMapperDataset(Dataset):
    def __init__(
        self,
        object_detection_dataset: ObjectDetectionDataset,
        polygon_rnn_dataset: PolygonRNNDataset,
    ) -> None:
        super(NaiveModPolyMapperDataset, self).__init__()
        self.object_detection_dataset = object_detection_dataset
        self.polygon_rnn_dataset = polygon_rnn_dataset

    def __len__(self) -> int:
        return len(self.object_detection_dataset)

    def get_polygonrnn_data(self, idx: int) -> Dict[str, Any]:
        (
            _,
            polygon_rnn_data,
        ) = self.polygon_rnn_dataset.get_training_images_from_image_path(
            self.object_detection_dataset.get_path(
                idx, key=self.object_detection_dataset.image_key, add_root_dir=False
            )
        )
        return {"polygon_rnn_data": polygon_rnn_data}

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any], int]:
        image, ds_item_dict, _ = self.object_detection_dataset[index]
        ds_item_dict.update(self.get_polygonrnn_data(index))
        return image, ds_item_dict, index

    @staticmethod
    def collate_fn(
        batch: List,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images, targets, indexes = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        indexes = torch.tensor(indexes, dtype=torch.int64)
        return images, list(targets), indexes


class ModPolyMapperDataset(NaiveModPolyMapperDataset):
    def __init__(
        self,
        object_detection_dataset: ObjectDetectionDataset,
        polygon_rnn_dataset: PolygonRNNDataset,
    ) -> None:
        super(ModPolyMapperDataset, self).__init__(
            object_detection_dataset=object_detection_dataset,
            polygon_rnn_dataset=polygon_rnn_dataset,
        )

    def get_polygonrnn_polygon_data(self, idx: int) -> List[Any]:
        image_path = self.object_detection_dataset.get_path(
            idx, key=self.object_detection_dataset.image_key, add_root_dir=False
        )
        entries = self.polygon_rnn_dataset.get_entries_from_image_path(image_path)
        return [
            self.polygon_rnn_dataset.__getitem__(idx) for idx in entries.index.tolist()
        ]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], int]:
        image, ds_item_dict, _ = self.object_detection_dataset[index]
        polygonrnn_data = self.get_polygonrnn_polygon_data(index)
        dict_items_to_update = {
            key: torch.stack([torch.tensor(item[key]) for item in polygonrnn_data])
            for key in polygonrnn_data[0].keys()
        }
        croped_images = dict_items_to_update.pop("image")
        ds_item_dict.update(dict_items_to_update)
        return image, ds_item_dict, index

    @staticmethod
    def collate_fn(
        batch: List,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images, targets, indexes = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        indexes = torch.tensor(indexes, dtype=torch.int64)
        return images, list(targets), indexes


if __name__ == "__main__":
    pass
