# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-05-13
        git sha              : $Format:%H$
        copyright            : (C) 2026 by Philipe Borba - Cartographic Engineer
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

import bisect
import gc
import json
import logging
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import albumentations as A
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import IterableDataset, get_worker_info

from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    AbstractDataset,
    _DTYPE_NORMALIZATION,
    _RasterioLRUCache,
    _VALID_IMAGE_DTYPES,
    _rasterio_read_lock,
    load_augmentation_object,
)
from pytorch_toolbelt.inference.tiles import ImageSlicer
from pytorch_toolbelt.utils.torch_utils import image_to_tensor

logger = logging.getLogger(__name__)


class ImageDataset(AbstractDataset):
    """Dataset for image-only tasks.

    Args:
        input_csv_path: CSV path with an image column.
        df: Optional in-memory DataFrame with an image column.
        root_dir: Root directory prepended to relative image paths.
        augmentation_list: Albumentations transforms applied to the image.
        data_loader: DataLoader configuration stored for the Lightning model.
        image_key: Image path column name. Defaults to ``"image"``.
        n_first_rows_to_read: Optional row limit when reading CSV files.
        seed: Optional seed passed to the Albumentations pipeline.

    Returns:
        Dict[str, Any]: Items with ``image`` and ``path`` keys.

    Example YAML:
        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.ImageDataset
          input_csv_path: /data/images.csv
          root_dir: /data
    """

    def __init__(
        self,
        input_csv_path: Path = None,
        df=None,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        n_first_rows_to_read=None,
        seed=None,
    ) -> None:
        super(ImageDataset, self).__init__(
            input_csv_path=input_csv_path,
            df=df,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            n_first_rows_to_read=n_first_rows_to_read,
            seed=seed,
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
        """Build one dataset per DataFrame group."""
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
        """Return one image-only sample."""
        idx = idx % self.len

        image = self.load_image(idx, key=self.image_key)
        result = (
            {"image": image} if self.transform is None else self.transform(image=image)
        )
        result.update({"path": self.get_path(idx, key=self.image_key)})
        return result


class CSVWindowedImageDataset(ImageDataset):
    """Image dataset that reads configured windows from larger rasters.

    Args:
        input_csv_path: CSV with image path, row offset, column offset and patch
            size columns.
        df: Optional pre-built DataFrame.
        root_dir: Root directory prepended to relative paths.
        augmentation_list: Albumentations transforms applied to each window.
        data_loader: DataLoader configuration stored for the Lightning model.
        image_key: Image path column. Defaults to ``"image"``.
        row_off_key: Row-offset column. Defaults to ``"row_off"``.
        col_off_key: Column-offset column. Defaults to ``"col_off"``.
        patch_size_key: Patch-size column. Defaults to ``"patch_size"``.
        selected_bands: Optional 1-based rasterio bands to read.
        image_dtype: Output dtype for rasterio-loaded arrays.

    Returns:
        Dict[str, Any]: Items with ``image`` and ``path`` keys.

    Example YAML:
        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.CSVWindowedImageDataset
          input_csv_path: patches.csv
          row_off_key: row_off
          col_off_key: col_off
          patch_size_key: patch_size
    """

    def __init__(
        self,
        input_csv_path: Path = None,
        df: Optional[pd.DataFrame] = None,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key="image",
        row_off_key="row_off",
        col_off_key="col_off",
        patch_size_key="patch_size",
        n_first_rows_to_read=None,
        selected_bands: Optional[List[int]] = None,
        use_rasterio: bool = True,
        reset_augmentation_function: bool = False,
        image_dtype: str = "uint8",
    ) -> None:
        super().__init__(
            input_csv_path=input_csv_path,
            df=df,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            n_first_rows_to_read=n_first_rows_to_read,
        )
        self.row_off_key = row_off_key
        self.col_off_key = col_off_key
        self.patch_size_key = patch_size_key
        self.selected_bands = selected_bands
        self.use_rasterio = use_rasterio
        self.reset_augmentation_function = reset_augmentation_function

        if image_dtype not in _VALID_IMAGE_DTYPES:
            raise ValueError(
                f"image_dtype '{image_dtype}' é inválido. "
                f"Valores aceitos: {sorted(_VALID_IMAGE_DTYPES)}"
            )
        self.image_dtype = image_dtype

        for col in [self.row_off_key, self.col_off_key, self.patch_size_key]:
            if col not in self.df.columns:
                raise ValueError(
                    f"A coluna '{col}' é obrigatória no CSV/DataFrame para CSVWindowedImageDataset."
                )

    def load_image(
        self,
        idx: int,
        key: str = None,
        is_mask: bool = False,
        force_rgb: bool = False,
        is_binary_mask: bool = True,
    ) -> np.ndarray:
        """Load the configured image window for ``idx``."""
        idx = idx % self.len
        row = self.df.iloc[idx]
        image_path = self.get_path(idx, key=key)

        window = rasterio.windows.Window(
            col_off=row[self.col_off_key],
            row_off=row[self.row_off_key],
            width=row[self.patch_size_key],
            height=row[self.patch_size_key],
        )

        try:
            with rasterio.open(image_path) as src:
                data = (
                    src.read(window=window)
                    if self.selected_bands is None
                    else src.read(self.selected_bands, window=window)
                )
                image = np.transpose(data, (1, 2, 0)).copy()
                if self.image_dtype == "native":
                    return image
                return np.array(image, dtype=np.dtype(self.image_dtype))
        except (rasterio.errors.RasterioIOError, Exception) as e:
            logger.error(f"Error reading window {window} from {image_path}: {e}")
            # Try another index
            return self.load_image(
                (idx + 1) % self.len, key, is_mask, force_rgb, is_binary_mask
            )


class TiledInferenceImageDataset(ImageDataset):
    """Image-only dataset that returns tiles for sliding-window inference.

    Args:
        input_csv_path: CSV with image path and image dimensions.
        df: Optional pre-built DataFrame.
        root_dir: Root directory prepended to relative paths.
        data_loader: DataLoader configuration stored for the prediction model.
        image_key: Image path column.
        normalize_output: If True, applies ``albumentations.Normalize`` before
            slicing.
        pad_if_needed: If True, pads the image to a multiple of
            ``model_input_shape``.
        model_input_shape: Tile size used by the model.
        step_shape: Sliding-window step. Defaults to ``(224, 224)``.

    Returns:
        Dict[str, Any]: Item with tiles, tile indices, tiler object and metadata.

    Example YAML:
        predict_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.TiledInferenceImageDataset
          input_csv_path: /data/inference.csv
          model_input_shape: [512, 512]
    """

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
        """Return all model tiles for one source image."""
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
        """Collate tiled inference samples into a single tile batch."""
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


class AutoencoderDataset(ImageDataset):
    """Dataset for whole-image autoencoder training.

    The clean image is returned as ``target``. Optional corruption augmentations
    are applied only to ``image``, while the base augmentation pipeline is applied
    synchronously to both ``image`` and ``target``.

    Args:
        corruption_augmentation_list: Albumentations configs applied only to the
            input image before synchronized transforms.
        **kwargs: Parameters accepted by :class:`ImageDataset`, such as
            ``input_csv_path``, ``root_dir`` and ``augmentation_list``.

    Returns:
        Dict[str, Any]: Items with ``image``, ``target`` and ``path`` keys.

    Example YAML:
        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.AutoencoderDataset
          input_csv_path: /data/images.csv
          root_dir: /data
          augmentation_list:
            - _target_: albumentations.Resize
              height: 256
              width: 256
            - _target_: albumentations.pytorch.ToTensorV2
    """

    def __init__(
        self,
        *args,
        corruption_augmentation_list: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.corruption_transform = (
            None
            if corruption_augmentation_list is None
            else load_augmentation_object(corruption_augmentation_list)
        )
        if self.transform is not None:
            self.transform = A.Compose(
                self.transform.transforms, additional_targets={"target": "image"}
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a clean target and optionally corrupted input image."""
        idx = idx % self.len
        image = self.load_image(idx, key=self.image_key)
        target = image.copy()

        if self.corruption_transform is not None:
            image = self.corruption_transform(image=image)["image"]

        if self.transform is not None:
            res = self.transform(image=image, target=target)
            result = {"image": res["image"], "target": res["target"]}
        else:
            result = {"image": image, "target": target}

        result.update({"path": self.get_path(idx, key=self.image_key)})
        return result


class AutoencoderRandomCropDataset(ImageDataset):
    """Random windowed crop dataset for autoencoder reconstruction training.

    The dataset reads fixed-size crops directly from full-size rasters using
    rasterio windowed reads. It can be built either from the existing CSV/DataFrame
    contract or by recursively scanning an ``image_dir``. The clean crop is used
    as ``target``; optional corruption augmentations affect only ``image``.

    Args:
        image_dir: Root folder scanned recursively for images. Mutually optional
            with ``input_csv_path``/``df``; one source must be provided.
        image_extensions: Extensions to include when scanning ``image_dir``.
            Defaults to common raster/image extensions.
        split: ``"all"``, ``"train"`` or ``"val"`` for deterministic folder
            splitting.
        val_fraction: Fraction assigned to the validation split.
        split_seed: Seed used to shuffle discovered paths before splitting.
        crop_size: Crop size as ``[height, width]``.
        samples_per_epoch: Number of random crops per epoch. If ``<= 0``,
            computes roughly 3x coverage of the valid image area.
        selected_bands: Optional 1-based rasterio band indices.
        image_dtype: One of ``"uint8"``, ``"uint16"``, ``"float32"`` or
            ``"native"``.
        corruption_augmentation_list: Albumentations configs applied only to
            ``image``.
        **kwargs: Compatibility parameters accepted by ``ImageDataset`` and Hydra.

    Returns:
        Dict[str, Any]: Items with ``image``, ``target`` and ``path`` keys.

    Example YAML:
        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.AutoencoderRandomCropDataset
          image_dir: /data/unlabeled_images
          split: train
          crop_size: [256, 256]
          samples_per_epoch: 20000
          augmentation_list:
            - _target_: albumentations.Normalize
            - _target_: albumentations.pytorch.ToTensorV2
    """

    DEFAULT_IMAGE_EXTENSIONS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

    def __init__(
        self,
        input_csv_path: Optional[Union[str, Path]] = None,
        df: Optional[pd.DataFrame] = None,
        image_dir: Optional[Union[str, Path]] = None,
        image_extensions: Optional[Sequence[str]] = None,
        split: str = "all",
        val_fraction: float = 0.2,
        split_seed: int = 42,
        crop_size: Optional[List[int]] = None,
        samples_per_epoch: int = 10000,
        selected_bands: Optional[List[int]] = None,
        image_dtype: str = "uint8",
        file_cache_maxsize: int = 0,
        corruption_augmentation_list: Optional[List[Dict[str, Any]]] = None,
        reset_augmentation_function: bool = False,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key=None,
        n_first_rows_to_read=None,
        max_retries: int = 10,
        **kwargs,
    ) -> None:
        if crop_size is None:
            crop_size = [256, 256]
        if len(crop_size) != 2:
            raise ValueError("crop_size must be [height, width]")
        if split not in {"all", "train", "val"}:
            raise ValueError("split must be one of: 'all', 'train', 'val'")
        if not (0.0 < val_fraction < 1.0):
            raise ValueError("val_fraction must be in (0.0, 1.0)")
        if image_dtype not in _VALID_IMAGE_DTYPES:
            raise ValueError(
                f"image_dtype '{image_dtype}' is invalid. "
                f"Accepted values: {sorted(_VALID_IMAGE_DTYPES)}"
            )
        if selected_bands is not None:
            if not all(isinstance(b, int) and b > 0 for b in selected_bands):
                raise ValueError(
                    "selected_bands must contain only positive integers (1-based)"
                )

        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.image_extensions = [
            self._normalize_extension(ext).lower()
            for ext in (image_extensions or self.DEFAULT_IMAGE_EXTENSIONS)
        ]
        self.split = split
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.crop_size = crop_size
        self.selected_bands = selected_bands
        self.image_dtype = image_dtype
        self.reset_augmentation_function = reset_augmentation_function
        self.max_retries = max_retries
        self.extra_kwargs = kwargs

        if df is None and input_csv_path is None and self.image_dir is not None:
            df = self._build_dataframe_from_folder(
                self.image_dir,
                self.image_extensions,
                split=split,
                val_fraction=val_fraction,
                split_seed=split_seed,
            )
            root_dir = None
        elif df is None and input_csv_path is None:
            raise ValueError("Must provide either input_csv_path, df, or image_dir")

        super().__init__(
            input_csv_path=input_csv_path,
            df=df,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            n_first_rows_to_read=n_first_rows_to_read,
        )

        self.corruption_transform = (
            None
            if corruption_augmentation_list is None
            else load_augmentation_object(corruption_augmentation_list)
        )
        if self.transform is not None:
            self.transform = A.Compose(
                self.transform.transforms, additional_targets={"target": "image"}
            )

        n_files = len(self.df) if hasattr(self, "df") else 500
        self.file_cache_maxsize = (
            file_cache_maxsize if file_cache_maxsize > 0 else n_files + 16
        )
        self._file_cache = _RasterioLRUCache(maxsize=self.file_cache_maxsize)

        self.image_dims = []
        self._valid_indices = []
        crop_h, crop_w = self.crop_size
        for i in range(len(self.df)):
            image_path = self.get_path(i, key=self.image_key)
            with rasterio.open(image_path) as src:
                width, height = src.width, src.height
            self.image_dims.append((width, height))
            if width >= crop_w and height >= crop_h:
                self._valid_indices.append(i)

        if not self._valid_indices:
            raise ValueError(
                f"No image large enough for crop {self.crop_size}. "
                f"Dimensions found: {self.image_dims}"
            )

        positions = []
        for idx in self._valid_indices:
            width, height = self.image_dims[idx]
            positions.append((width - crop_w + 1) * (height - crop_h + 1))
        total_positions = sum(positions)
        self.image_weights = np.array([p / total_positions for p in positions])

        if samples_per_epoch <= 0:
            crop_area = crop_h * crop_w
            total_pixel_area = sum(
                self.image_dims[idx][0] * self.image_dims[idx][1]
                for idx in self._valid_indices
            )
            self.samples_per_epoch = max(1, int(3 * total_pixel_area / crop_area))
            logger.info(
                "AutoencoderRandomCropDataset: samples_per_epoch automatically "
                "calculated as %s",
                self.samples_per_epoch,
            )
        else:
            self.samples_per_epoch = samples_per_epoch

    def __getstate__(self):
        """Drop open raster handles before DataLoader worker pickling."""
        state = self.__dict__.copy()
        state["_file_cache"] = _RasterioLRUCache(maxsize=self.file_cache_maxsize)
        return state

    def __setstate__(self, state):
        """Restore dataset state with an empty per-worker raster cache."""
        self.__dict__.update(state)

    def __del__(self):
        """Close cached raster handles."""
        self._close_cache()

    def _close_cache(self):
        if hasattr(self, "_file_cache") and isinstance(
            self._file_cache, _RasterioLRUCache
        ):
            self._file_cache.close_all()

    def __len__(self) -> int:
        """Return the configured number of random crops per epoch."""
        return self.samples_per_epoch

    def get_path(self, idx: int, key: str = None, add_root_dir: bool = True):
        """Wrap random-crop indices to the underlying image table."""
        wrapped_idx = idx % len(self.df)
        return super().get_path(wrapped_idx, key=key, add_root_dir=add_root_dir)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a random crop pair for reconstruction training."""
        image = self._get_random_crop()
        target = image.copy()

        if self.corruption_transform is not None:
            image = self.corruption_transform(image=image)["image"]

        if self.transform is not None:
            if not self.reset_augmentation_function:
                result = self.transform(image=image, target=target)
            else:
                transform_func = deepcopy(self.transform)
                result = transform_func(image=image, target=target)
                del transform_func
                if idx % 100 == 0:
                    gc.collect()
            output = {"image": result["image"], "target": result["target"]}
        else:
            output = self._to_tensors_no_transform(image, target)

        output["path"] = self.get_path(
            getattr(self, "_last_image_idx", idx), key=self.image_key
        )
        return output

    def _get_src(self, path: str):
        return self._file_cache.get(path)

    def _read_crop(self, image_path: str, x: int, y: int) -> np.ndarray:
        crop_h, crop_w = self.crop_size
        window = Window(col_off=x, row_off=y, width=crop_w, height=crop_h)
        try:
            src = self._get_src(image_path)
            data = (
                src.read(window=window)
                if self.selected_bands is None
                else src.read(self.selected_bands, window=window)
            )
        except (rasterio.errors.RasterioIOError, Exception) as e:
            logger.error(f"Error reading window {window} from {image_path}: {e}")
            # In random crop mode, we can just try another random crop
            # This is handled by the caller or we can return a zero array
            # but better to raise a custom exception that can be caught by _get_random_crop
            raise e

        image = np.transpose(data, (1, 2, 0)).copy()
        if self.image_dtype == "native":
            return image
        return image.astype(np.dtype(self.image_dtype))

    def _get_random_crop(self) -> np.ndarray:
        crop_h, crop_w = self.crop_size
        image = None
        img_idx = self._valid_indices[0]
        for _ in range(self.max_retries):
            choice = np.random.choice(len(self._valid_indices), p=self.image_weights)
            img_idx = self._valid_indices[choice]
            width, height = self.image_dims[img_idx]
            x = np.random.randint(0, width - crop_w + 1)
            y = np.random.randint(0, height - crop_h + 1)
            image_path = self.get_path(img_idx, key=self.image_key)
            try:
                image = self._read_crop(image_path, x, y)
                if image is not None and image.size > 0:
                    break
            except Exception:
                continue
        self._last_image_idx = img_idx
        return image

    def _to_tensors_no_transform(
        self, image: np.ndarray, target: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        image_tensor = torch.from_numpy(image.copy()).float().permute(2, 0, 1)
        target_tensor = torch.from_numpy(target.copy()).float().permute(2, 0, 1)
        norm_factor = _DTYPE_NORMALIZATION.get(self.image_dtype)
        if norm_factor is not None:
            image_tensor = image_tensor / norm_factor
            target_tensor = target_tensor / norm_factor
        return {"image": image_tensor, "target": target_tensor}

    @classmethod
    def _build_dataframe_from_folder(
        cls,
        image_dir: Path,
        image_extensions: Sequence[str],
        split: str,
        val_fraction: float,
        split_seed: int,
    ) -> pd.DataFrame:
        paths = []
        if image_dir.exists():
            valid_exts = {ext.lower() for ext in image_extensions}
            for path in image_dir.rglob("*"):
                if not path.is_file():
                    continue
                if path.name.endswith(".aux.xml"):
                    continue
                if path.suffix.lower() in valid_exts:
                    paths.append(path.resolve())
        paths = sorted(paths)
        if not paths:
            raise ValueError(
                f"No images found in '{image_dir}' with extensions "
                f"{sorted(image_extensions)}"
            )

        if split != "all":
            rng = np.random.default_rng(split_seed)
            order = rng.permutation(len(paths))
            n_val = max(1, int(round(len(paths) * val_fraction)))
            n_val = min(n_val, len(paths) - 1) if len(paths) > 1 else 1
            val_indices = set(order[:n_val].tolist())
            if split == "val":
                paths = [p for i, p in enumerate(paths) if i in val_indices]
            else:
                paths = [p for i, p in enumerate(paths) if i not in val_indices]
            if not paths:
                raise ValueError(f"Split '{split}' produced no images")

        return pd.DataFrame({"image": [str(path) for path in paths]})

    @staticmethod
    def _normalize_extension(ext: str) -> str:
        return ext if ext.startswith(".") else f".{ext}"


class WindowedImageDataset(ImageDataset):
    """Deterministic sliding-window patch dataset for image-only tasks.

    The dataset calculates a grid of patches for each image and allows
    global indexing across all patches. Useful for processing all areas
    of a set of rasters in a fixed grid.

    Args:
        input_csv_path: Optional CSV with image paths.
        df: Optional pre-built DataFrame.
        image_dir: Root folder scanned recursively for images.
        image_extensions: Extensions to include when scanning ``image_dir``.
        split: ``"all"``, ``"train"`` or ``"val"`` for deterministic folder
            splitting.
        val_fraction: Fraction assigned to the validation split.
        split_seed: Seed used to shuffle discovered paths before splitting.
        crop_size: Patch size as ``[height, width]``.
        stride: Stride between patches. Defaults to ``crop_size``.
        selected_bands: Optional 1-based rasterio band indices.
        image_dtype: One of ``"uint8"``, ``"uint16"``, ``"float32"`` or
            ``"native"``.
        file_cache_maxsize: Max number of open rasters in the LRU cache.
        verify_windows: If ``True``, reads every candidate window during
            initialisation and indexes only windows that can be read with the
            expected shape.
        window_index_cache: Optional JSON cache path used to persist the
            verified window index and avoid repeating the validation pass.
        serialize_rasterio_reads: If True, serializes rasterio window reads
            per source file across DataLoader workers using an OS-level lock.
            This is useful for compressed GeoTIFFs or network filesystems that
            fail under concurrent reads from the same file.
        rasterio_lock_dir: Directory used to store lock files when
            ``serialize_rasterio_reads`` is enabled.
        reopen_rasterio_on_read: If True, opens the raster inside each
            serialized read and closes it immediately after reading. This
            avoids persistent GDAL DatasetReader state in DataLoader workers.
        **kwargs: Compatibility parameters accepted by ``ImageDataset`` and Hydra.

    Returns:
        Dict[str, Any]: Items with ``image`` and ``path`` keys.

    Example YAML:
        val_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.WindowedImageDataset
          image_dir: /data/images
          crop_size: [256, 256]
          stride: 256
          verify_windows: true
          window_index_cache: /data/cache/windowed_image_index.json
    """

    def __init__(
        self,
        input_csv_path: Optional[Union[str, Path]] = None,
        df: Optional[pd.DataFrame] = None,
        image_dir: Optional[Union[str, Path]] = None,
        image_extensions: Optional[Sequence[str]] = None,
        split: str = "all",
        val_fraction: float = 0.2,
        split_seed: int = 42,
        crop_size: Optional[List[int]] = None,
        stride: Optional[Union[int, List[int]]] = None,
        selected_bands: Optional[List[int]] = None,
        image_dtype: str = "uint8",
        file_cache_maxsize: int = 0,
        verify_windows: bool = False,
        window_index_cache: Optional[Union[str, Path]] = None,
        serialize_rasterio_reads: bool = False,
        rasterio_lock_dir: Optional[Union[str, Path]] = None,
        reopen_rasterio_on_read: bool = False,
        **kwargs,
    ) -> None:
        if crop_size is None:
            crop_size = [256, 256]
        if len(crop_size) != 2:
            raise ValueError("crop_size must be [height, width]")

        self.crop_size = crop_size
        if stride is None:
            self.stride = crop_size
        else:
            self.stride = [stride, stride] if isinstance(stride, int) else stride
        if len(self.stride) != 2:
            raise ValueError("stride must be an int or [height, width]")

        if df is None and input_csv_path is None and image_dir is not None:
            df = AutoencoderRandomCropDataset._build_dataframe_from_folder(
                Path(image_dir),
                image_extensions
                or AutoencoderRandomCropDataset.DEFAULT_IMAGE_EXTENSIONS,
                split=split,
                val_fraction=val_fraction,
                split_seed=split_seed,
            )

        super().__init__(
            input_csv_path=input_csv_path,
            df=df,
            **kwargs,
        )

        self.selected_bands = selected_bands
        self.image_dtype = image_dtype
        self.verify_windows = verify_windows
        self.window_index_cache = (
            str(window_index_cache) if window_index_cache is not None else None
        )
        self.serialize_rasterio_reads = serialize_rasterio_reads
        self.rasterio_lock_dir = (
            str(rasterio_lock_dir) if rasterio_lock_dir is not None else None
        )
        self.reopen_rasterio_on_read = reopen_rasterio_on_read
        self._window_index = None

        # Calculate grid for each image
        self._image_info = []
        self._cumulative = [0]
        total_patches = 0

        crop_h, crop_w = self.crop_size
        stride_h, stride_w = self.stride

        for i in range(len(self.df)):
            path = self.get_path(i, key=self.image_key)
            with rasterio.open(path) as src:
                w, h = src.width, src.height

            if w < crop_w or h < crop_h:
                continue

            nx = (w - crop_w) // stride_w + 1
            ny = (h - crop_h) // stride_h + 1
            n_patches = nx * ny

            if n_patches > 0:
                self._image_info.append(
                    {
                        "img_idx": i,
                        "path": path,
                        "width": w,
                        "height": h,
                        "nx": nx,
                        "ny": ny,
                        "n_patches": n_patches,
                    }
                )
                total_patches += n_patches
                self._cumulative.append(total_patches)

        self._total_patches = total_patches
        if not self._image_info:
            logger.warning(
                "WindowedImageDataset: No images large enough for crop %s",
                self.crop_size,
            )

        n_files = len(self._image_info)
        self.file_cache_maxsize = (
            file_cache_maxsize if file_cache_maxsize > 0 else n_files + 16
        )
        self._file_cache = _RasterioLRUCache(maxsize=self.file_cache_maxsize)

        if self.verify_windows:
            self._build_verified_window_index()

    def __len__(self) -> int:
        return self._total_patches

    def get_path(self, idx: int, key: str = None, add_root_dir: bool = True) -> str:
        """Get the original image path for a given patch index.

        Args:
            idx: Global patch index.  If ``idx >= len(self._cumulative)``, it
                is assumed to be a patch index and mapped to the source image.
                Otherwise, it is treated as a source image index for backward
                compatibility with the base class.
            key: Dataframe column name.
            add_root_dir: Whether to prepend ``root_dir``.

        Returns:
            Source image path.
        """
        if self._window_index is not None and idx >= len(self.df):
            entry = self._window_index[idx]
            return entry["path"]

        if idx >= len(self.df):
            # Map global patch index to image index
            img_idx = bisect.bisect_right(self._cumulative, idx) - 1
            # Use the mapped image index to get the path
            return super().get_path(img_idx, key=key, add_root_dir=add_root_dir)

        return super().get_path(idx, key=key, add_root_dir=add_root_dir)

    def _get_src(self, path: str):
        return self._file_cache.get(path)

    def _build_verified_window_index(self) -> None:
        """Build or load the list of readable window positions."""
        if self.window_index_cache and Path(self.window_index_cache).exists():
            if self._load_window_index_cache():
                return

        window_index = []
        rejected = 0
        for image_info_idx, info in enumerate(self._image_info):
            for row_off, col_off in self._iter_window_offsets(info):
                if self._is_readable_window(info, row_off, col_off):
                    window_index.append(
                        {
                            "image_info_idx": image_info_idx,
                            "img_idx": info["img_idx"],
                            "path": info["path"],
                            "row_off": row_off,
                            "col_off": col_off,
                        }
                    )
                else:
                    rejected += 1

        self._window_index = window_index
        self._total_patches = len(window_index)
        self._cumulative = [0, self._total_patches]
        if rejected:
            logger.info(
                "WindowedImageDataset: rejected %s unreadable windows during init",
                rejected,
            )
        if not self._window_index:
            logger.warning(
                "WindowedImageDataset: No readable windows found for crop %s",
                self.crop_size,
            )
        if self.window_index_cache:
            self._save_window_index_cache()

    def _iter_window_offsets(self, info: Dict[str, Any]):
        """Yield deterministic row/column offsets for an image grid."""
        for grid_row in range(info["ny"]):
            for grid_col in range(info["nx"]):
                yield grid_row * self.stride[0], grid_col * self.stride[1]

    def _is_readable_window(
        self, info: Dict[str, Any], row_off: int, col_off: int
    ) -> bool:
        """Return whether a candidate raster window can be read completely."""
        window = Window(
            col_off=col_off,
            row_off=row_off,
            width=self.crop_size[1],
            height=self.crop_size[0],
        )
        try:
            with rasterio.open(info["path"]) as src:
                data = (
                    src.read(window=window)
                    if self.selected_bands is None
                    else src.read(self.selected_bands, window=window)
                )
        except Exception as exc:
            logger.warning(
                "WindowedImageDataset: excluding unreadable window %s from %s: %s",
                window,
                info["path"],
                exc,
            )
            return False

        expected_bands = (
            len(self.selected_bands) if self.selected_bands else data.shape[0]
        )
        expected_shape = (expected_bands, self.crop_size[0], self.crop_size[1])
        if data.shape != expected_shape or data.size == 0:
            logger.warning(
                "WindowedImageDataset: excluding window %s from %s with shape %s; "
                "expected %s",
                window,
                info["path"],
                data.shape,
                expected_shape,
            )
            return False
        return True

    def _window_cache_config(self) -> Dict[str, Any]:
        """Return the metadata used to validate a persisted window index."""
        paths = [info["path"] for info in self._image_info]
        return {
            "crop_size": list(self.crop_size),
            "stride": list(self.stride),
            "selected_bands": self.selected_bands,
            "image_dtype": self.image_dtype,
            "image_key": self.image_key,
            "paths": paths,
            "files": [self._file_fingerprint(path) for path in paths],
        }

    @staticmethod
    def _file_fingerprint(path: str) -> Dict[str, Any]:
        stat = Path(path).stat()
        return {
            "path": path,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _save_window_index_cache(self) -> None:
        """Persist the verified window index to JSON."""
        cache_path = Path(self.window_index_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = {
            "config": self._window_cache_config(),
            "windows": [
                {
                    "img_idx": entry["img_idx"],
                    "path": entry["path"],
                    "row_off": entry["row_off"],
                    "col_off": entry["col_off"],
                }
                for entry in self._window_index
            ],
        }
        with cache_path.open("w") as f:
            json.dump(cache, f)
        logger.info(
            "WindowedImageDataset: window index cache saved to %s (%s windows)",
            cache_path,
            len(self._window_index),
        )

    def _load_window_index_cache(self) -> bool:
        """Load a verified window index cache when its metadata still matches."""
        cache_path = Path(self.window_index_cache)
        try:
            with cache_path.open() as f:
                cache = json.load(f)
        except Exception as exc:
            logger.warning(
                "WindowedImageDataset: failed to load window index cache %s: %s",
                cache_path,
                exc,
            )
            return False

        current_config = self._window_cache_config()
        if cache.get("config") != current_config:
            logger.info(
                "WindowedImageDataset: window index cache %s is stale; rebuilding",
                cache_path,
            )
            return False

        info_lookup = {
            (info["img_idx"], info["path"]): image_info_idx
            for image_info_idx, info in enumerate(self._image_info)
        }
        window_index = []
        for entry in cache.get("windows", []):
            key = (entry.get("img_idx"), entry.get("path"))
            if key not in info_lookup:
                logger.info(
                    "WindowedImageDataset: window index cache %s references an "
                    "unknown image; rebuilding",
                    cache_path,
                )
                return False
            window_index.append(
                {
                    "image_info_idx": info_lookup[key],
                    "img_idx": entry["img_idx"],
                    "path": entry["path"],
                    "row_off": entry["row_off"],
                    "col_off": entry["col_off"],
                }
            )

        self._window_index = window_index
        self._total_patches = len(window_index)
        self._cumulative = [0, self._total_patches]
        logger.info(
            "WindowedImageDataset: window index cache loaded from %s (%s windows)",
            cache_path,
            self._total_patches,
        )
        return True

    def _get_window_info(self, idx: int):
        """Map a global patch index to image metadata and raster offsets."""
        if self._window_index is not None:
            entry = self._window_index[idx]
            return (
                self._image_info[entry["image_info_idx"]],
                entry["row_off"],
                entry["col_off"],
            )

        img_idx = bisect.bisect_right(self._cumulative, idx) - 1
        info = self._image_info[img_idx]
        local_idx = idx - self._cumulative[img_idx]
        grid_row = local_idx // info["nx"]
        grid_col = local_idx % info["nx"]
        return info, grid_row * self.stride[0], grid_col * self.stride[1]

    def _read_window(self, info: Dict[str, Any], row_off: int, col_off: int):
        """Read one indexed raster window."""
        window = Window(
            col_off=col_off,
            row_off=row_off,
            width=self.crop_size[1],
            height=self.crop_size[0],
        )
        with _rasterio_read_lock(
            info["path"],
            self.serialize_rasterio_reads,
            self.rasterio_lock_dir,
        ):
            if self.reopen_rasterio_on_read:
                with rasterio.open(info["path"]) as src:
                    data = self._read_window_data(src, window)
            else:
                src = self._get_src(info["path"])
                data = self._read_window_data(src, window)
        return data, window

    def _read_window_data(self, src, window: Window):
        """Read raw raster data for one window from an open DatasetReader."""
        if self.selected_bands is None:
            return src.read(window=window)
        return src.read(self.selected_bands, window=window)

    def _format_image_item(self, info: Dict[str, Any], data) -> Dict[str, Any]:
        """Convert a raw raster window into the public image sample."""
        image = np.transpose(data, (1, 2, 0)).copy()
        if self.image_dtype != "native":
            image = image.astype(np.dtype(self.image_dtype))

        if self.transform is not None:
            result = self.transform(image=image)
        else:
            image_tensor = torch.from_numpy(image.copy()).float().permute(2, 0, 1)
            norm_factor = _DTYPE_NORMALIZATION.get(self.image_dtype)
            if norm_factor is not None:
                image_tensor = image_tensor / norm_factor
            result = {"image": image_tensor}

        result.update({"path": info["path"]})
        return result

    def _build_item_from_window(
        self, info: Dict[str, Any], row_off: int, col_off: int
    ) -> Dict[str, Any]:
        """Read and format one image-only window sample."""
        data, _ = self._read_window(info, row_off, col_off)
        return self._format_image_item(info, data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self._total_patches:
            raise IndexError(f"Index {idx} out of range [0, {self._total_patches})")

        info, row_off, col_off = self._get_window_info(idx)
        window = Window(
            col_off=col_off,
            row_off=row_off,
            width=self.crop_size[1],
            height=self.crop_size[0],
        )
        try:
            return self._build_item_from_window(info, row_off, col_off)
        except (rasterio.errors.RasterioIOError, Exception) as e:
            logger.error(f"Error reading window {window} from {info['path']}: {e}")
            # Try another index to avoid crashing
            new_idx = (idx + 1) % self._total_patches
            return self.__getitem__(new_idx)

    def _iter_worker_image_info_indices(self):
        """Yield image-info indices assigned to the current DataLoader worker."""
        worker = get_worker_info()
        if worker is None:
            yield from range(len(self._image_info))
            return

        yield from range(worker.id, len(self._image_info), worker.num_workers)

    def _iter_window_specs_for_image_indices(self, image_info_indices):
        """Yield ``(info, row_off, col_off)`` for selected source images."""
        image_info_indices = set(image_info_indices)
        if self._window_index is not None:
            for entry in self._window_index:
                image_info_idx = entry["image_info_idx"]
                if image_info_idx not in image_info_indices:
                    continue
                yield (
                    self._image_info[image_info_idx],
                    entry["row_off"],
                    entry["col_off"],
                )
            return

        for image_info_idx in sorted(image_info_indices):
            info = self._image_info[image_info_idx]
            for row_off, col_off in self._iter_window_offsets(info):
                yield info, row_off, col_off

    def _iter_worker_window_specs(self):
        """Yield window specs for images assigned to this worker."""
        return self._iter_window_specs_for_image_indices(
            self._iter_worker_image_info_indices()
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_cache"] = _RasterioLRUCache(maxsize=self.file_cache_maxsize)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        if hasattr(self, "_file_cache") and isinstance(
            self._file_cache, _RasterioLRUCache
        ):
            self._file_cache.close_all()


class WindowedImageAutoencoderDataset(WindowedImageDataset):
    """Deterministic sliding-window patch dataset for autoencoder reconstruction.

    The clean crop is used as ``target``; optional corruption augmentations
    affect only ``image``.

    Args:
        corruption_augmentation_list: Albumentations configs applied only to
            ``image``.
        **kwargs: Parameters accepted by :class:`WindowedImageDataset`.

    Returns:
        Dict[str, Any]: Items with ``image``, ``target`` and ``path`` keys.

    Example YAML:
        val_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.WindowedImageAutoencoderDataset
          image_dir: /data/images
          crop_size: [256, 256]
          stride: 256
    """

    def __init__(
        self,
        corruption_augmentation_list: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.corruption_transform = (
            None
            if corruption_augmentation_list is None
            else load_augmentation_object(corruption_augmentation_list)
        )
        if self.transform is not None:
            self.transform = A.Compose(
                self.transform.transforms, additional_targets={"target": "image"}
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a deterministic crop pair for reconstruction."""
        # We reuse the window reading logic but need to handle target
        if idx < 0 or idx >= self._total_patches:
            raise IndexError(f"Index {idx} out of range [0, {self._total_patches})")

        info, row_off, col_off = self._get_window_info(idx)
        window = Window(
            col_off=col_off,
            row_off=row_off,
            width=self.crop_size[1],
            height=self.crop_size[0],
        )
        try:
            return self._build_item_from_window(info, row_off, col_off)
        except (rasterio.errors.RasterioIOError, Exception) as e:
            logger.error(f"Error reading window {window} from {info['path']}: {e}")
            # Try another index to avoid crashing
            new_idx = (idx + 1) % self._total_patches
            return self.__getitem__(new_idx)

    def _build_item_from_window(
        self, info: Dict[str, Any], row_off: int, col_off: int
    ) -> Dict[str, Any]:
        """Read and format one autoencoder reconstruction window sample."""
        data, _ = self._read_window(info, row_off, col_off)
        image = np.transpose(data, (1, 2, 0)).copy()
        if self.image_dtype != "native":
            image = image.astype(np.dtype(self.image_dtype))

        target = image.copy()

        if self.corruption_transform is not None:
            image = self.corruption_transform(image=image)["image"]

        if self.transform is not None:
            res = self.transform(image=image, target=target)
            result = {"image": res["image"], "target": res["target"]}
        else:
            result = self._to_tensors_no_transform(image, target)

        result.update({"path": info["path"]})
        return result

    def _to_tensors_no_transform(
        self, image: np.ndarray, target: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        image_tensor = torch.from_numpy(image.copy()).float().permute(2, 0, 1)
        target_tensor = torch.from_numpy(target.copy()).float().permute(2, 0, 1)
        norm_factor = _DTYPE_NORMALIZATION.get(self.image_dtype)
        if norm_factor is not None:
            image_tensor = image_tensor / norm_factor
            target_tensor = target_tensor / norm_factor
        return {"image": image_tensor, "target": target_tensor}


class IterableWindowedImageDataset(WindowedImageDataset, IterableDataset):
    """Iterable windowed image dataset that shards source images by worker.

    This variant yields the same image-only samples as ``WindowedImageDataset``
    but assigns whole source rasters to individual DataLoader workers. It avoids
    concurrent reads from the same GeoTIFF when ``num_workers > 0``.

    Args:
        **kwargs: Parameters accepted by :class:`WindowedImageDataset`.

    Returns:
        Dict[str, Any]: Items with ``image`` and ``path`` keys.

    Example YAML:
        val_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.IterableWindowedImageDataset
          image_dir: /data/images
          crop_size: [224, 224]
          stride: 224
          verify_windows: true
          data_loader:
            shuffle: false
            num_workers: 4
    """

    def __iter__(self):
        """Yield windows for images assigned to the current worker."""
        for info, row_off, col_off in self._iter_worker_window_specs():
            try:
                yield self._build_item_from_window(info, row_off, col_off)
            except (rasterio.errors.RasterioIOError, Exception) as exc:
                window = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=self.crop_size[1],
                    height=self.crop_size[0],
                )
                logger.error(
                    "Error reading window %s from %s: %s",
                    window,
                    info["path"],
                    exc,
                )


class IterableWindowedImageAutoencoderDataset(
    WindowedImageAutoencoderDataset, IterableDataset
):
    """Iterable windowed autoencoder dataset that shards images by worker.

    The yielded samples match ``WindowedImageAutoencoderDataset`` but each
    DataLoader worker receives a disjoint subset of source rasters, preventing
    multiple workers from reading the same GeoTIFF concurrently.

    Args:
        **kwargs: Parameters accepted by
            :class:`WindowedImageAutoencoderDataset`.

    Returns:
        Dict[str, Any]: Items with ``image``, ``target`` and ``path`` keys.

    Example YAML:
        val_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.image_dataset.IterableWindowedImageAutoencoderDataset
          image_dir: /data/images
          crop_size: [224, 224]
          stride: 224
          verify_windows: true
          data_loader:
            shuffle: false
            num_workers: 4
    """

    def __iter__(self):
        """Yield reconstruction windows for this worker's source images."""
        for info, row_off, col_off in self._iter_worker_window_specs():
            try:
                yield self._build_item_from_window(info, row_off, col_off)
            except (rasterio.errors.RasterioIOError, Exception) as exc:
                window = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=self.crop_size[1],
                    height=self.crop_size[0],
                )
                logger.error(
                    "Error reading window %s from %s: %s",
                    window,
                    info["path"],
                    exc,
                )
