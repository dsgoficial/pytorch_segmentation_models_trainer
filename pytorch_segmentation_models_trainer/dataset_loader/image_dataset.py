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

import gc
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

from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    AbstractDataset,
    _DTYPE_NORMALIZATION,
    _RasterioLRUCache,
    _VALID_IMAGE_DTYPES,
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
        src = self._get_src(image_path)
        data = (
            src.read(window=window)
            if self.selected_bands is None
            else src.read(self.selected_bands, window=window)
        )
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
            image = self._read_crop(image_path, x, y)
            if image.size > 0:
                break
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
