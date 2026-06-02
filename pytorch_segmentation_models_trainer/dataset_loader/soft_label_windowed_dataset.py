# -*- coding: utf-8 -*-
"""Dataset for soft-label training with windowed reads from full-scene rasters."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import albumentations as A
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
import torch

from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    load_augmentation_object,
)
from pytorch_segmentation_models_trainer.dataset_loader.soft_label_dataset import (
    SoftLabelDataset,
)

logger = logging.getLogger(__name__)


class SoftLabelWindowedDataset(SoftLabelDataset):
    """Soft-label dataset that reads fixed-size patches from full-scene rasters.

    Each CSV row defines one patch via pixel offsets into the full raster.
    This avoids the need to pre-cut tiles: the entire scene rasters (image,
    P_soft, W_conf) are stored once and patches are read on-the-fly with
    ``rasterio.windows.Window``.

    CSV columns required:
        - ``image_key`` — path to the full RGB image raster (default ``image_path``).
        - ``p_soft_key`` — path to the full P_soft raster (default ``p_soft_path``).
        - ``w_conf_key`` — path to the full W_conf raster (default ``w_conf_path``,
          optional; omit the column to train without confidence weighting).
        - ``row_off_key`` — top-left row offset of the patch in pixels (default ``row_off``).
        - ``col_off_key`` — top-left column offset of the patch in pixels (default ``col_off``).
        - ``patch_size_key`` — patch height and width in pixels (default ``patch_size``).

    The same P_soft and W_conf column paths may appear in multiple rows
    (one row per patch) — rasterio reads only the requested window each time.

    Args:
        input_csv_path: CSV file path. Mutually exclusive with ``df``.
        df: Pre-built DataFrame. Mutually exclusive with ``input_csv_path``.
        root_dir: Root directory prepended to relative paths in the CSV.
        augmentation_list: Albumentations transform configs applied identically
            to image, P_soft, and W_conf (via ``additional_targets``).
        data_loader: DataLoader configuration stored on the object.
        image_key: CSV column for image paths. Default ``"image_path"``.
        p_soft_key: CSV column for P_soft raster paths. Default ``"p_soft_path"``.
        w_conf_key: CSV column for W_conf raster paths. Default ``"w_conf_path"``.
        row_off_key: CSV column for patch row offset. Default ``"row_off"``.
        col_off_key: CSV column for patch column offset. Default ``"col_off"``.
        patch_size_key: CSV column for patch size (square). Default ``"patch_size"``.
        n_first_rows_to_read: Optional row limit for CSV reading.
        seed: Optional seed for the augmentation pipeline.
        **kwargs: Reserved for Hydra compatibility.

    Returns:
        Dict with keys:

        - ``"image"``: ``(3, patch_size, patch_size)`` float32 tensor in ``[0, 1]``.
        - ``"mask"``: dict with

          - ``"mask"``: ``(C, patch_size, patch_size)`` float32 — soft label distribution.
          - ``"w_conf"``: ``(1, patch_size, patch_size)`` float32 — confidence weights
            *(present only when ``w_conf_key`` column exists in the CSV)*.

        - ``"path"``: image file path string.

    Example YAML:
        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.soft_label_windowed_dataset.SoftLabelWindowedDataset
          input_csv_path: /data/patches.csv
          row_off_key: row_off
          col_off_key: col_off
          patch_size_key: patch_size
          augmentation_list:
            - _target_: albumentations.HorizontalFlip
              p: 0.5
            - _target_: albumentations.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
            - _target_: albumentations.pytorch.ToTensorV2
    """

    def __init__(
        self,
        input_csv_path: Optional[Union[str, Path]] = None,
        df: Optional[pd.DataFrame] = None,
        root_dir=None,
        augmentation_list=None,
        data_loader=None,
        image_key: str = "image_path",
        p_soft_key: str = "p_soft_path",
        w_conf_key: str = "w_conf_path",
        row_off_key: str = "row_off",
        col_off_key: str = "col_off",
        patch_size_key: str = "patch_size",
        n_first_rows_to_read: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            input_csv_path=input_csv_path,
            df=df,
            root_dir=root_dir,
            augmentation_list=augmentation_list,
            data_loader=data_loader,
            image_key=image_key,
            p_soft_key=p_soft_key,
            w_conf_key=w_conf_key,
            n_first_rows_to_read=n_first_rows_to_read,
            seed=seed,
        )
        self.row_off_key = row_off_key
        self.col_off_key = col_off_key
        self.patch_size_key = patch_size_key

        for col in [row_off_key, col_off_key, patch_size_key]:
            if col not in self.df.columns:
                raise ValueError(
                    f"Column '{col}' is required in the CSV/DataFrame for "
                    f"SoftLabelWindowedDataset."
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _window(self, idx: int) -> rasterio.windows.Window:
        """Build the rasterio Window for the patch at ``idx``."""
        row = self.df.iloc[idx]
        patch = int(row[self.patch_size_key])
        return rasterio.windows.Window(
            col_off=int(row[self.col_off_key]),
            row_off=int(row[self.row_off_key]),
            width=patch,
            height=patch,
        )

    def _load_image_window(
        self, path: str, window: rasterio.windows.Window
    ) -> np.ndarray:
        """Read an RGB image window and return it as (H, W, 3) uint8."""
        with rasterio.open(path) as src:
            data = src.read(window=window)  # (C, H, W)
        return np.transpose(data, (1, 2, 0)).copy()

    def _load_raster_hwc_window(
        self, path: str, window: rasterio.windows.Window
    ) -> np.ndarray:
        """Read a multi-band float raster window and return (H, W, C) float32."""
        with rasterio.open(path) as src:
            data = src.read(window=window)  # (C, H, W)
        return np.transpose(data, (1, 2, 0)).astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % self.len
        window = self._window(idx)

        image_np = self._load_image_window(
            self.get_path(idx, key=self.image_key), window
        )
        p_soft_np = self._load_raster_hwc_window(
            self.get_path(idx, key=self.p_soft_key), window
        )
        w_conf_np: Optional[np.ndarray] = None
        if self._has_w_conf:
            w_conf_np = self._load_raster_hwc_window(
                self.get_path(idx, key=self.w_conf_key), window
            )

        if self.transform is not None:
            kwargs: Dict[str, Any] = {"image": image_np, "p_soft": p_soft_np}
            if w_conf_np is not None:
                kwargs["w_conf"] = w_conf_np
            result = self.transform(**kwargs)
            image_out = result["image"]
            p_soft_out = result["p_soft"]
            w_conf_out = result.get("w_conf")
        else:
            image_out = image_np
            p_soft_out = p_soft_np
            w_conf_out = w_conf_np

        image_t = self._image_to_chw_float(image_out)
        p_soft_t = self._to_chw_float(p_soft_out)

        mask_dict: Dict[str, torch.Tensor] = {"mask": p_soft_t}
        if w_conf_out is not None:
            mask_dict["w_conf"] = self._to_chw_float(w_conf_out)

        return {
            "image": image_t,
            "mask": mask_dict,
            "path": self.get_path(idx, key=self.image_key),
        }
