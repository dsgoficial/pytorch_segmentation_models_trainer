# -*- coding: utf-8 -*-
"""Dataset for soft-label training with per-pixel confidence weights."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import albumentations as A
import numpy as np
import pandas as pd
import rasterio
import torch

from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    AbstractDataset,
)

logger = logging.getLogger(__name__)


class SoftLabelDataset(AbstractDataset):
    """Dataset for training with probabilistic soft labels and confidence weights.

    Reads per-tile GeoTIFF rasters:
    - **image**: planet-style RGB image (3 bands, uint8).
    - **p_soft**: per-class probability distribution, shape ``(C, H, W)`` float32.
    - **w_conf** *(optional)*: pixel-wise confidence weight, shape ``(1, H, W)``
      float32, values in ``[0, 1]``.

    The ``"mask"`` entry in each returned sample is a **dict** so that both
    ``p_soft`` and ``w_conf`` reach the loss function intact via the existing
    FrameField compound-loss path in ``SoftLabelModel._compute_loss``.

    .. note::
        Do **not** include ``albumentations.pytorch.ToTensorV2`` with
        ``transpose_mask=True`` in ``augmentation_list``.  Tensor conversion is
        handled internally.  Plain ``ToTensorV2`` (default) is supported.

    Args:
        input_csv_path: CSV file with at least ``image_path`` and
            ``p_soft_path`` columns.  ``w_conf_path`` is optional.
        df: Pre-built DataFrame (alternative to ``input_csv_path``).
        root_dir: Root directory prepended to relative paths in the CSV.
        augmentation_list: Albumentations transform configs.  Geometric
            transforms are applied identically to image, p_soft and w_conf.
        data_loader: DataLoader configuration stored on the object for the
            Lightning model.
        image_key: CSV column for image paths. Default ``"image_path"``.
        p_soft_key: CSV column for P_soft raster paths. Default
            ``"p_soft_path"``.
        w_conf_key: CSV column for W_conf raster paths. Default
            ``"w_conf_path"``.  When absent from the CSV, w_conf is not loaded
            and the returned mask dict contains only ``"mask"``.
        n_first_rows_to_read: Optional row limit for CSV reading.
        seed: Optional seed for the augmentation pipeline.
        **kwargs: Reserved for Hydra compatibility.

    Returns:
        Dict with keys:

        - ``"image"``: ``(3, H, W)`` float32 tensor, values in ``[0, 1]``.
        - ``"mask"``: dict with

          - ``"mask"``: ``(C, H, W)`` float32 — soft label distribution.
          - ``"w_conf"``: ``(1, H, W)`` float32 — confidence weights
            *(present only when* ``w_conf_key`` *column exists in the CSV)*.

        - ``"path"``: image file path string.

    Example YAML:
        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.soft_label_dataset.SoftLabelDataset
          input_csv_path: /data/train.csv
          augmentation_list:
            - _target_: albumentations.HorizontalFlip
              p: 0.5
            - _target_: albumentations.VerticalFlip
              p: 0.5
            - _target_: albumentations.RandomRotate90
              p: 0.5
            - _target_: albumentations.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
            - _target_: albumentations.pytorch.ToTensorV2
          data_loader:
            batch_size: 16
            num_workers: 8
            shuffle: true
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
            n_first_rows_to_read=n_first_rows_to_read,
            seed=seed,
        )
        self.p_soft_key = p_soft_key
        self.w_conf_key = w_conf_key
        self._has_w_conf = w_conf_key in self.df.columns

        # Rebuild the Compose with additional_targets so geometric transforms
        # are applied identically to p_soft and w_conf.
        if self.transform is not None:
            additional_targets: Dict[str, str] = {"p_soft": "mask"}
            if self._has_w_conf:
                additional_targets["w_conf"] = "mask"
            self.transform = A.Compose(
                self.transform.transforms,
                additional_targets=additional_targets,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_raster_hwc(self, path: str) -> np.ndarray:
        """Read a multi-band GeoTIFF and return it as (H, W, C) float32."""
        with rasterio.open(path) as src:
            data = src.read()  # (C, H, W)
        return np.transpose(data, (1, 2, 0)).astype(np.float32)

    @staticmethod
    def _to_chw_float(
        x: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Convert a HWC array/tensor to a CHW float32 tensor.

        albumentations ToTensorV2 converts *image* targets to CHW tensors but
        leaves *mask* targets in HWC format (no transpose by default).  This
        helper normalises both cases.
        """
        if isinstance(x, torch.Tensor):
            if x.ndim == 3:
                # mask target from ToTensorV2 arrives as HWC tensor
                return x.permute(2, 0, 1).float().contiguous()
            return x.unsqueeze(0).float()
        # numpy
        if x.ndim == 3:
            return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()
        return torch.from_numpy(x[np.newaxis].copy()).float()

    @staticmethod
    def _image_to_chw_float(
        x: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Convert image output to a CHW float32 tensor in [0, 1].

        ToTensorV2 already converts image targets to CHW uint8/float tensors.
        When no ToTensorV2 is used the image is numpy HWC uint8.
        """
        if isinstance(x, torch.Tensor):
            t = x.float()
            if t.max() > 1.0 + 1e-3:
                t = t / 255.0
            return t
        # numpy HWC
        arr = np.ascontiguousarray(x.transpose(2, 0, 1))
        t = torch.from_numpy(arr).float()
        if t.max() > 1.0 + 1e-3:
            t = t / 255.0
        return t

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % self.len

        # Load image (H, W, 3) uint8 via parent helper
        image_np = self.load_image(idx, key=self.image_key)

        # Load P_soft (H, W, C) float32
        p_soft_np = self._load_raster_hwc(self.get_path(idx, key=self.p_soft_key))

        # Load W_conf (H, W, 1) float32 when column exists
        w_conf_np: Optional[np.ndarray] = None
        if self._has_w_conf:
            w_conf_np = self._load_raster_hwc(self.get_path(idx, key=self.w_conf_key))

        # Apply augmentations consistently across all spatial targets
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

        # Convert to CHW float tensors
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
