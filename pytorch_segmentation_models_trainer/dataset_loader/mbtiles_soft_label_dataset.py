# -*- coding: utf-8 -*-
"""Soft-label dataset that reads RGB from MBTiles on mask-referenced windows."""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import albumentations as A
import numpy as np
import rasterio
import torch

from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    load_augmentation_object,
)
from pytorch_segmentation_models_trainer.dataset_loader.mbtiles_mask_dataset import (
    MBTilesMaskWindowedDataset,
)
from pytorch_segmentation_models_trainer.dataset_loader.soft_label_dataset import (
    SoftLabelDataset,
)
from pytorch_segmentation_models_trainer.tools.mbtiles.alignment import (
    read_mask_window,
    read_source_aligned_to_mask_window,
)
from pytorch_segmentation_models_trainer.tools.soft_labels.build_soft_labels import (
    compute_p_soft,
    compute_w_conf,
)


class MBTilesSoftLabelMaskWindowedDataset(MBTilesMaskWindowedDataset):
    """Build soft labels on-the-fly for MBTiles imagery and mask windows.

    Each sample is defined by a window over a cartographic mask GeoTIFF. RGB is
    read from the MBTiles source and reprojected into that mask-window grid.
    The soft target ``P_soft`` is an equal vote over the BAGS mask window and
    every LULC raster listed in ``lulc_paths``. Optional ``W_conf`` is computed
    with the same ``compute_w_conf`` implementation used by the offline
    ``build-soft-labels`` command.

    Args:
        mbtiles_path: RGB MBTiles or rasterio-readable source.
        lulc_paths: Single-band LULC rasters/VRTs with class ids.
        return_w_conf: Include ``w_conf`` in the returned mask dict.
        alpha: Entropy component weight for ``W_conf``.
        use_border: Include the BAGS border-distance component in ``W_conf``.
        entropy_norm: ``"max_entropy"`` or ``"minmax"``.
        border_radius: Radius in pixels for BAGS border weighting.
        num_classes: Number of segmentation classes.
        **kwargs: Other arguments accepted by ``MBTilesMaskWindowedDataset``.

    Returns:
        Dict with ``image`` tensor ``(3,H,W)`` and ``mask`` dict containing
        ``mask`` tensor ``(C,H,W)`` and optionally ``w_conf`` tensor
        ``(1,H,W)``.

    Example YAML:
        ```yaml
        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.mbtiles_soft_label_dataset.MBTilesSoftLabelMaskWindowedDataset
          mbtiles_path: /data/tiles.mbtiles
          mask_dir: /data/masks
          window_index_cache: /data/coreset.csv
          lulc_paths: [/data/mapbiomas.vrt, /data/esri.vrt, /data/dw.vrt]
          num_classes: 6
          return_w_conf: true
          alpha: 0.6
          use_border: true
        ```
    """

    def __init__(
        self,
        mbtiles_path,
        lulc_paths: Sequence[Union[str, Path]],
        mask_paths: Optional[Sequence[str]] = None,
        mask_dir: Optional[str] = None,
        mask_extension: str = ".tif",
        patch_size: int = 512,
        stride: Optional[int] = None,
        selected_bands: Optional[Sequence[int]] = None,
        image_dtype: str = "uint8",
        image_resampling: str = "bilinear",
        num_classes: int = 6,
        lulc_resampling: str = "nearest",
        return_w_conf: bool = True,
        alpha: float = 0.6,
        use_border: bool = True,
        entropy_norm: str = "max_entropy",
        border_radius: int = 10,
        augmentation_list=None,
        data_loader=None,
        return_metadata: bool = True,
        window_index_cache: Optional[str] = None,
        seed: Optional[int] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            mbtiles_path=mbtiles_path,
            mask_paths=mask_paths,
            mask_dir=mask_dir,
            mask_extension=mask_extension,
            patch_size=patch_size,
            stride=stride,
            selected_bands=selected_bands,
            image_dtype=image_dtype,
            image_resampling=image_resampling,
            n_classes=num_classes,
            augmentation_list=None,
            data_loader=data_loader,
            return_metadata=return_metadata,
            window_index_cache=window_index_cache,
            **kwargs,
        )
        self.lulc_paths = [Path(p) for p in lulc_paths]
        self.num_classes = num_classes
        self.lulc_resampling = lulc_resampling
        self.return_w_conf = return_w_conf
        self.alpha = alpha
        self.use_border = use_border
        self.entropy_norm = entropy_norm
        self.border_radius = border_radius
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        self.transform = None
        if augmentation_list is not None:
            base = load_augmentation_object(augmentation_list, seed=seed)
            additional_targets: Dict[str, str] = {"p_soft": "mask"}
            if return_w_conf:
                additional_targets["w_conf"] = "mask"
            self.transform = A.Compose(
                base.transforms,
                additional_targets=additional_targets,
            )

    def _cache_key(self, record: dict) -> str:
        key = f"{record['mask_path']}|{record['row_off']}|{record['col_off']}"
        return hashlib.sha1(key.encode()).hexdigest()[:16] + ".npy"

    def _load_cached_p_soft(self, record: dict) -> Optional[np.ndarray]:
        if self.cache_dir is None:
            return None
        cache_file = self.cache_dir / self._cache_key(record)
        if cache_file.exists():
            return np.load(cache_file)
        return None

    def _save_cached_p_soft(self, record: dict, p_soft: np.ndarray) -> None:
        if self.cache_dir is None:
            return
        cache_file = self.cache_dir / self._cache_key(record)
        tmp_file = cache_file.with_name(f"{cache_file.stem}_{os.getpid()}.tmp")
        with open(tmp_file, "wb") as f:
            np.save(f, p_soft)
        tmp_file.rename(cache_file)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one MBTiles RGB patch with on-the-fly soft labels.

        Args:
            idx: Dataset item index.

        Returns:
            Dict compatible with ``SoftLabelModel``.

        Raises:
            IndexError: If *idx* is outside the dataset range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for size {len(self)}.")

        record = self.windows[idx]
        mask_path = record["mask_path"]
        window = rasterio.windows.Window(
            record["col_off"],
            record["row_off"],
            record["width"],
            record["height"],
        )

        with rasterio.open(mask_path) as mask_src:
            image_chw = read_source_aligned_to_mask_window(
                source_path=self.mbtiles_path,
                mask_src=mask_src,
                window=window,
                selected_bands=self.selected_bands,
                image_dtype=self.image_dtype,
                image_resampling=self.image_resampling,
            )
            bags_mask = read_mask_window(mask_src, window, n_classes=self.n_classes)

            p_soft = self._load_cached_p_soft(record)
            if p_soft is None:
                lulc_maps = [
                    read_source_aligned_to_mask_window(
                        source_path=path,
                        mask_src=mask_src,
                        window=window,
                        selected_bands=[1],
                        image_dtype="uint8",
                        image_resampling=self.lulc_resampling,
                    )[0]
                    for path in self.lulc_paths
                ]
                p_soft = compute_p_soft([bags_mask, *lulc_maps], self.num_classes)
                self._save_cached_p_soft(record, p_soft)

        w_conf = None
        if self.return_w_conf:
            w_conf = compute_w_conf(
                p_soft,
                alpha=self.alpha,
                use_border=self.use_border,
                entropy_norm=self.entropy_norm,
                bags_mask=bags_mask,
                border_radius=self.border_radius,
            )

        image_hwc = np.ascontiguousarray(image_chw.transpose(1, 2, 0))
        p_soft_hwc = np.ascontiguousarray(p_soft.transpose(1, 2, 0))
        w_conf_hwc = (
            None if w_conf is None else np.ascontiguousarray(w_conf.transpose(1, 2, 0))
        )

        if self.transform is not None:
            kwargs: Dict[str, Any] = {"image": image_hwc, "p_soft": p_soft_hwc}
            if w_conf_hwc is not None:
                kwargs["w_conf"] = w_conf_hwc
            result = self.transform(**kwargs)
            image_out = result["image"]
            p_soft_out = result["p_soft"]
            w_conf_out = result.get("w_conf")
        else:
            image_out = image_hwc
            p_soft_out = p_soft_hwc
            w_conf_out = w_conf_hwc

        mask_dict: Dict[str, torch.Tensor] = {
            "mask": SoftLabelDataset._to_chw_float(p_soft_out)
        }
        if w_conf_out is not None:
            mask_dict["w_conf"] = SoftLabelDataset._to_chw_float(w_conf_out)

        sample: Dict[str, Any] = {
            "image": SoftLabelDataset._image_to_chw_float(image_out),
            "mask": mask_dict,
            "path": str(mask_path),
        }
        if self.return_metadata:
            sample["metadata"] = {
                "mask_path": str(mask_path),
                "row_off": int(record["row_off"]),
                "col_off": int(record["col_off"]),
            }
        return sample
