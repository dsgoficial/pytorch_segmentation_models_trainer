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

    Three independent disk caches (all optional) short-circuit expensive
    operations after the first epoch:

    - ``cache_dir``: caches ``P_soft`` — skips reading all LULC VRTs.
    - ``img_cache_dir``: caches the RGB patch — skips MBTiles read and
      reprojection.
    - ``wconf_cache_dir``: caches ``W_conf`` — skips entropy computation and
      border distance transform.

    When all three caches are warm the mask raster is never opened.  Each cache
    uses a PID-qualified temp file and an atomic rename, so concurrent DataLoader
    workers writing the same tile are safe.

    Args:
        mbtiles_path: RGB MBTiles or rasterio-readable source.
        lulc_paths: Single-band LULC rasters/VRTs with class ids.
        return_w_conf: Include ``w_conf`` in the returned mask dict.
        alpha: Entropy component weight for ``W_conf``.
        use_border: Include the BAGS border-distance component in ``W_conf``.
        entropy_norm: ``"max_entropy"`` or ``"minmax"``.
        border_radius: Radius in pixels for BAGS border weighting.
        num_classes: Number of segmentation classes.
        cache_dir: Directory for ``P_soft`` cache (shared across experiments).
        img_cache_dir: Directory for image patch cache (shared across
            experiments with the same ``selected_bands``/``image_dtype``).
        wconf_cache_dir: Directory for ``W_conf`` cache. Keys encode
            ``alpha``, ``use_border``, ``entropy_norm``, and ``border_radius``
            so experiments with different parameters coexist safely.
        **kwargs: Other arguments accepted by ``MBTilesMaskWindowedDataset``.

    Returns:
        Dict with ``image`` tensor ``(3,H,W)`` and ``mask`` dict containing
        ``mask`` tensor ``(C,H,W)`` and optionally ``w_conf`` tensor
        ``(1,H,W)``.
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
        source_weights: Optional[Sequence[float]] = None,
        classes_to_soften: Optional[Sequence[int]] = None,
        augmentation_list=None,
        data_loader=None,
        return_metadata: bool = True,
        window_index_cache: Optional[str] = None,
        seed: Optional[int] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        img_cache_dir: Optional[Union[str, Path]] = None,
        wconf_cache_dir: Optional[Union[str, Path]] = None,
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
        self.source_weights = list(source_weights) if source_weights is not None else None
        self.classes_to_soften = set(classes_to_soften) if classes_to_soften is not None else None
        self.cache_dir = self._init_cache_dir(cache_dir)
        self.img_cache_dir = self._init_cache_dir(img_cache_dir)
        self.wconf_cache_dir = self._init_cache_dir(wconf_cache_dir)

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

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_cache_dir(path: Optional[Union[str, Path]]) -> Optional[Path]:
        if path is None:
            return None
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def _write_array_to_cache(cache_file: Path, array: np.ndarray) -> None:
        tmp_file = cache_file.with_name(f"{cache_file.stem}_{os.getpid()}.tmp")
        with open(tmp_file, "wb") as f:
            np.save(f, array)
        tmp_file.rename(cache_file)

    def _psoft_cache_key(self, record: dict) -> str:
        key = f"{record['mask_path']}|{record['row_off']}|{record['col_off']}|{self.source_weights}"
        return hashlib.sha1(key.encode()).hexdigest()[:16] + ".npy"

    def _img_cache_key(self, record: dict) -> str:
        key = (
            f"{record['mask_path']}|{record['row_off']}|{record['col_off']}"
            f"|{self.selected_bands}|{self.image_dtype}|{self.image_resampling}"
        )
        return hashlib.sha1(key.encode()).hexdigest()[:16] + ".npy"

    def _wconf_cache_key(self, record: dict) -> str:
        key = (
            f"{record['mask_path']}|{record['row_off']}|{record['col_off']}"
            f"|{self.alpha}|{self.use_border}|{self.entropy_norm}|{self.border_radius}"
        )
        return hashlib.sha1(key.encode()).hexdigest()[:16] + ".npy"

    def _load_cached_p_soft(self, record: dict) -> Optional[np.ndarray]:
        if self.cache_dir is None:
            return None
        f = self.cache_dir / self._psoft_cache_key(record)
        return np.load(f) if f.exists() else None

    def _save_cached_p_soft(self, record: dict, p_soft: np.ndarray) -> None:
        if self.cache_dir is None:
            return
        self._write_array_to_cache(
            self.cache_dir / self._psoft_cache_key(record), p_soft
        )

    def _load_cached_img(self, record: dict) -> Optional[np.ndarray]:
        if self.img_cache_dir is None:
            return None
        f = self.img_cache_dir / self._img_cache_key(record)
        return np.load(f) if f.exists() else None

    def _save_cached_img(self, record: dict, image_chw: np.ndarray) -> None:
        if self.img_cache_dir is None:
            return
        self._write_array_to_cache(
            self.img_cache_dir / self._img_cache_key(record), image_chw
        )

    def _load_cached_wconf(self, record: dict) -> Optional[np.ndarray]:
        if self.wconf_cache_dir is None:
            return None
        f = self.wconf_cache_dir / self._wconf_cache_key(record)
        return np.load(f) if f.exists() else None

    def _save_cached_wconf(self, record: dict, w_conf: np.ndarray) -> None:
        if self.wconf_cache_dir is None:
            return
        self._write_array_to_cache(
            self.wconf_cache_dir / self._wconf_cache_key(record), w_conf
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

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

        image_chw = self._load_cached_img(record)
        p_soft = self._load_cached_p_soft(record)
        w_conf = self._load_cached_wconf(record) if self.return_w_conf else None

        need_rasterio = (
            image_chw is None
            or p_soft is None
            or (self.return_w_conf and w_conf is None)
            or self.classes_to_soften is not None
        )

        bags_mask = None
        if need_rasterio:
            with rasterio.open(mask_path) as mask_src:
                if image_chw is None:
                    image_chw = read_source_aligned_to_mask_window(
                        source_path=self.mbtiles_path,
                        mask_src=mask_src,
                        window=window,
                        selected_bands=self.selected_bands,
                        image_dtype=self.image_dtype,
                        image_resampling=self.image_resampling,
                    )
                    self._save_cached_img(record, image_chw)

                if p_soft is None or (self.return_w_conf and w_conf is None) or self.classes_to_soften is not None:
                    bags_mask = read_mask_window(
                        mask_src, window, n_classes=self.n_classes
                    )

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
                    p_soft = compute_p_soft([bags_mask, *lulc_maps], self.num_classes, self.source_weights)
                    self._save_cached_p_soft(record, p_soft)

        if self.return_w_conf and w_conf is None:
            w_conf = compute_w_conf(
                p_soft,
                alpha=self.alpha,
                use_border=self.use_border,
                entropy_norm=self.entropy_norm,
                bags_mask=bags_mask,
                border_radius=self.border_radius,
            )
            self._save_cached_wconf(record, w_conf)

        # For E6/E7: override soft labels with 1-hot for classes not in classes_to_soften.
        # pixels where BAGS label is reliable (not grassland/cropland) get hard labels.
        if self.classes_to_soften is not None and bags_mask is not None:
            hard_mask = ~np.isin(bags_mask, list(self.classes_to_soften)) & (bags_mask < self.num_classes)
            if hard_mask.any():
                clipped = np.clip(bags_mask, 0, self.num_classes - 1)
                one_hot = np.eye(self.num_classes, dtype=p_soft.dtype)[clipped].transpose(2, 0, 1)
                p_soft = np.where(hard_mask[np.newaxis], one_hot, p_soft)
                if w_conf is not None:
                    w_conf = np.where(hard_mask[np.newaxis], np.ones_like(w_conf), w_conf)

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
