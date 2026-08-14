# -*- coding: utf-8 -*-
"""Segmentation dataset with pre-selected crop windows and a rasterio-readable mask.

Windows are loaded from a pre-existing source (CSV/Parquet or vector file) rather
than generated via sliding window. The mask can be any rasterio-readable source —
a VRT mosaicking multiple GeoTIFFs, a single GeoTIFF, or another MBTile.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    _DTYPE_NORMALIZATION,
    _VALID_IMAGE_DTYPES,
    load_augmentation_object,
)
from pytorch_segmentation_models_trainer.dataset_loader.mbtiles_dataset import (
    _apply_color_map,
    _load_window_index_cache,
    _parse_color_map,
    _save_window_index_cache,
    _validate_cache_path,
)
from pytorch_segmentation_models_trainer.tools.mbtiles.alignment import (
    read_source_aligned_to_mask_window,
)

try:
    import geopandas as gpd

    _HAS_GEO = True
except ImportError:  # pragma: no cover
    _HAS_GEO = False

logger = logging.getLogger(__name__)

_VECTOR_EXTENSIONS = {".gpkg", ".shp", ".geojson", ".json", ".gml", ".kml"}
_TABULAR_EXTENSIONS = {".csv", ".parquet"}


class MBTilesCropsGeoTifMaskDataset(Dataset):
    """Segmentation dataset with pre-selected crop windows and a rasterio-readable mask.

    Windows are loaded from a pre-existing source rather than generated via
    sliding window:

    - **CSV / Parquet**: must contain ``row_off`` and ``col_off`` columns with
      pixel-space coordinates relative to the image raster.
    - **Vector file** (GeoPackage, Shapefile, GeoJSON, …): each feature defines
      one window. The feature bounding-box centre is projected to image pixel
      space and a ``patch_size × patch_size`` window is centred on that point
      (bbox-snap), then clamped to the raster extent.

    The mask can be any rasterio-readable source — a VRT mosaicking multiple
    GeoTIFFs, a single GeoTIFF, or another MBTile. It is warped onto each image
    window via ``WarpedVRT`` so image and mask are always spatially aligned and
    share the same fixed size, regardless of CRS or resolution differences.
    Mask resampling is always nearest-neighbour to preserve class-index integrity.

    Args:
        image_mbtiles_path: Path to the image raster (reference grid). Any
            rasterio-readable source is accepted (MBTile, GeoTIFF, VRT, …).
        mask_path: Path to the mask raster (VRT, GeoTIFF, MBTile, …).
        crops_path: Path to a CSV/Parquet file with pixel-space windows, or a
            vector file where each feature defines one window.
        patch_size: Patch height and width in image pixels. Always fixed.
        color_map: List of ``[R, G, B, class_idx]`` entries for multi-band
            color-coded masks. Required when the mask has more than one band.
        n_classes: Number of classes for single-band masks. When ``2``, all
            non-zero values are mapped to foreground class ``1``.
        selected_bands: Optional 1-based image band indices. ``None`` reads all
            bands.
        image_dtype: Output numpy dtype for imagery: ``"uint8"``, ``"uint16"``,
            ``"float32"``, or ``"native"``.
        image_resampling: Rasterio resampling method for source imagery.
            ``"bilinear"`` is recommended for continuous imagery.
        crops_layer: Layer name when ``crops_path`` is a multi-layer vector file.
        col_off_key: Column name in CSV/Parquet for the horizontal pixel offset.
        row_off_key: Column name in CSV/Parquet for the vertical pixel offset.
        augmentation_list: Albumentations transform configs applied to
            ``image`` (HWC) and ``mask`` (HW).
        data_loader: DataLoader configuration stored for the Lightning model.
        return_metadata: When ``True``, each sample includes a ``"metadata"``
            dict with ``row_off`` and ``col_off``.
        window_index_cache: Optional ``.csv`` or ``.parquet`` path for
            persisting the computed window index.
        **kwargs: Compatibility kwargs for Hydra instantiation.

    Returns:
        Each ``__getitem__`` returns a dict with:

        - ``"image"``: ``(C, H, W)`` float32 tensor.
        - ``"mask"``: ``(H, W)`` int64 tensor with class indices.
        - ``"metadata"`` *(only when* ``return_metadata=True`` *)*: dict with
          keys ``row_off`` and ``col_off``.

    Raises:
        ValueError: If the mask has more than one band and ``color_map`` is not
            provided.
        ValueError: If ``image_dtype`` is not one of the accepted values.
        ValueError: If ``patch_size`` is not a positive integer.

    Example YAML:
        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.mbtiles_crops_dataset.MBTilesCropsGeoTifMaskDataset
          image_mbtiles_path: /data/imagery.mbtiles
          mask_path: /data/masks/mask.vrt
          crops_path: /data/crops.csv
          patch_size: 256
          color_map:
            - [255, 0, 0, 1]
            - [0, 255, 0, 2]
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
        image_mbtiles_path: Union[str, Path],
        mask_path: Union[str, Path],
        crops_path: Union[str, Path],
        patch_size: int,
        color_map: Optional[List[List[int]]] = None,
        n_classes: int = 2,
        selected_bands: Optional[List[int]] = None,
        image_dtype: str = "uint8",
        image_resampling: str = "bilinear",
        crops_layer: Optional[str] = None,
        col_off_key: str = "col_off",
        row_off_key: str = "row_off",
        augmentation_list=None,
        data_loader=None,
        return_metadata: bool = False,
        window_index_cache: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        if image_dtype not in _VALID_IMAGE_DTYPES:
            raise ValueError(
                f"image_dtype '{image_dtype}' is invalid. "
                f"Accepted values: {sorted(_VALID_IMAGE_DTYPES)}"
            )
        if patch_size <= 0:
            raise ValueError("patch_size must be a positive integer.")

        self.image_mbtiles_path = Path(image_mbtiles_path)
        self.mask_path = Path(mask_path)
        self.crops_path = Path(crops_path)
        self.patch_size = int(patch_size)
        self.n_classes = n_classes
        self.selected_bands = list(selected_bands) if selected_bands else None
        self.image_dtype = image_dtype
        self.image_resampling = image_resampling
        self.crops_layer = crops_layer
        self.col_off_key = col_off_key
        self.row_off_key = row_off_key
        self.data_loader = data_loader
        self.return_metadata = return_metadata

        self.color_map: Optional[Dict[Tuple[int, int, int], int]] = (
            _parse_color_map(color_map) if color_map is not None else None
        )

        if window_index_cache is not None:
            self.window_index_cache = Path(window_index_cache)
            _validate_cache_path(self.window_index_cache)
        else:
            self.window_index_cache = None

        self.transform = (
            None
            if augmentation_list is None
            else load_augmentation_object(augmentation_list)
        )

        with rasterio.open(self.image_mbtiles_path) as src:
            self._image_crs = src.crs
            self._image_transform = src.transform
            self._image_width = src.width
            self._image_height = src.height

        with rasterio.open(self.mask_path) as mask_src:
            mask_band_count = mask_src.count

        if mask_band_count > 1 and self.color_map is None:
            raise ValueError(
                f"mask_path '{self.mask_path}' has {mask_band_count} bands but no "
                "color_map was provided. Multi-band masks require a color_map "
                "with [R, G, B, class_idx] entries."
            )

        if self.window_index_cache is not None and self.window_index_cache.exists():
            self._index = _load_window_index_cache(self.window_index_cache)
        else:
            self._index = self._load_window_index()
            if self.window_index_cache is not None:
                _save_window_index_cache(self.window_index_cache, self._index)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one image/mask patch.

        Args:
            idx: Dataset index.

        Returns:
            Dict with ``image`` and ``mask`` tensors (and optionally
            ``metadata``).

        Raises:
            IndexError: If *idx* is outside ``[0, len(self))``.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        rec = self._index[idx]
        window = Window(
            rec["col_off"], rec["row_off"], self.patch_size, self.patch_size
        )

        with rasterio.open(self.image_mbtiles_path) as img_src:
            bands = (
                self.selected_bands
                if self.selected_bands is not None
                else list(range(1, img_src.count + 1))
            )
            image_chw = img_src.read(bands, window=window)

            mask_chw = read_source_aligned_to_mask_window(
                source_path=self.mask_path,
                mask_src=img_src,
                window=window,
                selected_bands=None,
                image_dtype="uint8",
                image_resampling="nearest",
            )

        if self.image_dtype != "native":
            image_chw = image_chw.astype(np.dtype(self.image_dtype))

        mask_hwc = np.transpose(mask_chw, (1, 2, 0))
        if self.color_map is not None:
            mask_class = _apply_color_map(mask_hwc, self.color_map)
        else:
            mask_class = mask_hwc[:, :, 0].astype(np.uint8)
            if self.n_classes == 2:
                mask_class = (mask_class > 0).astype(np.uint8)

        image_hwc = np.transpose(image_chw, (1, 2, 0))

        if self.transform is not None:
            out = self.transform(image=image_hwc, mask=mask_class)
            img_t: torch.Tensor = out["image"]
            mask_t: torch.Tensor = out["mask"]
            if not isinstance(mask_t, torch.Tensor):
                mask_t = torch.from_numpy(np.asarray(mask_t))
            mask_t = mask_t.long()
        else:
            img_t, mask_t = self._to_tensors(image_hwc, mask_class)

        sample: Dict[str, Any] = {"image": img_t, "mask": mask_t}
        if self.return_metadata:
            sample["metadata"] = {
                "row_off": rec["row_off"],
                "col_off": rec["col_off"],
            }
        return sample

    # ------------------------------------------------------------------
    # Window index loading
    # ------------------------------------------------------------------

    def _load_window_index(self) -> List[Dict[str, Any]]:
        ext = self.crops_path.suffix.lower()
        if ext in _TABULAR_EXTENSIONS:
            return self._windows_from_tabular(self.crops_path)
        if ext in _VECTOR_EXTENSIONS:
            return self._windows_from_vector(self.crops_path)
        # Unknown extension: try CSV reading first, then vector.
        try:
            return self._windows_from_csv_fallback(self.crops_path)
        except Exception:
            return self._windows_from_vector(self.crops_path)

    def _windows_from_csv_fallback(self, path: Path) -> List[Dict[str, Any]]:
        """Read windows as CSV regardless of file extension (fallback path)."""
        import pandas as pd

        df = pd.read_csv(path)
        missing = {self.row_off_key, self.col_off_key} - set(df.columns)
        if missing:
            raise ValueError(
                f"crops_path '{path}' is missing required columns: {sorted(missing)}"
            )
        return [
            {
                "row_off": int(row[self.row_off_key]),
                "col_off": int(row[self.col_off_key]),
            }
            for _, row in df.iterrows()
        ]

    def _windows_from_tabular(self, path: Path) -> List[Dict[str, Any]]:
        """Load window index from a CSV or Parquet file.

        Args:
            path: Path to the CSV or Parquet file.

        Returns:
            List of ``{"row_off": int, "col_off": int}`` dicts.

        Raises:
            ValueError: If required columns are missing.
        """
        import pandas as pd

        df = (
            pd.read_csv(path)
            if path.suffix.lower() == ".csv"
            else pd.read_parquet(path)
        )
        missing = {self.row_off_key, self.col_off_key} - set(df.columns)
        if missing:
            raise ValueError(
                f"crops_path '{path}' is missing required columns: {sorted(missing)}"
            )
        return [
            {
                "row_off": int(row[self.row_off_key]),
                "col_off": int(row[self.col_off_key]),
            }
            for _, row in df.iterrows()
        ]

    def _windows_from_vector(self, path: Path) -> List[Dict[str, Any]]:
        """Load window index from a vector file (bbox-snap per feature).

        Each feature's bounding-box centre is projected to image pixel space and
        a ``patch_size × patch_size`` window is centred on that point, then
        clamped to the raster extent.

        Args:
            path: Path to the vector file.

        Returns:
            List of ``{"row_off": int, "col_off": int}`` dicts.

        Raises:
            ImportError: If geopandas is not installed.
            ValueError: If the vector file has no CRS.
        """
        if not _HAS_GEO:  # pragma: no cover
            raise ImportError(
                "geopandas and shapely are required to load windows from a vector file. "
                "Install them with: pip install geopandas shapely"
            )
        read_kwargs: Dict[str, Any] = {}
        if self.crops_layer is not None:
            read_kwargs["layer"] = self.crops_layer
        crops = gpd.read_file(path, **read_kwargs)

        if crops.crs is None:
            raise ValueError(f"Vector file '{path}' has no CRS defined.")
        if crops.crs != self._image_crs:
            crops = crops.to_crs(self._image_crs)

        inv_transform = ~self._image_transform
        ps = self.patch_size
        records: List[Dict[str, Any]] = []

        for geom in crops.geometry:
            if geom is None or geom.is_empty:
                continue
            lon_min, lat_min, lon_max, lat_max = geom.bounds
            corners = [
                inv_transform * (lon_min, lat_min),
                inv_transform * (lon_min, lat_max),
                inv_transform * (lon_max, lat_min),
                inv_transform * (lon_max, lat_max),
            ]
            cols = [c[0] for c in corners]
            rows = [c[1] for c in corners]
            center_col = (min(cols) + max(cols)) / 2.0
            center_row = (min(rows) + max(rows)) / 2.0
            col_off = int(round(center_col - ps / 2.0))
            row_off = int(round(center_row - ps / 2.0))
            col_off = max(0, min(col_off, self._image_width - ps))
            row_off = max(0, min(row_off, self._image_height - ps))
            records.append({"row_off": row_off, "col_off": col_off})

        if not records:
            logger.warning(
                "MBTilesCropsGeoTifMaskDataset: no valid windows found in '%s'.", path
            )
        return records

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------

    def _to_tensors(
        self, image_hwc: np.ndarray, mask_hw: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_t = torch.from_numpy(
            np.ascontiguousarray(image_hwc.transpose(2, 0, 1))
        ).float()
        norm = _DTYPE_NORMALIZATION.get(self.image_dtype)
        if norm is not None:
            img_t = img_t / norm
        return img_t, torch.from_numpy(mask_hw.copy()).long()
