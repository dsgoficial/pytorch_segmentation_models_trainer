# -*- coding: utf-8 -*-
"""Segmentation dataset backed by paired image/mask MBTiles rasters.

Both image and mask are rasterio-readable sources (typically MBTiles files
opened via the GDAL MBTILES driver).  Sliding-window patches are generated
over the image raster at its native (maximum) resolution and filtered so that
only patches **fully contained** within the input vector polygons are indexed
for training.

The mask source is warped onto each image window via WarpedVRT so that image
and mask are always spatially aligned regardless of their original projections.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from torch.utils.data import Dataset

from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    _DTYPE_NORMALIZATION,
    _VALID_IMAGE_DTYPES,
    load_augmentation_object,
)
from pytorch_segmentation_models_trainer.tools.mbtiles.alignment import (
    read_source_aligned_to_mask_window,
)

try:
    import geopandas as gpd
    from shapely.geometry import box as shapely_box

    _HAS_GEO = True
except ImportError:  # pragma: no cover
    _HAS_GEO = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Color-map helpers
# ---------------------------------------------------------------------------


def _parse_color_map(
    color_map: List[List[int]],
) -> Dict[Tuple[int, int, int], int]:
    """Parse a list of ``[R, G, B, class_idx]`` entries into a lookup dict.

    Args:
        color_map: Each entry must be a four-element list ``[R, G, B, class_idx]``.

    Returns:
        Mapping from ``(R, G, B)`` tuples to integer class indices.

    Raises:
        ValueError: If any entry has fewer than four elements.
    """
    result: Dict[Tuple[int, int, int], int] = {}
    for entry in color_map:
        if len(entry) < 4:
            raise ValueError(
                f"Each color_map entry must be [R, G, B, class_idx]; got {entry}"
            )
        r, g, b, cls = int(entry[0]), int(entry[1]), int(entry[2]), int(entry[3])
        result[(r, g, b)] = cls
    return result


def _apply_color_map(
    mask_hwc: np.ndarray,
    color_map: Dict[Tuple[int, int, int], int],
) -> np.ndarray:
    """Convert an RGB/RGBA color-coded mask to a single-channel class-index array.

    Pixels whose RGB triplet is absent from *color_map* are assigned class
    ``0`` (background).

    Args:
        mask_hwc: Array of shape ``(H, W, C)`` with ``C >= 3``, dtype uint8.
        color_map: Mapping from ``(R, G, B)`` to integer class index.

    Returns:
        ``uint8`` array of shape ``(H, W)`` with class indices.
    """
    h, w = mask_hwc.shape[:2]
    rgb = mask_hwc[:, :, :3].astype(np.uint32)
    packed = rgb[:, :, 0] * 65536 + rgb[:, :, 1] * 256 + rgb[:, :, 2]
    result = np.zeros((h, w), dtype=np.uint8)
    for (r, g, b), class_idx in color_map.items():
        result[packed == (r * 65536 + g * 256 + b)] = class_idx
    return result


# ---------------------------------------------------------------------------
# Window-index cache helpers
# ---------------------------------------------------------------------------


def _validate_cache_path(path: Path) -> None:
    if path.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError(
            "window_index_cache must end with '.csv' or '.parquet'; "
            f"got '{path.suffix}'."
        )


def _save_window_index_cache(path: Path, records: List[Dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    logger.info(
        "MBTilesPolygonDataset: window index cached to %s (%d windows)",
        path,
        len(records),
    )


def _load_window_index_cache(path: Path) -> List[Dict[str, Any]]:
    import pandas as pd

    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    required = {"row_off", "col_off"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"window_index_cache at '{path}' is missing columns: {sorted(missing)}"
        )
    records = df.to_dict(orient="records")
    for rec in records:
        rec["row_off"] = int(rec["row_off"])
        rec["col_off"] = int(rec["col_off"])
    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MBTilesPolygonDataset(Dataset):
    """Segmentation dataset driven by sliding-window patches over paired MBTiles rasters.

    The image raster is treated as the reference grid.  A sliding-window index
    is generated over its full extent (at native/maximum resolution) and then
    filtered so that only windows **fully contained** within the input vector
    polygons are kept for training.

    For each valid window the mask source is warped onto the exact image-window
    grid via a WarpedVRT, keeping image and mask spatially aligned regardless
    of their original projections or resolutions.

    Mask rasters are expected to be RGB or RGBA color-coded images.  The
    ``color_map`` parameter maps each ``[R, G, B]`` triplet to an integer class
    index; unmatched pixels default to class ``0`` (background).

    Args:
        image_mbtiles_path: Path to the image raster.  Any source readable by
            rasterio is accepted (MBTiles, GeoTIFF, VRT, …).
        mask_mbtiles_path: Path to the mask raster.  Must be color-coded RGB or
            RGBA (see *color_map*).
        regions_path: Path to a vector file (Shapefile, GeoJSON, GeoPackage, …)
            containing the polygon regions that define valid training areas.
            Only windows whose geographic extent is **fully contained** within
            the union of these polygons are indexed.
        patch_size: Patch height and width in image pixels.
        stride: Sliding-window step in image pixels.  Defaults to *patch_size*
            (non-overlapping grid).
        color_map: List of ``[R, G, B, class_idx]`` entries mapping mask pixel
            colors to integer class indices.  Background (unmatched) pixels are
            assigned class ``0``.
        selected_bands: Optional 1-based image band indices.  ``None`` reads all
            bands.
        image_dtype: Output numpy dtype for imagery.  One of ``"uint8"``,
            ``"uint16"``, ``"float32"``, or ``"native"``.  ``uint8`` and
            ``uint16`` values are divided by their max when no augmentation
            pipeline is provided.
        mask_resampling: Rasterio resampling method used when warping the mask
            onto the image window.  ``"nearest"`` is strongly recommended for
            class-index or color-coded masks.
        augmentation_list: Albumentations transform configs applied synchronously
            to ``image`` (HWC) and ``mask`` (HW).
        data_loader: DataLoader configuration stored for the Lightning model.
        return_metadata: When ``True``, each sample includes a ``"metadata"``
            dict with ``row_off`` and ``col_off`` in image pixel coordinates.
        window_index_cache: Optional ``.csv`` or ``.parquet`` path for
            persisting the computed window index.  When the file exists, the
            polygon-containment pass is skipped on subsequent runs.
        regions_layer: Layer name to read from the vector file when it contains
            multiple layers.
        index_build_workers: Number of threads used to build the window index in
            parallel (one thread per polygon).  ``None`` (default) auto-selects
            ``min(n_polygons, cpu_count)``.  Set to ``1`` to force serial
            processing.  Threading is safe here because it runs entirely inside
            ``__init__`` and all threads complete before the dataset is returned
            to the caller — DataLoader workers are spawned only afterward and
            have no conflict.  Do **not** use threading inside ``__getitem__``;
            DataLoader's ``num_workers`` already provides that parallelism.
        **kwargs: Compatibility kwargs for Hydra instantiation.

    Returns:
        Each ``__getitem__`` returns a dict with:

        - ``"image"``: ``(C, H, W)`` float32 tensor.
        - ``"mask"``: ``(H, W)`` int64 tensor with class indices.
        - ``"metadata"`` *(only when* ``return_metadata=True`` *)*: dict with
          keys ``row_off`` and ``col_off``.

    Example YAML:
        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.mbtiles_dataset.MBTilesPolygonDataset
          image_mbtiles_path: /data/imagery.mbtiles
          mask_mbtiles_path: /data/masks.mbtiles
          regions_path: /data/regions.gpkg
          patch_size: 512
          stride: 512
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
        mask_mbtiles_path: Union[str, Path],
        regions_path: Union[str, Path],
        patch_size: int,
        stride: Optional[int] = None,
        color_map: Optional[List[List[int]]] = None,
        selected_bands: Optional[List[int]] = None,
        image_dtype: str = "uint8",
        mask_resampling: str = "nearest",
        augmentation_list=None,
        data_loader=None,
        return_metadata: bool = False,
        window_index_cache: Optional[Union[str, Path]] = None,
        regions_layer: Optional[str] = None,
        index_build_workers: Optional[int] = None,
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
        self.mask_mbtiles_path = Path(mask_mbtiles_path)
        self.regions_path = Path(regions_path)
        self.patch_size = int(patch_size)
        self.stride = int(stride) if stride is not None else self.patch_size
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer.")

        self.color_map: Optional[Dict[Tuple[int, int, int], int]] = (
            _parse_color_map(color_map) if color_map is not None else None
        )
        self.selected_bands = list(selected_bands) if selected_bands else None
        self.image_dtype = image_dtype
        self.mask_resampling = mask_resampling
        self.data_loader = data_loader
        self.return_metadata = return_metadata
        self.regions_layer = regions_layer
        self._index_build_workers = index_build_workers

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

        # Read image raster metadata (CRS, transform, dimensions).
        with rasterio.open(self.image_mbtiles_path) as src:
            self._image_crs = src.crs
            self._image_transform = src.transform
            self._image_width = src.width
            self._image_height = src.height

        if self.window_index_cache is not None and self.window_index_cache.exists():
            self._index = _load_window_index_cache(self.window_index_cache)
        else:
            self._index = self._build_window_index()
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
            # Read image patch in CHW format.
            bands = (
                self.selected_bands
                if self.selected_bands is not None
                else list(range(1, img_src.count + 1))
            )
            image_chw = img_src.read(bands, window=window)

            # Warp mask onto the exact image window grid.
            mask_chw = read_source_aligned_to_mask_window(
                source_path=self.mask_mbtiles_path,
                mask_src=img_src,
                window=window,
                selected_bands=None,  # read all bands for color-map lookup
                image_dtype="uint8",
                image_resampling=self.mask_resampling,
            )

        # Cast image dtype.
        if self.image_dtype != "native":
            image_chw = image_chw.astype(np.dtype(self.image_dtype))

        # Decode mask: CHW → HWC → single-channel class indices.
        mask_hwc = np.transpose(mask_chw, (1, 2, 0))
        if self.color_map is not None:
            mask_class = _apply_color_map(mask_hwc, self.color_map)
        else:
            # Single-band mask: use first band directly as class indices.
            mask_class = mask_hwc[:, :, 0].astype(np.uint8)

        # CHW → HWC for augmentation pipeline.
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_window_index(self) -> List[Dict[str, Any]]:
        """Generate and filter the sliding-window index using a polygon-first approach.

        For each polygon in the region file, only stride-aligned grid positions
        inside that polygon's pixel bounding box are tested for containment.
        This avoids iterating the full raster grid when polygons cover a small
        fraction of the image extent.

        When more than one polygon is present and ``index_build_workers != 1``,
        the per-polygon work runs in a ``ThreadPoolExecutor``.  Shapely
        containment checks release the GIL, so real parallelism is achieved.
        Results from all threads are deduplicated and sorted into row-major
        order before being returned.
        """
        if not _HAS_GEO:  # pragma: no cover
            raise ImportError(
                "geopandas and shapely are required to build the window index. "
                "Install them with: pip install geopandas shapely"
            )

        read_kwargs: Dict[str, Any] = {}
        if self.regions_layer is not None:
            read_kwargs["layer"] = self.regions_layer
        regions = gpd.read_file(self.regions_path, **read_kwargs)

        if regions.crs is None:
            raise ValueError(
                f"The vector file '{self.regions_path}' has no CRS defined."
            )
        if regions.crs != self._image_crs:
            regions = regions.to_crs(self._image_crs)

        geometries: List[Any] = list(regions.geometry)

        workers = self._index_build_workers
        if workers is None:
            workers = min(len(geometries), os.cpu_count() or 1)

        seen: set = set()
        all_windows: List[Dict[str, Any]] = []

        if workers <= 1 or len(geometries) <= 1:
            for geom in geometries:
                for rec in self._windows_for_polygon(geom):
                    key = (rec["row_off"], rec["col_off"])
                    if key not in seen:
                        all_windows.append(rec)
                        seen.add(key)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for windows in executor.map(self._windows_for_polygon, geometries):
                    for rec in windows:
                        key = (rec["row_off"], rec["col_off"])
                        if key not in seen:
                            all_windows.append(rec)
                            seen.add(key)

        all_windows.sort(key=lambda r: (r["row_off"], r["col_off"]))

        if not all_windows:
            logger.warning(
                "MBTilesPolygonDataset: no %dx%d patches are fully contained "
                "within '%s'. The dataset is empty.",
                self.patch_size,
                self.patch_size,
                self.regions_path,
            )

        return all_windows

    def _windows_for_polygon(self, polygon: Any) -> List[Dict[str, Any]]:
        """Return all stride-aligned windows fully contained within *polygon*.

        Iterates only the grid positions whose pixel coordinates fall inside the
        polygon's bounding box, then applies a strict ``contains`` test.
        Handles both ``Polygon`` and ``MultiPolygon`` geometries by processing
        each sub-geometry independently; duplicate windows from overlapping
        sub-geometries are suppressed via an internal ``seen`` set.

        Args:
            polygon: A shapely geometry already reprojected to the image CRS.

        Returns:
            List of ``{"row_off": int, "col_off": int}`` dicts in the order
            they were found (not sorted — caller is responsible for sorting).
        """
        parts: List[Any] = (
            list(polygon.geoms) if hasattr(polygon, "geoms") else [polygon]
        )

        inv_transform = ~self._image_transform
        ps = self.patch_size
        st = self.stride
        h = self._image_height
        w = self._image_width

        seen: set = set()
        windows: List[Dict[str, Any]] = []

        for part in parts:
            lon_min, lat_min, lon_max, lat_max = part.bounds

            # Map all four bbox corners to pixel space and take the extremes.
            # This is correct for any raster orientation (not just north-up).
            corners = [
                inv_transform * (lon_min, lat_min),
                inv_transform * (lon_min, lat_max),
                inv_transform * (lon_max, lat_min),
                inv_transform * (lon_max, lat_max),
            ]
            cols_px = [c[0] for c in corners]
            rows_px = [c[1] for c in corners]
            col_min_px = min(cols_px)
            col_max_px = max(cols_px)
            row_min_px = min(rows_px)
            row_max_px = max(rows_px)

            # Snap to stride grid; clamp to valid raster patch positions.
            col_start = max(0, int(np.ceil(col_min_px / st)) * st)
            col_end = min(w - ps, int(np.floor((col_max_px - ps) / st)) * st)
            row_start = max(0, int(np.ceil(row_min_px / st)) * st)
            row_end = min(h - ps, int(np.floor((row_max_px - ps) / st)) * st)

            if col_start > col_end or row_start > row_end:
                continue

            for row_off in range(row_start, row_end + 1, st):
                for col_off in range(col_start, col_end + 1, st):
                    key = (row_off, col_off)
                    if key in seen:
                        continue
                    left, bottom, right, top = window_bounds(
                        Window(col_off, row_off, ps, ps), self._image_transform
                    )
                    win_geom = shapely_box(left, bottom, right, top)
                    if part.contains(win_geom):
                        windows.append({"row_off": row_off, "col_off": col_off})
                        seen.add(key)

        return windows

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
