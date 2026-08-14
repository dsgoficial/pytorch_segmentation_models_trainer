# -*- coding: utf-8 -*-
"""Dataset that lazily computes P_soft and caches it to disk on first access.

P_soft is computed once per tile (from multiple LULC sources) and written as a
GeoTIFF to ``cache_dir``.  Subsequent accesses read from the cache.  The write
is atomic (write to ``.tmp``, then rename) so concurrent DataLoader workers do
not corrupt the cache.

W_conf is computed from the cached P_soft on every access, which keeps the cache
parameter-independent: ``alpha``, ``entropy_norm``, and ``use_border`` can be
changed without invalidating the cache.

This class accepts the same wide ``sources_csv`` format as the
``build-soft-labels`` CLI:
``tile_id,image_path,mask_path,<lulc columns...>``. ``tile_id`` is optional.
External LULC source columns are listed with ``lulc_keys``.

Windowed reads are supported via two mechanisms:

1. ``patch_size`` — enumerates a regular grid of patches over every tile.
2. ``windows_csv`` — reads an explicit list of windows from a CSV file, in
   either image pixel coordinates (``windows_coord_type="image"``) or
   geographic world coordinates (``windows_coord_type="world"``).
   When ``windows_csv`` is given it takes priority over ``patch_size``.

Returned sample format matches ``SoftLabelDataset``:
- ``"image"``: (3, H, W) float32 tensor in [0, 1]
- ``"mask"``: dict with
  - ``"mask"``: (C, H, W) float32 — P_soft probability distribution
  - ``"w_conf"``: (1, H, W) float32 — confidence weights in [0, 1]
- ``"path"``: image file path string
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
import rasterio
import rasterio.warp
import rasterio.windows
import torch
from torch.utils.data import Dataset

from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    load_augmentation_object,
)
from pytorch_segmentation_models_trainer.tools.soft_labels.build_soft_labels import (
    _CARTOGRAPHIC_SOURCE_NAME,
    _get_image_grid,
    _normalize_sources_dataframe,
    _reproject_lulc_to_image_grid,
    _write_geotiff,
    compute_p_soft,
    compute_w_conf,
)

logger = logging.getLogger(__name__)

# Type alias for an item in the windowed index.
# (tile_id, image_path, source_rows, window)
_WindowEntry = Tuple[str, str, "pd.DataFrame", "rasterio.windows.Window"]


class SoftLabelCachedDataset(Dataset):
    """Soft-label dataset with lazy P_soft computation and atomic disk cache.

    On first access to a tile, all LULC sources are reprojected to the image
    grid, P_soft is computed and written to ``cache_dir/{tile_id}.tif``.
    Subsequent accesses read P_soft directly from the cache.  W_conf is
    recomputed from P_soft on every access so that ``alpha``, ``entropy_norm``,
    and ``use_border`` can be varied without invalidating the cache.

    **Windowed read modes**

    When ``patch_size`` is given the dataset enumerates a regular grid of
    patches over every tile.  ``__len__`` equals the total number of patches
    and each ``__getitem__`` reads only the requested patch from the image and
    cache via rasterio's windowed reads.

    When ``windows_csv`` is given it overrides ``patch_size`` and the dataset
    uses the windows listed in the CSV.  Two coordinate types are supported:

    - ``windows_coord_type="image"`` (default): CSV columns
      ``tile_id, row_off, col_off, patch_h, patch_w`` in pixel space.
    - ``windows_coord_type="world"``: CSV columns
      ``tile_id, minx, miny, maxx, maxy`` in geographic space, with an
      optional ``crs`` column.  When ``crs`` is present and differs from the
      tile's CRS, the bounds are reprojected automatically using
      ``rasterio.warp.transform_bounds``.  When ``crs`` is absent the bounds
      are assumed to be in the tile's own CRS.

    In all windowed modes the P_soft cache is still stored as a full-tile
    GeoTIFF; only the requested window is loaded into memory per access.

    Args:
        sources_csv: Path to a wide CSV with one row per tile.
        cache_dir: Directory where P_soft GeoTIFFs are cached.  Created
            automatically if absent.
        num_classes: Number of land-cover classes (0-indexed).
        alpha: Entropy component weight for W_conf.
        use_border: When True (default), include the border-distance component
            in W_conf.  Set to False for the original paper formula.
        entropy_norm: Entropy normalization strategy. ``"max_entropy"``
            (default) normalizes by ``log(C)``; ``"minmax"`` applies per-tile
            min-max normalization (LaTeX Eq. 9-10, Experiment E4).
        image_key: Column name for image paths. Default ``"image_path"``.
        mask_key: Column name for the BAGS cartographic mask. Default
            ``"mask_path"``.
        lulc_keys: CSV columns with external LULC source paths. The soft label
            is an equal vote over ``mask_key`` plus these columns.
        augmentation_list: Albumentations transform configs.  Geometric
            transforms are applied identically to image, P_soft, and W_conf.
        data_loader: DataLoader configuration stored on the object for the
            Lightning model.
        n_first_rows_to_read: Limit the dataset to the first N unique tiles.
            Useful for debugging.
        patch_size: ``(H, W)`` of each patch.  Ignored when ``windows_csv``
            is also given.
        patch_stride: ``(H_stride, W_stride)`` for the regular-grid mode.
            Defaults to ``patch_size`` (no overlap).
        windows_csv: Path to a CSV of explicit windows.  Takes priority over
            ``patch_size``.
        windows_coord_type: Coordinate system of ``windows_csv``.
            ``"image"`` (default) — pixel row/col offsets.
            ``"world"`` — geographic bounding boxes.
        seed: Unused; kept for Hydra config compatibility.
        border_radius: Fixed radius *R* in pixels for the
            ``w_border_carta`` normalization (default 10).
        **kwargs: Reserved for Hydra compatibility.

    Example YAML::

        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader.soft_label_cached_dataset.SoftLabelCachedDataset
          sources_csv: /data/sources.csv
          cache_dir: /data/cache/p_soft
          num_classes: 6
          alpha: 0.6
          use_border: false
          entropy_norm: minmax
          windows_csv: /data/windows.csv
          windows_coord_type: world
          data_loader:
            batch_size: 16
            num_workers: 8
            shuffle: true
    """

    def __init__(
        self,
        sources_csv: Union[str, Path],
        cache_dir: Union[str, Path],
        num_classes: int,
        alpha: float = 0.6,
        use_border: bool = True,
        entropy_norm: str = "max_entropy",
        image_key: str = "image_path",
        mask_key: str = "mask_path",
        lulc_keys: Optional[List[str]] = None,
        augmentation_list: Optional[List] = None,
        data_loader: Optional[Any] = None,
        n_first_rows_to_read: Optional[int] = None,
        patch_size: Optional[Tuple[int, int]] = None,
        patch_stride: Optional[Tuple[int, int]] = None,
        windows_csv: Optional[Union[str, Path]] = None,
        windows_coord_type: str = "image",
        seed: Optional[int] = None,
        border_radius: int = 10,
        **kwargs,
    ) -> None:
        self.num_classes = num_classes
        self.alpha = alpha
        self.use_border = use_border
        self.entropy_norm = entropy_norm
        self.border_radius = border_radius
        self.image_key = image_key
        self.mask_key = mask_key
        self.lulc_keys = list(lulc_keys) if lulc_keys else []
        self.data_loader = data_loader or {}
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        df = _normalize_sources_dataframe(
            pd.read_csv(sources_csv),
            image_key=self.image_key,
            mask_key=self.mask_key,
            lulc_keys=self.lulc_keys,
        )
        grouped = [
            (tile_id, grp.iloc[0]["image_path"], grp)
            for tile_id, grp in df.groupby("tile_id", sort=False)
        ]
        if n_first_rows_to_read is not None:
            grouped = grouped[:n_first_rows_to_read]
        self._tiles: List[Tuple[str, str, pd.DataFrame]] = grouped

        # Build windowed index — windows_csv takes priority over patch_size.
        self._windows: Optional[List[_WindowEntry]] = None
        if windows_csv is not None:
            self._windows = self._build_windows_from_csv(
                Path(windows_csv), windows_coord_type
            )
        elif patch_size is not None:
            self._windows = self._build_windows_from_patch_size(
                patch_size, patch_stride
            )

        self.transform: Optional[A.Compose] = None
        if augmentation_list:
            base = load_augmentation_object(augmentation_list, seed=seed)
            self.transform = A.Compose(
                base.transforms,
                additional_targets={"p_soft": "mask", "w_conf": "mask"},
            )

    # ------------------------------------------------------------------
    # Window index builders
    # ------------------------------------------------------------------

    def _build_windows_from_patch_size(
        self,
        patch_size: Tuple[int, int],
        patch_stride: Optional[Tuple[int, int]],
    ) -> List[_WindowEntry]:
        ph, pw = patch_size
        sh, sw = patch_stride if patch_stride is not None else patch_size
        entries: List[_WindowEntry] = []
        for tile_id, image_path, rows in self._tiles:
            with rasterio.open(image_path) as src:
                tile_h, tile_w = src.height, src.width
            for row_off in range(0, tile_h - ph + 1, sh):
                for col_off in range(0, tile_w - pw + 1, sw):
                    win = rasterio.windows.Window(
                        col_off=col_off, row_off=row_off, width=pw, height=ph
                    )
                    entries.append((tile_id, image_path, rows, win))
        return entries

    def _build_windows_from_csv(
        self, windows_csv: Path, coord_type: str
    ) -> List[_WindowEntry]:
        win_df = pd.read_csv(windows_csv)
        tile_by_id = {tid: (img_path, rows) for tid, img_path, rows in self._tiles}
        has_crs_col = "crs" in win_df.columns
        entries: List[_WindowEntry] = []

        for _, win_row in win_df.iterrows():
            tile_id = str(win_row["tile_id"])
            if tile_id not in tile_by_id:
                logger.warning(
                    "windows_csv: tile_id '%s' not found in sources_csv — skipped",
                    tile_id,
                )
                continue
            image_path, rows = tile_by_id[tile_id]

            if coord_type == "image":
                win = rasterio.windows.Window(
                    col_off=int(win_row["col_off"]),
                    row_off=int(win_row["row_off"]),
                    width=int(win_row["patch_w"]),
                    height=int(win_row["patch_h"]),
                )
            elif coord_type == "world":
                win = self._world_bbox_to_window(
                    image_path=image_path,
                    minx=float(win_row["minx"]),
                    miny=float(win_row["miny"]),
                    maxx=float(win_row["maxx"]),
                    maxy=float(win_row["maxy"]),
                    win_crs_str=(
                        str(win_row["crs"])
                        if has_crs_col and pd.notna(win_row["crs"])
                        else None
                    ),
                )
            else:
                raise ValueError(
                    f"Unknown windows_coord_type: '{coord_type}'. "
                    "Must be 'image' or 'world'."
                )
            entries.append((tile_id, image_path, rows, win))
        return entries

    @staticmethod
    def _world_bbox_to_window(
        image_path: str,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        win_crs_str: Optional[str],
    ) -> "rasterio.windows.Window":
        """Convert a geographic bounding box to a pixel Window on ``image_path``.

        When ``win_crs_str`` is provided and differs from the tile's CRS, the
        bounding box is reprojected using ``rasterio.warp.transform_bounds``
        before conversion to pixel coordinates.
        """
        with rasterio.open(image_path) as src:
            tile_transform = src.transform
            tile_crs = src.crs

        if win_crs_str is not None:
            win_crs = rasterio.crs.CRS.from_user_input(win_crs_str)
            if win_crs != tile_crs:
                minx, miny, maxx, maxy = rasterio.warp.transform_bounds(
                    win_crs, tile_crs, minx, miny, maxx, maxy
                )

        raw = rasterio.windows.from_bounds(minx, miny, maxx, maxy, tile_transform)
        return rasterio.windows.Window(
            col_off=round(raw.col_off),
            row_off=round(raw.row_off),
            width=round(raw.width),
            height=round(raw.height),
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self._windows is not None:
            return len(self._windows)
        return len(self._tiles)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._windows is not None:
            return self._getitem_windowed(idx)
        return self._getitem_full(idx)

    # ------------------------------------------------------------------
    # Full-tile item
    # ------------------------------------------------------------------

    def _getitem_full(self, idx: int) -> Dict[str, Any]:
        tile_id, image_path, rows = self._tiles[idx % len(self._tiles)]

        image_np = self._read_image(image_path)
        p_soft_chw = self._ensure_p_soft_cached(tile_id, rows, image_path)

        bags_mask = self._load_bags_mask_full(rows, image_path)
        w_conf_chw = compute_w_conf(
            p_soft_chw,
            alpha=self.alpha,
            use_border=self.use_border,
            entropy_norm=self.entropy_norm,
            bags_mask=bags_mask,
            border_radius=self.border_radius,
        )

        p_soft_hwc = np.ascontiguousarray(p_soft_chw.transpose(1, 2, 0))
        w_conf_hwc = np.ascontiguousarray(w_conf_chw.transpose(1, 2, 0))

        if self.transform is not None:
            result = self.transform(
                image=image_np, p_soft=p_soft_hwc, w_conf=w_conf_hwc
            )
            image_out = result["image"]
            p_soft_out = result["p_soft"]
            w_conf_out = result["w_conf"]
        else:
            image_out, p_soft_out, w_conf_out = image_np, p_soft_hwc, w_conf_hwc

        return {
            "image": self._image_to_tensor(image_out),
            "mask": {
                "mask": self._to_chw_tensor(p_soft_out),
                "w_conf": self._to_chw_tensor(w_conf_out),
            },
            "path": image_path,
        }

    # ------------------------------------------------------------------
    # Windowed item
    # ------------------------------------------------------------------

    def _getitem_windowed(self, idx: int) -> Dict[str, Any]:
        assert self._windows is not None
        tile_id, image_path, rows, win = self._windows[idx % len(self._windows)]

        image_np = self._read_image_window(image_path, win)
        p_soft_chw = self._get_p_soft_window(tile_id, rows, image_path, win)

        bags_mask = self._load_bags_mask_window(rows, image_path, win)
        w_conf_chw = compute_w_conf(
            p_soft_chw,
            alpha=self.alpha,
            use_border=self.use_border,
            entropy_norm=self.entropy_norm,
            bags_mask=bags_mask,
            border_radius=self.border_radius,
        )

        p_soft_hwc = np.ascontiguousarray(p_soft_chw.transpose(1, 2, 0))
        w_conf_hwc = np.ascontiguousarray(w_conf_chw.transpose(1, 2, 0))

        if self.transform is not None:
            result = self.transform(
                image=image_np, p_soft=p_soft_hwc, w_conf=w_conf_hwc
            )
            image_out = result["image"]
            p_soft_out = result["p_soft"]
            w_conf_out = result["w_conf"]
        else:
            image_out, p_soft_out, w_conf_out = image_np, p_soft_hwc, w_conf_hwc

        return {
            "image": self._image_to_tensor(image_out),
            "mask": {
                "mask": self._to_chw_tensor(p_soft_out),
                "w_conf": self._to_chw_tensor(w_conf_out),
            },
            "path": image_path,
        }

    # ------------------------------------------------------------------
    # BAGS mask helpers (w_border_carta, E5)
    # ------------------------------------------------------------------

    def _load_bags_mask_full(
        self,
        rows: pd.DataFrame,
        image_path: str,
    ) -> Optional[np.ndarray]:
        """Return the reprojected BAGS mask (H, W) uint8, or None.

        When the cartographic source is absent from *rows*, returns ``None`` so
        that ``compute_w_conf`` falls back to the argmax-based border.
        """
        if not self.use_border:
            return None
        bags_rows = rows[rows["source_name"] == _CARTOGRAPHIC_SOURCE_NAME]
        if bags_rows.empty:
            logger.warning(
                "cartographic mask source not found in sources; "
                "falling back to argmax-based border",
            )
            return None
        img_h, img_w, img_transform, img_crs, _ = _get_image_grid(image_path)
        return _reproject_lulc_to_image_grid(
            bags_rows.iloc[0]["lulc_path"], img_h, img_w, img_transform, img_crs
        )

    def _load_bags_mask_window(
        self,
        rows: pd.DataFrame,
        image_path: str,
        window: "rasterio.windows.Window",
    ) -> Optional[np.ndarray]:
        """Return the window slice of the BAGS mask (ph, pw) uint8, or None."""
        bags_mask_full = self._load_bags_mask_full(rows, image_path)
        if bags_mask_full is None:
            return None
        ro = int(window.row_off)
        co = int(window.col_off)
        ph = int(window.height)
        pw = int(window.width)
        return bags_mask_full[ro : ro + ph, co : co + pw]

    # ------------------------------------------------------------------
    # Cache logic
    # ------------------------------------------------------------------

    def _ensure_p_soft_cached(
        self,
        tile_id: str,
        rows: pd.DataFrame,
        image_path: str,
    ) -> np.ndarray:
        """Return P_soft (C, H, W) float32, reading from cache or computing it."""
        cache_path = self._cache_dir / f"{tile_id}.tif"

        if cache_path.exists():
            with rasterio.open(cache_path) as src:
                return src.read().astype(np.float32)

        img_h, img_w, img_transform, img_crs, img_profile = _get_image_grid(image_path)

        lulc_maps = []
        for _, row in rows.iterrows():
            lulc = _reproject_lulc_to_image_grid(
                row["lulc_path"], img_h, img_w, img_transform, img_crs
            )
            lulc_maps.append(lulc)

        p_soft = compute_p_soft(lulc_maps, self.num_classes)

        # Atomic write: .tmp → rename (safe for concurrent DataLoader workers)
        tmp_path = self._cache_dir / f"{tile_id}.tif.tmp"
        _write_geotiff(tmp_path, p_soft, img_profile, img_transform, img_crs)
        tmp_path.rename(cache_path)

        logger.debug("P_soft cached for tile %s → %s", tile_id, cache_path)
        return p_soft

    # kept as alias so existing callers are not broken
    _get_or_compute_p_soft = _ensure_p_soft_cached

    def _get_p_soft_window(
        self,
        tile_id: str,
        rows: pd.DataFrame,
        image_path: str,
        window: "rasterio.windows.Window",
    ) -> np.ndarray:
        """Return P_soft for a window (C, H, W) float32.

        Ensures the full-tile cache exists (computing it if needed), then reads
        only the requested window — avoiding loading the full tile on subsequent
        accesses.
        """
        cache_path = self._cache_dir / f"{tile_id}.tif"

        if not cache_path.exists():
            # Compute full tile, cache it, slice the window from memory.
            p_soft_full = self._ensure_p_soft_cached(tile_id, rows, image_path)
            ro = int(window.row_off)
            co = int(window.col_off)
            ph = int(window.height)
            pw = int(window.width)
            return p_soft_full[:, ro : ro + ph, co : co + pw]

        with rasterio.open(cache_path) as src:
            return src.read(window=window).astype(np.float32)

    # ------------------------------------------------------------------
    # Tensor conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_image(path: str) -> np.ndarray:
        """Read a multi-band GeoTIFF and return (H, W, C) uint8."""
        with rasterio.open(path) as src:
            data = src.read()  # (C, H, W)
        return np.ascontiguousarray(data.transpose(1, 2, 0))

    @staticmethod
    def _read_image_window(path: str, window: "rasterio.windows.Window") -> np.ndarray:
        """Read a window from a multi-band GeoTIFF. Returns (H, W, C) uint8."""
        with rasterio.open(path) as src:
            data = src.read(window=window)  # (C, H, W)
        return np.ascontiguousarray(data.transpose(1, 2, 0))

    @staticmethod
    def _image_to_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert image output to (C, H, W) float32 in [0, 1]."""
        if isinstance(x, torch.Tensor):
            t = x.float()
            if t.max() > 1.0 + 1e-3:
                t = t / 255.0
            return t
        arr = np.ascontiguousarray(x.transpose(2, 0, 1))
        t = torch.from_numpy(arr).float()
        if t.max() > 1.0 + 1e-3:
            t = t / 255.0
        return t

    @staticmethod
    def _to_chw_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert (H, W, C) or CHW tensor/array to (C, H, W) float32."""
        if isinstance(x, torch.Tensor):
            if x.ndim == 3:
                return x.permute(2, 0, 1).float().contiguous()
            return x.unsqueeze(0).float()
        if x.ndim == 3:
            return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()
        return torch.from_numpy(x[np.newaxis].copy()).float()
