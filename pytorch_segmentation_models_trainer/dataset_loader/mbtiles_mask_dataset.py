# -*- coding: utf-8 -*-
"""Dataset for MBTiles imagery aligned to GeoTIFF mask windows."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.windows import Window, from_bounds
from torch.utils.data import Dataset

from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    _DTYPE_NORMALIZATION,
    _VALID_IMAGE_DTYPES,
    load_augmentation_object,
)
from pytorch_segmentation_models_trainer.tools.mbtiles.alignment import (
    read_mask_window,
    read_source_aligned_to_mask_window,
)


class MBTilesMaskWindowedDataset(Dataset):
    """Windowed segmentation dataset using masks as the reference grid.

    The dataset enumerates fixed-size windows over one or more GeoTIFF mask
    rasters. For each mask window, source imagery from an MBTiles file, or any
    raster readable by rasterio, is warped into the exact mask CRS, transform,
    resolution, and shape. This makes the mask the ground-truth grid for
    supervised segmentation.

    Args:
        mbtiles_path: Path to MBTiles imagery or rasterio-readable source.
        mask_paths: Explicit mask GeoTIFF paths.
        mask_dir: Directory scanned recursively when ``mask_paths`` is omitted.
        mask_extension: Extension used while scanning ``mask_dir``.
        patch_size: Square patch size in mask pixels.
        stride: Sliding-window stride in mask pixels. Defaults to
            ``patch_size``.
        selected_bands: Optional 1-based source band indexes.
        image_dtype: Source image dtype interpretation. ``uint8`` and
            ``uint16`` are normalized to ``[0, 1]`` when no augmentation is
            used. ``native`` preserves source dtype before tensor conversion.
        image_resampling: Rasterio resampling method for source imagery.
        n_classes: Number of mask classes. ``2`` binarizes masks.
        augmentation_list: Optional Albumentations pipeline.
        data_loader: DataLoader config stored for the framework ``Model``.
        return_metadata: Include ``mask_path``, ``row_off``, and ``col_off`` in
            returned samples.
        window_index_cache: Optional CSV or Parquet file used to persist the
            computed mask-window index. When the file exists, the dataset loads
            windows from it and skips mask-directory scanning. The cache must
            contain a mask path column and either pixel windows
            (``row_off``, ``col_off`` plus ``width``/``height`` or
            ``patch_size``) or world-coordinate bounds
            (``minx``, ``miny``, ``maxx``, ``maxy``) in the mask CRS.
        window_index_mask_path_key: Column containing mask paths in
            ``window_index_cache``.
        window_index_coordinate_mode: ``auto``, ``pixel``, or ``bounds``.
        mask_base_path: Optional base directory prepended to relative mask
            paths read from ``window_index_cache``. Has no effect when paths
            are already absolute. Useful for storing only filenames in the CSV
            and resolving them at runtime via the YAML config.
        **kwargs: Compatibility kwargs for Hydra instantiation.

    Returns:
        Each item is a dict with ``image`` tensor ``(C,H,W)`` and ``mask``
        tensor ``(H,W)``. Without augmentations, image is ``float32`` and mask
        is ``int64``.

    Example YAML:
        ```yaml
        train_dataset:
          _target_: pytorch_segmentation_models_trainer.dataset_loader...
          mbtiles_path: /data/source.mbtiles
          mask_dir: /data/masks
          patch_size: 512
          stride: 512
          selected_bands: [1, 2, 3]
          image_resampling: bilinear
        ```
    """

    def __init__(
        self,
        mbtiles_path,
        mask_paths: Optional[Sequence[str]] = None,
        mask_dir: Optional[str] = None,
        mask_extension: str = ".tif",
        patch_size: int = 512,
        stride: Optional[int] = None,
        selected_bands: Optional[Sequence[int]] = None,
        image_dtype: str = "uint8",
        image_resampling: str = "bilinear",
        n_classes: int = 2,
        augmentation_list=None,
        data_loader=None,
        return_metadata: bool = True,
        window_index_cache: Optional[str] = None,
        window_index_mask_path_key: str = "mask_path",
        window_index_coordinate_mode: str = "auto",
        mask_base_path: Optional[str] = None,
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

        self.mbtiles_path = Path(mbtiles_path)
        self.patch_size = int(patch_size)
        self.stride = int(stride or patch_size)
        if self.stride <= 0:
            raise ValueError("stride must be a positive integer.")
        self.selected_bands = list(selected_bands) if selected_bands else None
        self.image_dtype = image_dtype
        self.image_resampling = image_resampling
        self.n_classes = n_classes
        self.data_loader = data_loader
        self.return_metadata = return_metadata
        self.window_index_mask_path_key = window_index_mask_path_key
        self.window_index_coordinate_mode = window_index_coordinate_mode
        self.mask_base_path = Path(mask_base_path) if mask_base_path else None
        if self.window_index_coordinate_mode not in {"auto", "pixel", "bounds"}:
            raise ValueError(
                "window_index_coordinate_mode must be 'auto', 'pixel', or 'bounds'."
            )
        self.window_index_cache = None
        if window_index_cache is not None:
            self.window_index_cache = Path(window_index_cache)
        if self.window_index_cache is not None:
            self._validate_window_index_cache_path(self.window_index_cache)
        self.transform = (
            None
            if augmentation_list is None
            else load_augmentation_object(augmentation_list)
        )
        cache_exists = self.window_index_cache is not None
        cache_exists = cache_exists and self.window_index_cache.exists()
        if cache_exists:
            self.windows = self._read_window_index_cache(
                self.window_index_cache,
                mask_path_key=self.window_index_mask_path_key,
                coordinate_mode=self.window_index_coordinate_mode,
                mask_base_path=self.mask_base_path,
            )
            self.mask_paths = sorted({record["mask_path"] for record in self.windows})
            p = self.window_index_cache
            self.df = (
                pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_parquet(p)
            )
        else:
            self.mask_paths = self._collect_mask_paths(
                mask_paths=mask_paths,
                mask_dir=mask_dir,
                mask_extension=mask_extension,
            )
            self.windows = self._build_window_index(self.mask_paths)
            self.df = None
            if self.window_index_cache is not None:
                self._write_window_index_cache(self.window_index_cache, self.windows)

    def __len__(self) -> int:
        """Return number of valid mask windows."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one mask-aligned image/mask training sample.

        Args:
            idx: Dataset item index.

        Returns:
            Dict with image/mask tensors and optional metadata.

        Raises:
            IndexError: If *idx* is outside the dataset range.
        """
        if idx < 0 or idx >= len(self):
            msg = f"Index {idx} out of bounds for size {len(self)}."
            raise IndexError(msg)
        record = self.windows[idx]
        mask_path = record["mask_path"]
        window = Window(
            record["col_off"],
            record["row_off"],
            record["width"],
            record["height"],
        )

        with rasterio.open(mask_path) as mask_src:
            image = read_source_aligned_to_mask_window(
                source_path=self.mbtiles_path,
                mask_src=mask_src,
                window=window,
                selected_bands=self.selected_bands,
                image_dtype=self.image_dtype,
                image_resampling=self.image_resampling,
            )
            mask = read_mask_window(mask_src, window, n_classes=self.n_classes)

        if self.transform is not None:
            transformed = self.transform(
                image=np.transpose(image, (1, 2, 0)),
                mask=mask,
            )
            sample = {
                "image": transformed["image"],
                "mask": transformed["mask"],
            }
        else:
            tensor_img = torch.from_numpy(image).float()
            norm_factor = _DTYPE_NORMALIZATION.get(self.image_dtype)
            if norm_factor is not None:
                tensor_img = tensor_img / norm_factor
            sample = {
                "image": tensor_img,
                "mask": torch.from_numpy(mask).long(),
            }

        if self.return_metadata:
            sample.update(
                {
                    "mask_path": str(mask_path),
                    "row_off": int(record["row_off"]),
                    "col_off": int(record["col_off"]),
                }
            )
        return sample

    @staticmethod
    def _collect_mask_paths(
        mask_paths: Optional[Iterable[str]],
        mask_dir: Optional[str],
        mask_extension: str,
    ) -> List[Path]:
        if mask_paths is not None:
            paths = sorted(Path(p) for p in mask_paths)
        elif mask_dir is not None:
            ext = (
                mask_extension
                if mask_extension.startswith(".")
                else f".{mask_extension}"
            )
            root = Path(mask_dir)
            paths = sorted(root.rglob(f"*{ext}")) if root.exists() else []
        else:
            raise ValueError("Either mask_paths or mask_dir must be provided.")

        if not paths:
            raise ValueError("No mask files found for MBTiles mask dataset.")
        return paths

    def _build_window_index(self, mask_paths: Sequence[Path]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for mask_path in mask_paths:
            with rasterio.open(mask_path) as mask_src:
                width = mask_src.width
                height = mask_src.height
            if width < self.patch_size or height < self.patch_size:
                continue
            for row_off in range(0, height - self.patch_size + 1, self.stride):
                for col_off in range(0, width - self.patch_size + 1, self.stride):
                    records.append(
                        {
                            "mask_path": mask_path,
                            "row_off": row_off,
                            "col_off": col_off,
                            "width": self.patch_size,
                            "height": self.patch_size,
                        }
                    )
        if not records:
            raise ValueError(
                "No valid mask windows found. Check patch_size, stride, and "
                "mask dimensions."
            )
        return records

    @staticmethod
    def _validate_window_index_cache_path(path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".parquet"}:
            raise ValueError("window_index_cache must end with '.csv' or '.parquet'.")

    @staticmethod
    def _read_window_index_cache(
        path: Path,
        mask_path_key: str = "mask_path",
        coordinate_mode: str = "auto",
        mask_base_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
        if mask_path_key not in df.columns:
            raise ValueError(
                "window_index_cache missing mask path column: " f"'{mask_path_key}'."
            )
        if coordinate_mode not in {"auto", "pixel", "bounds"}:
            raise ValueError("coordinate_mode must be 'auto', 'pixel', or 'bounds'.")

        has_pixel_offsets = {"row_off", "col_off"}.issubset(df.columns)
        has_width_height = {"width", "height"}.issubset(df.columns)
        has_patch_size = "patch_size" in df.columns
        has_pixel_size = has_width_height or has_patch_size
        has_pixel_windows = has_pixel_offsets and has_pixel_size
        bounds_columns = MBTilesMaskWindowedDataset._resolve_bounds_columns(df.columns)
        has_bounds = bounds_columns is not None
        if coordinate_mode == "auto":
            resolved_mode = "pixel" if has_pixel_windows else "bounds"
        else:
            resolved_mode = coordinate_mode

        if resolved_mode == "pixel" and not has_pixel_windows:
            missing = {"row_off", "col_off"} - set(df.columns)
            if not has_pixel_size:
                missing.add("width/height or patch_size")
            raise ValueError(
                "window_index_cache missing pixel-window columns: " f"{sorted(missing)}"
            )
        if resolved_mode == "bounds" and not has_bounds:
            raise ValueError(
                "window_index_cache with bounds mode must include "
                "['minx', 'miny', 'maxx', 'maxy']."
            )
        records = []
        for row in df.to_dict(orient="records"):
            mask_path = Path(row[mask_path_key])
            if mask_base_path is not None and not mask_path.is_absolute():
                mask_path = mask_base_path / mask_path
            if resolved_mode == "pixel":
                width = (
                    int(row["width"]) if has_width_height else int(row["patch_size"])
                )
                height = (
                    int(row["height"]) if has_width_height else int(row["patch_size"])
                )
                row_off = int(row["row_off"])
                col_off = int(row["col_off"])
            else:
                assert bounds_columns is not None
                bounds_record = tuple(
                    float(row[bounds_columns[name]])
                    for name in ["minx", "miny", "maxx", "maxy"]
                )
                row_off, col_off, width, height = (
                    MBTilesMaskWindowedDataset._bounds_to_pixel_window(
                        mask_path,
                        bounds_record,
                        row.get("crs"),
                    )
                )
            records.append(
                {
                    "mask_path": mask_path,
                    "row_off": row_off,
                    "col_off": col_off,
                    "width": width,
                    "height": height,
                }
            )
        if not records:
            raise ValueError("window_index_cache contains no mask windows.")
        return records

    @staticmethod
    def _resolve_bounds_columns(columns) -> Optional[Dict[str, str]]:
        names = set(columns)
        for prefix in ("", "tile_"):
            candidate = {
                "minx": f"{prefix}minx",
                "miny": f"{prefix}miny",
                "maxx": f"{prefix}maxx",
                "maxy": f"{prefix}maxy",
            }
            if set(candidate.values()).issubset(names):
                return candidate
        return None

    @staticmethod
    def _bounds_to_pixel_window(
        mask_path: Path,
        bounds_record: Sequence[float],
        crs_value: Any = None,
    ) -> Sequence[int]:
        minx, miny, maxx, maxy = bounds_record
        left = min(minx, maxx)
        right = max(minx, maxx)
        bottom = min(miny, maxy)
        top = max(miny, maxy)
        with rasterio.open(mask_path) as mask_src:
            if crs_value is not None and not pd.isna(crs_value):
                cache_crs = CRS.from_user_input(crs_value)
                if mask_src.crs is not None and cache_crs != mask_src.crs:
                    raise ValueError(
                        "window_index_cache bounds CRS does not match mask CRS "
                        f"for {mask_path}: {cache_crs} != {mask_src.crs}."
                    )
            window = from_bounds(
                left=left,
                bottom=bottom,
                right=right,
                top=top,
                transform=mask_src.transform,
            )
            col_off = int(round(window.col_off))
            row_off = int(round(window.row_off))
            width = int(round(window.width))
            height = int(round(window.height))
            if width <= 0 or height <= 0:
                raise ValueError(
                    "window_index_cache bounds produced an empty window "
                    f"for {mask_path}."
                )
            if (
                col_off < 0
                or row_off < 0
                or col_off + width > mask_src.width
                or row_off + height > mask_src.height
            ):
                raise ValueError(
                    "window_index_cache bounds fall outside the mask raster "
                    f"for {mask_path}. Check that bounds are in the mask CRS."
                )
        return row_off, col_off, width, height

    @staticmethod
    def _write_window_index_cache(path: Path, windows: Sequence[Dict[str, Any]]):
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "mask_path": str(record["mask_path"]),
                "row_off": int(record["row_off"]),
                "col_off": int(record["col_off"]),
                "width": int(record["width"]),
                "height": int(record["height"]),
            }
            for record in windows
        ]
        df = pd.DataFrame(rows)
        if path.suffix.lower() == ".csv":
            df.to_csv(path, index=False)
        else:
            df.to_parquet(path, index=False)
