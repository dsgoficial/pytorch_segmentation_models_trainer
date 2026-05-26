# -*- coding: utf-8 -*-
"""Export MBTiles/source imagery aligned to GeoTIFF mask grids for QA."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window

from pytorch_segmentation_models_trainer.tools.mbtiles.alignment import (
    read_mask_window,
    read_source_aligned_to_mask_window,
)


@dataclass
class MBTilesMaskAlignedExportResult:
    """Summary of exported mask-aligned imagery.

    Args:
        manifest_path: CSV manifest written by the exporter.
        image_paths: Exported image GeoTIFF paths.
        mask_paths: Exported mask GeoTIFF paths.
        preview_paths: Exported PNG preview paths.
    """

    manifest_path: Path
    image_paths: List[Path]
    mask_paths: List[Path]
    preview_paths: List[Path]

    @property
    def count(self) -> int:
        """Return number of exported image/mask pairs."""
        return len(self.image_paths)


def _as_paths(paths: Optional[Iterable[Path]]) -> Optional[List[Path]]:
    if paths is None:
        return None
    return [Path(p) for p in paths]


def _collect_mask_paths(
    mask_paths: Optional[Iterable[Path]],
    mask_dir: Optional[Path],
    mask_extension: str,
) -> List[Path]:
    if mask_paths is not None:
        paths = sorted(_as_paths(mask_paths))
    elif mask_dir is not None:
        ext = mask_extension if mask_extension.startswith(".") else f".{mask_extension}"
        paths = sorted(Path(mask_dir).rglob(f"*{ext}"))
    else:
        raise ValueError("Either mask_paths or mask_dir must be provided.")

    if not paths:
        raise ValueError("No mask files found for MBTiles aligned export.")
    return paths


def _iter_windows(width: int, height: int, patch_size: int, stride: int):
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive integers.")
    if width < patch_size or height < patch_size:
        return
    for row_off in range(0, height - patch_size + 1, stride):
        for col_off in range(0, width - patch_size + 1, stride):
            yield Window(col_off, row_off, patch_size, patch_size)


def _profile_for_image(
    mask_src: rasterio.io.DatasetReader,
    window: Window,
    image: np.ndarray,
    output_format: str,
) -> dict:
    profile = mask_src.profile.copy()
    profile.update(
        {
            "driver": ("GTiff" if output_format.lower() in {"tif", "tiff"} else "PNG"),
            "height": int(window.height),
            "width": int(window.width),
            "count": int(image.shape[0]),
            "dtype": str(image.dtype),
            "transform": mask_src.window_transform(window),
            "crs": mask_src.crs,
            "nodata": None,
        }
    )
    return profile


def _profile_for_mask(
    mask_src: rasterio.io.DatasetReader,
    window: Window,
    mask: np.ndarray,
    output_format: str,
) -> dict:
    profile = mask_src.profile.copy()
    profile.update(
        {
            "driver": ("GTiff" if output_format.lower() in {"tif", "tiff"} else "PNG"),
            "height": int(window.height),
            "width": int(window.width),
            "count": 1,
            "dtype": str(mask.dtype),
            "transform": mask_src.window_transform(window),
            "crs": mask_src.crs,
            "nodata": None,
        }
    )
    return profile


def _to_preview_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError("Expected image array with shape (C, H, W).")
    channels = image[:3]
    if channels.shape[0] == 1:
        channels = np.repeat(channels, 3, axis=0)
    if channels.shape[0] == 2:
        channels = np.concatenate([channels, channels[:1]], axis=0)
    rgb = np.transpose(channels, (1, 2, 0))
    if rgb.dtype != np.uint8:
        max_value = float(np.nanmax(rgb)) if rgb.size else 0.0
        if max_value <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _write_preview(
    image: np.ndarray,
    mask: np.ndarray,
    preview_path: Path,
    overlay_alpha: float,
) -> None:
    rgb = _to_preview_rgb(image).astype(np.float32)
    overlay = np.zeros_like(rgb)
    overlay[..., 0] = 255
    mask_bool = mask > 0
    rgb[mask_bool] = (1.0 - overlay_alpha) * rgb[mask_bool] + overlay_alpha * overlay[
        mask_bool
    ]
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(preview_path)


def _safe_stem(mask_path: Path, window: Window, full_mask: bool) -> str:
    if full_mask:
        return mask_path.stem
    row = int(window.row_off)
    col = int(window.col_off)
    return f"{mask_path.stem}__r{row:06d}_c{col:06d}"


def _empty_manifest(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(
        columns=[
            "image_path",
            "mask_path",
            "preview_path",
            "source_mask_path",
            "row_off",
            "col_off",
            "width",
            "height",
            "crs",
            "transform",
        ]
    )
    df.to_csv(path, index=False)
    return df


def export_mask_aligned_images(
    mbtiles_path: Path,
    output_dir: Path,
    mask_paths: Optional[Sequence[Path]] = None,
    mask_dir: Optional[Path] = None,
    mask_extension: str = ".tif",
    patch_size: Optional[int] = None,
    stride: Optional[int] = None,
    full_mask: bool = False,
    selected_bands: Optional[Sequence[int]] = None,
    image_dtype: str = "uint8",
    output_format: str = "tif",
    image_resampling: str = "bilinear",
    write_sidecar_png: bool = True,
    overlay_alpha: float = 0.35,
    skip_empty_masks: bool = False,
    n_classes: int = 2,
    **kwargs,
) -> MBTilesMaskAlignedExportResult:
    """Export MBTiles imagery reprojected to GeoTIFF mask windows.

    Args:
        mbtiles_path: Path to MBTiles imagery, or any raster readable by
            rasterio. The imagery is treated as source data.
        output_dir: Directory receiving ``images/``, ``masks/``,
            ``previews/``, and ``manifest.csv``.
        mask_paths: Explicit mask GeoTIFF paths.
        mask_dir: Directory scanned recursively for masks when *mask_paths* is
            omitted.
        mask_extension: Mask extension used with *mask_dir*.
        patch_size: Patch size in mask pixels for windowed export.
        stride: Patch stride in mask pixels. Defaults to *patch_size*.
        full_mask: When ``True``, export one image for each full mask raster.
        selected_bands: Optional 1-based bands to read from the source raster.
        image_dtype: Output imagery dtype, or ``"native"``.
        output_format: Raster extension/driver. ``"tif"`` is recommended.
        image_resampling: Resampling method used for imagery.
        write_sidecar_png: Write RGB overlay PNG previews when ``True``.
        overlay_alpha: Mask overlay alpha in PNG previews.
        skip_empty_masks: Skip windows whose mask has no non-zero pixels.
        n_classes: Number of mask classes. ``2`` binarizes mask windows.
        **kwargs: Ignored compatibility kwargs for Hydra configs.

    Returns:
        :class:`MBTilesMaskAlignedExportResult` with output paths.

    Example YAML:
        ```yaml
        mbtiles_export:
          _target_: pytorch_segmentation_models_trainer.tools.mbtiles...
          mbtiles_path: /data/source.mbtiles
          mask_dir: /data/masks
          output_dir: /data/qa_tiles
          patch_size: 512
          stride: 512
          write_sidecar_png: true
        ```

    Raises:
        ValueError: If neither full-mask nor patch export mode is configured.
    """
    del kwargs
    mbtiles_path = Path(mbtiles_path)
    output_dir = Path(output_dir)

    if not full_mask and patch_size is None:
        raise ValueError("Set full_mask=True or patch_size for patch export.")
    if stride is None:
        stride = patch_size

    masks = _collect_mask_paths(mask_paths, mask_dir, mask_extension)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    previews_dir = output_dir / "previews"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if write_sidecar_png:
        previews_dir.mkdir(parents=True, exist_ok=True)

    image_paths: List[Path] = []
    exported_mask_paths: List[Path] = []
    preview_paths: List[Path] = []
    rows = []
    ext = output_format.lower().lstrip(".")

    for mask_path in masks:
        with rasterio.open(mask_path) as mask_src:
            if full_mask:
                windows = [Window(0, 0, mask_src.width, mask_src.height)]
            else:
                windows = list(
                    _iter_windows(
                        mask_src.width,
                        mask_src.height,
                        patch_size,
                        stride,
                    )
                )

            for window in windows:
                mask = read_mask_window(mask_src, window, n_classes=n_classes)
                if skip_empty_masks and not np.any(mask):
                    continue

                image = read_source_aligned_to_mask_window(
                    source_path=mbtiles_path,
                    mask_src=mask_src,
                    window=window,
                    selected_bands=selected_bands,
                    image_dtype=image_dtype,
                    image_resampling=image_resampling,
                )

                stem = _safe_stem(Path(mask_path), window, full_mask)
                image_path = images_dir / f"{stem}.{ext}"
                exported_mask_path = masks_dir / f"{stem}.{ext}"
                preview_path = previews_dir / f"{stem}.png"

                image_profile = _profile_for_image(
                    mask_src, window, image, output_format
                )
                with rasterio.open(image_path, "w", **image_profile) as dst:
                    dst.write(image)

                with rasterio.open(
                    exported_mask_path,
                    "w",
                    **_profile_for_mask(mask_src, window, mask, output_format),
                ) as dst:
                    dst.write(mask, 1)

                if write_sidecar_png:
                    _write_preview(image, mask, preview_path, overlay_alpha)
                    preview_value = str(preview_path)
                    preview_paths.append(preview_path)
                else:
                    preview_value = ""

                image_paths.append(image_path)
                exported_mask_paths.append(exported_mask_path)
                rows.append(
                    {
                        "image_path": str(image_path),
                        "mask_path": str(exported_mask_path),
                        "preview_path": preview_value,
                        "source_mask_path": str(mask_path),
                        "row_off": int(window.row_off),
                        "col_off": int(window.col_off),
                        "width": int(window.width),
                        "height": int(window.height),
                        "crs": str(mask_src.crs),
                        "transform": repr(tuple(mask_src.window_transform(window))),
                    }
                )

    manifest_path = output_dir / "manifest.csv"
    if rows:
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
    else:
        _empty_manifest(manifest_path)

    return MBTilesMaskAlignedExportResult(
        manifest_path=manifest_path,
        image_paths=image_paths,
        mask_paths=exported_mask_paths,
        preview_paths=preview_paths,
    )
