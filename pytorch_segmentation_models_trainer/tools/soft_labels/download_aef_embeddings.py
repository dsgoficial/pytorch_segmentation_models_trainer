# -*- coding: utf-8 -*-
"""Download AlphaEarth Foundation (AEF) embeddings for a set of tiles.

Two download modes are supported:

**GCS mode** (``--source gcs``):
    Downloads per-pixel dense embedding GeoTIFFs from Google Cloud Storage using
    ``gsutil``.  Each tile produces a ``{tile_id}.tif`` file with shape ``(D, H, W)``
    that can be loaded by ``load_aef_embedding`` in ``build_soft_labels.py``.

**HuggingFace mode** (``--source hf``):
    Queries ``Major-TOM/Core-AlphaEarth-Embeddings`` on HuggingFace for the grid
    cell whose centre is geographically closest to each tile, downloads its 64-D
    patch-level embedding, and saves it as ``{tile_id}.npy``.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS

try:
    import datasets
except ImportError:  # pragma: no cover
    datasets = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

HF_DATASET_NAME = "Major-TOM/Core-AlphaEarth-Embeddings"


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------


def _get_tile_bbox(image_path: str) -> Tuple[float, float, float, float]:
    """Return (left, bottom, right, top) bounding box of a raster in WGS84.

    Args:
        image_path: Path to a GeoTIFF image.

    Returns:
        ``(left, bottom, right, top)`` in EPSG:4326 degrees.
    """
    with rasterio.open(image_path) as src:
        bounds = src.bounds
        if src.crs and not src.crs.is_geographic:
            from rasterio.warp import transform_bounds

            left, bottom, right, top = transform_bounds(
                src.crs,
                CRS.from_epsg(4326),
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
            )
        else:
            left = float(bounds.left)
            bottom = float(bounds.bottom)
            right = float(bounds.right)
            top = float(bounds.top)
    return left, bottom, right, top


def _find_hf_cell_for_bbox(
    bbox: Tuple[float, float, float, float],
    cells_df: pd.DataFrame,
) -> Optional[int]:
    """Find the HuggingFace Major-TOM grid cell index closest to the tile centre.

    Args:
        bbox: ``(left, bottom, right, top)`` in WGS84 degrees.
        cells_df: DataFrame with ``centre_lat`` and ``centre_lon`` columns.

    Returns:
        Row index (usable with ``.loc[]``) of the nearest cell, or ``None`` if
        ``cells_df`` is empty.
    """
    if cells_df.empty:
        return None

    tile_centre_lon = (bbox[0] + bbox[2]) / 2.0
    tile_centre_lat = (bbox[1] + bbox[3]) / 2.0

    dist_sq = (cells_df["centre_lat"] - tile_centre_lat) ** 2 + (
        cells_df["centre_lon"] - tile_centre_lon
    ) ** 2
    return int(dist_sq.idxmin())


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------


def download_gcs_embeddings(
    gcs_paths_csv: str,
    output_dir: Path,
    max_workers: int = 4,
) -> None:
    """Download per-pixel GeoTIFF embeddings from GCS using ``gsutil``.

    Skips tiles whose output file already exists.  Requires ``gsutil`` to be
    installed and authenticated.

    Args:
        gcs_paths_csv: CSV file with columns ``tile_id`` and ``gcs_uri``.
        output_dir: Directory to write downloaded GeoTIFFs (``{tile_id}.tif``).
        max_workers: Reserved for future parallel implementation; currently unused
                     (``gsutil`` handles its own parallelism).
    """
    df = pd.read_csv(gcs_paths_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        tile_id = str(row["tile_id"])
        gcs_uri = str(row["gcs_uri"])
        out_path = output_dir / f"{tile_id}.tif"

        if out_path.exists():
            logger.info("Skipping %s — already downloaded", tile_id)
            continue

        logger.info("Downloading %s → %s", gcs_uri, out_path)
        result = subprocess.run(
            ["gsutil", "cp", gcs_uri, str(out_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("Failed to download %s: %s", gcs_uri, result.stderr)
        else:
            logger.info("Done: %s", tile_id)


def download_hf_embeddings(
    tiles_csv: str,
    output_dir: Path,
) -> None:
    """Download patch-level embeddings from HuggingFace Major-TOM/Core-AlphaEarth-Embeddings.

    For each tile, finds the nearest HF grid cell by geographic proximity, extracts
    its embedding vector, and saves it as a ``{tile_id}.npy`` float32 file.

    Requires the ``datasets`` package (``pip install datasets``).

    Args:
        tiles_csv: CSV file with columns ``tile_id`` and ``image_path``.
        output_dir: Directory to write ``.npy`` embedding files.
    """
    if datasets is None:  # pragma: no cover
        raise ImportError(
            "The 'datasets' package is required for HuggingFace downloads.  "
            "Install it with: pip install datasets"
        )

    df = pd.read_csv(tiles_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading HuggingFace dataset %s…", HF_DATASET_NAME)
    ds = datasets.load_dataset(HF_DATASET_NAME, split="train")
    cells_df = ds.to_pandas()[["centre_lat", "centre_lon", "embeddings"]]

    for _, row in df.iterrows():
        tile_id = str(row["tile_id"])
        out_path = output_dir / f"{tile_id}.npy"

        if out_path.exists():
            logger.info("Skipping %s — already exists", tile_id)
            continue

        bbox = _get_tile_bbox(str(row["image_path"]))
        cell_idx = _find_hf_cell_for_bbox(bbox, cells_df[["centre_lat", "centre_lon"]])

        if cell_idx is None:
            logger.warning("No HF cell found for tile %s — skipping", tile_id)
            continue

        embedding = np.array(cells_df.loc[cell_idx, "embeddings"], dtype=np.float32)
        np.save(out_path, embedding)
        logger.info("Saved HF embedding for %s (shape=%s)", tile_id, embedding.shape)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    source: str,
    output_dir: Path,
    gcs_paths_csv: Optional[str] = None,
    tiles_csv: Optional[str] = None,
    max_workers: int = 4,
) -> None:
    """Download AEF embeddings for all tiles.

    Args:
        source: ``"gcs"`` for per-pixel GeoTIFFs from GCS, ``"hf"`` for
                patch-level vectors from HuggingFace.
        output_dir: Directory where downloaded embedding files will be written.
        gcs_paths_csv: CSV with tile_id and gcs_uri (required for source=gcs).
        tiles_csv: CSV with tile_id and image_path (required for source=hf).
        max_workers: Reserved for future parallel GCS downloads.

    Raises:
        ValueError: if required CSV argument is missing for the chosen source.
    """
    output_dir = Path(output_dir)
    if source == "gcs":
        if not gcs_paths_csv:
            raise ValueError("gcs_paths_csv is required when source='gcs'")
        download_gcs_embeddings(gcs_paths_csv, output_dir, max_workers)
    else:
        if not tiles_csv:
            raise ValueError("tiles_csv is required when source='hf'")
        download_hf_embeddings(tiles_csv, output_dir)
