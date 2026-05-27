# -*- coding: utf-8 -*-
"""Download AlphaEarth Foundation (AEF) embeddings for a set of tiles.

Three download modes are supported:

**GCS mode** (``--source gcs``):
    Downloads per-pixel dense embedding GeoTIFFs from Google Cloud Storage using
    ``gsutil``.  Each tile produces a ``{tile_id}.tif`` file with shape ``(D, H, W)``
    that can be loaded by ``load_aef_embedding`` in ``build_soft_labels.py``.

**HuggingFace mode** (``--source hf``):
    Queries ``Major-TOM/Core-AlphaEarth-Embeddings`` on HuggingFace for the grid
    cell whose centre is geographically closest to each tile, downloads its 64-D
    patch-level embedding, and saves it as ``{tile_id}.npy``.

**Source Cooperative mode** (``--source sourcecoop``):
    Reads the public Source Cooperative STAC GeoParquet index, selects the annual
    COG intersecting each tile footprint, and writes a cropped 64-band GeoTIFF as
    ``{tile_id}.tif``.
"""

import math
import logging
import re
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry import box

try:
    import datasets
except ImportError:  # pragma: no cover
    datasets = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

HF_DATASET_NAME = "Major-TOM/Core-AlphaEarth-Embeddings"
SOURCECOOP_AEF_INDEX_URL = (
    "https://data.source.coop/tge-labs/aef/v1/annual/"
    "aef_index_stac_geoparquet.parquet"
)
SOURCECOOP_S3_PREFIX = "s3://us-west-2.opendata.source.coop/"
SOURCECOOP_PUBLIC_PREFIX = "https://data.source.coop/"


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


def _infer_sourcecoop_year(row: pd.Series, year: Optional[int]) -> int:
    """Infer the annual AEF product year for a tile row.

    Args:
        row: Tile CSV row. A ``year`` column is used when present; otherwise the
            first 4-digit year in ``image_path`` is used.
        year: Explicit year override. When provided, this value takes precedence.

    Returns:
        Annual AEF product year.

    Raises:
        ValueError: If no year can be inferred.
    """
    if year is not None:
        return int(year)

    if "year" in row and not pd.isna(row["year"]):
        return int(row["year"])

    image_path = str(row["image_path"])
    match = re.search(r"(20\d{2})", Path(image_path).name)
    if match:
        return int(match.group(1))

    raise ValueError(
        "Could not infer AEF year from tiles CSV. Provide a 'year' column or "
        "pass --year."
    )


def _href_to_public_sourcecoop_url(href: str) -> str:
    """Convert Source Cooperative S3 hrefs to public HTTPS URLs.

    Args:
        href: STAC asset href, usually an ``s3://us-west-2.opendata.source.coop``
            URI.

    Returns:
        URL/path usable by Rasterio. Local paths and existing HTTP URLs are
        returned unchanged.
    """
    if href.startswith(SOURCECOOP_S3_PREFIX):
        return href.replace(SOURCECOOP_S3_PREFIX, SOURCECOOP_PUBLIC_PREFIX, 1)
    return href


def _load_sourcecoop_index(
    index_path: Optional[str], index_url: str
) -> gpd.GeoDataFrame:
    """Load the Source Cooperative AEF STAC GeoParquet index.

    Args:
        index_path: Optional local GeoParquet index path.
        index_url: Remote GeoParquet URL used when ``index_path`` is absent.

    Returns:
        GeoDataFrame containing STAC items.
    """
    if index_path:
        return gpd.read_parquet(index_path)

    parsed = urlparse(index_url)
    if parsed.scheme in {"http", "https"}:
        request = Request(index_url, headers={"User-Agent": "pytorch-smt-tools"})
        with urlopen(request, timeout=60) as response:
            return gpd.read_parquet(BytesIO(response.read()))

    return gpd.read_parquet(index_url)


def _select_sourcecoop_item(
    index_gdf: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
    year: int,
) -> Optional[pd.Series]:
    """Select the Source Cooperative STAC item intersecting a tile footprint.

    Args:
        index_gdf: Source Cooperative STAC GeoParquet index.
        bbox: Tile bounds as ``(left, bottom, right, top)`` in EPSG:4326 degrees.
        year: Annual AEF product year.

    Returns:
        The intersecting item with the largest overlap area, or ``None`` when no
        item intersects the tile for that year.
    """
    if index_gdf.empty:
        return None

    gdf = index_gdf
    if gdf.crs is not None and str(gdf.crs) not in {"OGC:CRS84", "EPSG:4326"}:
        gdf = gdf.to_crs("OGC:CRS84")

    datetimes = pd.to_datetime(gdf["datetime"])
    candidates = gdf[datetimes.dt.year == int(year)].copy()
    if candidates.empty:
        return None

    tile_geom = box(*bbox)
    candidates = candidates[candidates.intersects(tile_geom)].copy()
    if candidates.empty:
        return None

    overlaps = gpd.GeoSeries(
        candidates.geometry.intersection(tile_geom),
        crs="OGC:CRS84",
    )
    candidates["_overlap_area"] = overlaps.to_crs("EPSG:6933").area.to_numpy()
    return candidates.sort_values("_overlap_area", ascending=False).iloc[0]


def _window_covering_bounds(
    src: rasterio.io.DatasetReader,
    bounds: Tuple[float, float, float, float],
) -> Window:
    """Build a pixel window that fully covers projected bounds.

    Args:
        src: Open embedding COG.
        bounds: ``(left, bottom, right, top)`` in ``src.crs``.

    Returns:
        Rasterio window clipped to the source raster extent.
    """
    left, bottom, right, top = bounds
    xmin, xmax = sorted((left, right))
    ymin, ymax = sorted((bottom, top))
    transform = src.transform
    pixel_width = transform.a
    pixel_height = transform.e

    col0 = math.floor((xmin - transform.c) / pixel_width) - 1
    col1 = math.ceil((xmax - transform.c) / pixel_width) + 1
    if pixel_height > 0:
        row0 = math.floor((ymin - transform.f) / pixel_height) - 1
        row1 = math.ceil((ymax - transform.f) / pixel_height) + 1
    else:
        row0 = math.floor((transform.f - ymax) / abs(pixel_height)) - 1
        row1 = math.ceil((transform.f - ymin) / abs(pixel_height)) + 1

    col0 = max(0, col0)
    row0 = max(0, row0)
    col1 = min(src.width, col1)
    row1 = min(src.height, row1)
    return Window(col0, row0, col1 - col0, row1 - row0)


def _write_sourcecoop_crop(
    href: str,
    image_path: str,
    out_path: Path,
) -> None:
    """Write a cropped Source Cooperative AEF COG for one tile.

    Args:
        href: Local path or public URL to the selected AEF COG.
        image_path: Tile image path whose footprint defines the crop.
        out_path: Output GeoTIFF path.
    """
    with rasterio.open(image_path) as tile:
        tile_bounds = tile.bounds
        tile_crs = tile.crs

    env_options = {}
    parsed = urlparse(href)
    if parsed.scheme in {"http", "https"}:
        env_options.update(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif",
        )

    with rasterio.Env(**env_options):
        with rasterio.open(href) as src:
            projected_bounds = transform_bounds(
                tile_crs,
                src.crs,
                tile_bounds.left,
                tile_bounds.bottom,
                tile_bounds.right,
                tile_bounds.top,
                densify_pts=21,
            )
            window = _window_covering_bounds(src, projected_bounds)
            data = src.read(window=window)
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                height=data.shape[1],
                width=data.shape[2],
                transform=src.window_transform(window),
                compress="DEFLATE",
                tiled=True,
                blockxsize=256,
                blockysize=256,
                BIGTIFF="IF_SAFER",
                interleave="band",
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data)


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


def download_sourcecoop_embeddings(
    tiles_csv: str,
    output_dir: Path,
    index_path: Optional[str] = None,
    index_url: str = SOURCECOOP_AEF_INDEX_URL,
    year: Optional[int] = None,
) -> None:
    """Download cropped per-pixel AEF embeddings from Source Cooperative COGs.

    For each tile, this mode reads the Source Cooperative STAC GeoParquet index,
    selects the annual COG whose footprint intersects the tile, and writes only
    the crop covering that tile. Output files are 64-band GeoTIFFs named
    ``{tile_id}.tif``.

    Example YAML:
        ```yaml
        # Pre-download AEF crops used by SoftLabelModel preprocessing.
        aef_download:
          source: sourcecoop
          tiles_csv: /data/tiles.csv
          output_dir: /data/aef_embeddings
          year: 2025
        ```

    Args:
        tiles_csv: CSV file with ``tile_id`` and ``image_path`` columns. A
            ``year`` column may be provided per tile when ``year`` is not set.
        output_dir: Directory to write cropped GeoTIFF embeddings.
        index_path: Optional local Source Cooperative STAC GeoParquet index.
        index_url: Remote Source Cooperative STAC GeoParquet URL.
        year: Optional annual AEF product year override for all tiles.

    Raises:
        ValueError: If a tile year cannot be inferred or no intersecting COG is
            available for a tile/year pair.
    """
    df = pd.read_csv(tiles_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_gdf = _load_sourcecoop_index(index_path, index_url)

    for _, row in df.iterrows():
        tile_id = str(row["tile_id"])
        out_path = output_dir / f"{tile_id}.tif"
        if out_path.exists():
            logger.info("Skipping %s — already exists", tile_id)
            continue

        tile_year = _infer_sourcecoop_year(row, year)
        bbox = _get_tile_bbox(str(row["image_path"]))
        item = _select_sourcecoop_item(index_gdf, bbox, tile_year)
        if item is None:
            raise ValueError(
                f"No Source Cooperative AEF COG intersects tile '{tile_id}' "
                f"for year {tile_year}."
            )

        href = _href_to_public_sourcecoop_url(str(item["assets"]["data"]["href"]))
        logger.info("Writing Source Cooperative AEF crop for %s from %s", tile_id, href)
        _write_sourcecoop_crop(href, str(row["image_path"]), out_path)
        logger.info("Saved Source Cooperative AEF crop for %s to %s", tile_id, out_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    source: str,
    output_dir: Path,
    gcs_paths_csv: Optional[str] = None,
    tiles_csv: Optional[str] = None,
    sourcecoop_index_path: Optional[str] = None,
    sourcecoop_index_url: str = SOURCECOOP_AEF_INDEX_URL,
    year: Optional[int] = None,
    max_workers: int = 4,
) -> None:
    """Download AEF embeddings for all tiles.

    Args:
        source: ``"gcs"`` for per-pixel GeoTIFFs from GCS, ``"hf"`` for
            patch-level vectors from HuggingFace, or ``"sourcecoop"`` for
            cropped per-pixel GeoTIFFs from Source Cooperative COGs.
        output_dir: Directory where downloaded embedding files will be written.
        gcs_paths_csv: CSV with tile_id and gcs_uri (required for source=gcs).
        tiles_csv: CSV with tile_id and image_path (required for source=hf and
            source=sourcecoop).
        sourcecoop_index_path: Optional local Source Cooperative STAC GeoParquet
            index path.
        sourcecoop_index_url: Remote Source Cooperative STAC GeoParquet URL.
        year: Optional annual AEF product year override for source=sourcecoop.
        max_workers: Reserved for future parallel GCS downloads.

    Raises:
        ValueError: if required CSV argument is missing for the chosen source.
    """
    output_dir = Path(output_dir)
    if source == "gcs":
        if not gcs_paths_csv:
            raise ValueError("gcs_paths_csv is required when source='gcs'")
        download_gcs_embeddings(gcs_paths_csv, output_dir, max_workers)
    elif source == "hf":
        if not tiles_csv:
            raise ValueError("tiles_csv is required when source='hf'")
        download_hf_embeddings(tiles_csv, output_dir)
    elif source == "sourcecoop":
        if not tiles_csv:
            raise ValueError("tiles_csv is required when source='sourcecoop'")
        download_sourcecoop_embeddings(
            tiles_csv,
            output_dir,
            index_path=sourcecoop_index_path,
            index_url=sourcecoop_index_url,
            year=year,
        )
    else:
        raise ValueError("source must be one of: gcs, hf, sourcecoop")
