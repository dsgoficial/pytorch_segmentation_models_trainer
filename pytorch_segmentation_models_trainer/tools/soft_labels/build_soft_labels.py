# -*- coding: utf-8 -*-
"""Build P_soft and W_conf rasters from multiple LULC source products.

Pipeline for the E3-E5 ablation:

1. Load the BAGS cartographic mask and external LULC rasters for each tile.
2. Reproject every source to the image grid.
3. Compute equal-vote P_soft(i, c) over the valid source labels.
4. Compute entropy-based uncertainty:
     H(i) = -Σ_c P_soft(i,c) · log(P_soft(i,c))
     w_entropy(i) = 1 - H(i) / log(C)
5. Compute border distance weight:
     if the BAGS cartographic mask is available:
         w_border_carta(i) = min(1, d_carta(i) / R)
     otherwise:
         w_border(i) = distance_transform_edt(~argmax_border)(i) / max_dist
6. Combine: W_conf = alpha * w_entropy + (1 - alpha) * w_border
7. Write P_soft and W_conf as GeoTIFF rasters aligned to the image grid.

The image raster is the spatial reference: all LULC sources are reprojected to
its CRS, transform, and pixel dimensions before computing P_soft. This guarantees
that P_soft and W_conf are pixel-aligned with the training images regardless of
the original resolution of each LULC product.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject as warp_reproject
from scipy.ndimage import distance_transform_edt

from pytorch_segmentation_models_trainer.tools.soft_labels.aef_utils import (
    AEFResamplingStrategy,
    aggregate_aef_raster_to_grid,
    normalize_aef_vectors,
    read_aef_raster,
    resample_aef_raster_to_grid,
    valid_aef_vector_mask,
)

logger = logging.getLogger(__name__)

_CARTOGRAPHIC_SOURCE_NAME = "bags"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _get_image_grid(image_path: str):
    """Return (height, width, transform, crs, profile) of the reference image."""
    with rasterio.open(image_path) as src:
        return src.height, src.width, src.transform, src.crs, src.profile


def _tile_id_from_image_path(image_path: str, row_index: int) -> str:
    """Return a stable tile id when a wide CSV does not provide one."""
    stem = Path(str(image_path)).stem
    return stem if stem else f"tile_{row_index}"


def _normalize_sources_dataframe(
    df: pd.DataFrame,
    image_key: str = "image_path",
    mask_key: str = "mask_path",
    lulc_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Normalize the wide source CSV to internal equal-vote source rows.

    The CSV has one row per tile, for example:
    ``tile_id,image_path,mask_path,mapbiomas_path,esri_path,dw_path``.
    ``tile_id`` is optional; when absent it is derived from ``image_key``.
    LULC source columns are listed explicitly via ``lulc_keys``.

    Args:
        df: Raw CSV dataframe.
        image_key: CSV column with image paths.
        mask_key: CSV column with the cartographic mask path.
        lulc_keys: CSV columns with external LULC source paths.

    Returns:
        Dataframe with ``tile_id, image_path, source_name, lulc_path``.

    Raises:
        ValueError: If the CSV does not match the supported wide format.
    """
    if image_key not in df.columns:
        raise ValueError(f"Soft-label source CSV must contain image_key '{image_key}'.")
    if mask_key not in df.columns:
        raise ValueError(
            f"Soft-label source CSV must contain mask_key '{mask_key}' "
            "for the BAGS cartographic mask."
        )

    source_columns = [(_CARTOGRAPHIC_SOURCE_NAME, mask_key)]
    for key in list(lulc_keys) if lulc_keys else []:
        if key not in df.columns:
            raise ValueError(
                f"Soft-label source CSV is missing LULC key '{key}'. "
                f"Available columns: {list(df.columns)}"
            )
        source_columns.append((key, key))

    rows = []
    for row_index, row in df.iterrows():
        image_path = row[image_key]
        tile_id = (
            str(row["tile_id"])
            if "tile_id" in df.columns and pd.notna(row["tile_id"])
            else _tile_id_from_image_path(image_path, row_index)
        )
        for source_name, column in source_columns:
            lulc_path = row[column]
            if pd.isna(lulc_path) or str(lulc_path).strip() == "":
                continue
            rows.append(
                {
                    "tile_id": tile_id,
                    "image_path": image_path,
                    "source_name": source_name,
                    "lulc_path": lulc_path,
                }
            )

    if not rows:
        raise ValueError("Soft-label source CSV contains no valid LULC source paths.")
    return pd.DataFrame(rows)


def _detect_boundary(mask: np.ndarray) -> np.ndarray:
    """Return a bool border mask using an 8-connected 3x3 neighborhood.

    A pixel is on the boundary if any of its 8 neighbors has a different integer
    class value. This matches the "3x3 morphological filter" described in LaTeX
    Eq. 13 for the w_border_carta computation (Experiment E5).

    Args:
        mask: (H, W) integer array of class labels.

    Returns:
        (H, W) bool array; True where the pixel is on a class boundary.
    """
    border = np.zeros(mask.shape, dtype=bool)
    border[:-1, :] |= mask[:-1, :] != mask[1:, :]
    border[1:, :] |= mask[1:, :] != mask[:-1, :]
    border[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    border[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    border[:-1, :-1] |= mask[:-1, :-1] != mask[1:, 1:]
    border[1:, 1:] |= mask[1:, 1:] != mask[:-1, :-1]
    border[:-1, 1:] |= mask[:-1, 1:] != mask[1:, :-1]
    border[1:, :-1] |= mask[1:, :-1] != mask[:-1, 1:]
    return border


def _reproject_lulc_to_image_grid(
    lulc_path: str,
    dst_height: int,
    dst_width: int,
    dst_transform: Affine,
    dst_crs,
) -> np.ndarray:
    """Read a LULC raster and reproject it to the image pixel grid.

    Uses nearest-neighbour resampling to preserve integer class labels.
    Works regardless of whether the LULC resolution matches the image.

    Args:
        lulc_path: Path to the single-band integer LULC GeoTIFF.
        dst_height: Target height in pixels (from the image raster).
        dst_width: Target width in pixels (from the image raster).
        dst_transform: Affine transform of the target grid.
        dst_crs: CRS of the target grid.

    Returns:
        (H, W) uint8 array aligned to the image grid.
    """
    dst = np.zeros((dst_height, dst_width), dtype=np.uint8)
    with rasterio.open(lulc_path) as src:
        warp_reproject(
            source=src.read(1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    return dst


def compute_p_soft(
    lulc_maps: List[np.ndarray],
    num_classes: int,
    source_weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Compute weighted-vote P_soft from LULC maps.

    Args:
        lulc_maps: List of (H, W) integer arrays, one per source.
        num_classes: Number of land-cover classes (0-indexed).
        source_weights: Per-source vote weights. ``None`` gives equal vote.

    Returns:
        p_soft: (C, H, W) float32 array summing to 1 along axis 0.
    """
    h, w = lulc_maps[0].shape
    p_soft = np.zeros((num_classes, h, w), dtype=np.float32)

    weights = source_weights if source_weights is not None else [1.0] * len(lulc_maps)
    if len(weights) != len(lulc_maps):
        raise ValueError(
            f"source_weights length ({len(weights)}) must match "
            f"lulc_maps length ({len(lulc_maps)})"
        )
    total = sum(weights)

    for lulc, weight in zip(lulc_maps, weights):
        for c in range(num_classes):
            p_soft[c] += weight * (lulc == c)

    p_soft /= total
    return p_soft


def compute_w_conf(
    p_soft: np.ndarray,
    alpha: float = 0.6,
    w_embed: Optional[np.ndarray] = None,
    beta: float = 0.0,
    use_border: bool = True,
    entropy_norm: str = "max_entropy",
    bags_mask: Optional[np.ndarray] = None,
    border_radius: int = 10,
) -> np.ndarray:
    """Compute W_conf confidence weight map.

    Two formulas are available, controlled by *use_border*:

    * ``use_border=True`` (default, with the border-mitigation contribution):
      ``W_conf = alpha·w_entropy + (1-alpha-beta)·w_border + beta·w_embed``
      Constraint: ``alpha + beta <= 1.0``.

    * ``use_border=False`` (original paper formula, no border component):
      ``W_conf = (alpha·w_entropy + beta·w_embed) / (alpha + beta_eff)``
      where ``beta_eff = beta`` when *w_embed* is provided, else 0.
      No constraint on ``alpha + beta``.

    Two entropy normalization modes are available, controlled by *entropy_norm*:

    * ``"max_entropy"`` (default): ``w_entropy = 1 - H(P_soft) / log(C)``
      where ``log(C)`` is the theoretical maximum entropy for *C* classes.

    * ``"minmax"``: per-tile min-max normalization following LaTeX Eq. 9-10:
      ``U_norm(i) = (U(i) - min U) / (max U - min U)``
      ``w_entropy(i) = 1 - U_norm(i)``
      When all pixels share the same entropy (zero range), ``w_entropy=1`` for all.

    When *bags_mask* is provided and ``use_border=True``, the border-distance
    component follows LaTeX Eq. 13 (Experiment E5):

    * Boundary pixels are detected on *bags_mask* with an 8-connected
      3x3 morphological filter using :func:`_detect_boundary`.
    * ``d_carta(i)`` is the Euclidean distance from pixel *i* to the nearest
      boundary pixel.
    * ``w_border_carta(i) = min(1, d_carta(i) / R)`` where *R* is
      *border_radius* (default 10 pixels at 2.5 m/pixel).

    When *bags_mask* is ``None`` and ``use_border=True``, the original
    argmax-based border is used with ``w_border = dist / max_dist`` (no fixed
    radius).

    When ``use_border=False``, *bags_mask* is ignored entirely.

    Args:
        p_soft: (C, H, W) float32 probability distribution.
        alpha: Entropy component weight (default 0.6).
        w_embed: (1, H, W) float32 embedding similarity weight, or None.
        beta: Embedding component weight (default 0.0, ignored when w_embed is None).
        use_border: When True (default) include the border-distance component.
            Set to False to replicate the original paper formula.
        entropy_norm: Entropy normalization strategy. ``"max_entropy"`` (default)
            normalizes by the theoretical maximum ``log(C)``; ``"minmax"``
            applies per-tile min-max normalization (LaTeX Eq. 9-10, Experiment E4).
        bags_mask: Optional (H, W) integer array of the BAGS cartographic mask.
            When provided and ``use_border=True``, the boundary is detected on
            *bags_mask* and ``w_border_carta`` is computed via LaTeX Eq. 13.
            Ignored when ``use_border=False``.
        border_radius: Fixed radius *R* in pixels for the ``w_border_carta``
            normalization (default 10). Only used when *bags_mask* is provided.

    Returns:
        w_conf: (1, H, W) float32 array with values in [0, 1].

    Raises:
        ValueError: if ``use_border=True`` and ``alpha + beta > 1.0``, or if
            ``entropy_norm`` is not ``"max_entropy"`` or ``"minmax"``.
    """
    if use_border and alpha + beta > 1.0 + 1e-8:
        raise ValueError(
            f"alpha + beta must be <= 1.0, got alpha={alpha}, beta={beta}."
        )
    if use_border and bags_mask is not None and border_radius <= 0:
        raise ValueError(f"border_radius must be > 0, got {border_radius}.")

    num_classes = p_soft.shape[0]
    eps = 1e-8

    log_p = np.log(np.clip(p_soft, eps, 1.0))
    entropy = -(p_soft * log_p).sum(axis=0)  # (H, W)

    if entropy_norm == "max_entropy":
        w_entropy = 1.0 - entropy / np.log(num_classes)  # (H, W)
    elif entropy_norm == "minmax":
        e_min, e_max = entropy.min(), entropy.max()
        denom = e_max - e_min
        u_norm = (entropy - e_min) / denom if denom > 0.0 else np.zeros_like(entropy)
        w_entropy = 1.0 - u_norm  # (H, W)
    else:
        raise ValueError(
            f"Unknown entropy_norm: '{entropy_norm}'. "
            "Must be 'max_entropy' or 'minmax'."
        )

    if use_border:
        if bags_mask is not None:
            border = _detect_boundary(bags_mask)
            if border.any():
                dist = distance_transform_edt(~border).astype(np.float32)
                w_border = np.minimum(1.0, dist / border_radius)  # (H, W)
            else:
                w_border = np.ones(bags_mask.shape, dtype=np.float32)
        else:
            argmax_map = p_soft.argmax(axis=0)
            border = np.zeros_like(argmax_map, dtype=bool)
            border[:-1, :] |= argmax_map[:-1, :] != argmax_map[1:, :]
            border[1:, :] |= argmax_map[1:, :] != argmax_map[:-1, :]
            border[:, :-1] |= argmax_map[:, :-1] != argmax_map[:, 1:]
            border[:, 1:] |= argmax_map[:, 1:] != argmax_map[:, :-1]
            dist = distance_transform_edt(~border).astype(np.float32)
            max_dist = dist.max() if dist.max() > 0 else 1.0
            w_border = dist / max_dist  # (H, W)

        border_weight = 1.0 - alpha - beta
        w_conf = alpha * w_entropy + border_weight * w_border  # (H, W)

        if w_embed is not None and beta > 0.0:
            w_conf = w_conf + beta * w_embed.squeeze(0)
    else:
        has_embed = w_embed is not None and beta > 0.0
        total_weight = alpha + (beta if has_embed else 0.0)
        w_conf = alpha * w_entropy
        if has_embed:
            w_conf = w_conf + beta * w_embed.squeeze(0)
        if total_weight > 0.0:
            w_conf = w_conf / total_weight

    return w_conf[np.newaxis].astype(np.float32)  # (1, H, W)


# ---------------------------------------------------------------------------
# AEF embedding helpers
# ---------------------------------------------------------------------------


def aggregate_aef_to_image_grid(
    aef_path: str,
    dst_height: int,
    dst_width: int,
    dst_transform: Affine,
    dst_crs,
) -> np.ndarray:
    """Aggregate AEF embeddings using the official Google algorithm.

    Implements the three-step pipeline from the AEF-on-GCS documentation:

    1. **Dequantise** — convert raw integers to floats:
       ``dequant = sign(v) * (v / 127.5)^2``
    2. **Element-wise vector sum** — aggregate dequantised source pixels to
       each target pixel using area-weighted summation (``Resampling.sum``).
    3. **L2 normalise** — divide each pixel vector by its Euclidean norm.

    Only valid for **downscaling** (target pixel area ≥ source pixel area).
    For fine-detail LULC work, use AEF at native 10 m resolution without
    any aggregation.

    Reference:
        https://developers.google.com/earth-engine/guides/aef_on_gcs_readme

    Args:
        aef_path: Path to the raw-integer AEF GeoTIFF ``(D, H_src, W_src)``.
        dst_height: Target grid height in pixels.
        dst_width: Target grid width in pixels.
        dst_transform: Target affine transform.
        dst_crs: Target CRS.

    Returns:
        (H_dst, W_dst, D) float32 L2-normalised embedding array.

    Raises:
        ValueError: if the target pixel area is smaller than the source (upscaling).
    """
    return aggregate_aef_raster_to_grid(
        aef_path, dst_height, dst_width, dst_transform, dst_crs
    )


def load_aef_embedding(
    tile_id: str,
    aef_dir: Path,
    source: str = "gcs",
    dst_height: Optional[int] = None,
    dst_width: Optional[int] = None,
    dst_transform: Optional[Affine] = None,
    dst_crs=None,
    aef_resampling: AEFResamplingStrategy = "auto",
) -> np.ndarray:
    """Load a pre-downloaded AlphaEarth Foundation embedding for a tile.

    For GCS source, when ``dst_height``, ``dst_width``, ``dst_transform``, and
    ``dst_crs`` are all provided, ``aef_resampling`` controls spatial alignment.
    ``"auto"`` uses vector-sum aggregation for downsampling and nearest-neighbor
    assignment for upsampling.

    When the dst parameters are omitted, the raw array is returned at native
    resolution without any transformation.

    **Bilinear/cubic interpolation must NOT be used on AEF embeddings.**  The
    vectors are learned manifold representations; pixel-level interpolation
    produces off-manifold synthetic vectors that corrupt cosine similarities.

    See: https://developers.google.com/earth-engine/guides/aef_on_gcs_readme

    Args:
        tile_id: Tile identifier matching the filename stem.
        aef_dir: Directory containing downloaded embedding files.
        source: ``"gcs"`` for per-pixel GeoTIFF ``{tile_id}.tif`` → ``(H, W, D)``
                float32; ``"hf"`` for patch-level NumPy ``{tile_id}.npy`` → ``(D,)``
                float32.
        dst_height: Target grid height (GCS + aggregation only).
        dst_width: Target grid width (GCS + aggregation only).
        dst_transform: Target affine transform (GCS + aggregation only).
        dst_crs: Target CRS (GCS + aggregation only).
        aef_resampling: ``"auto"``, ``"aggregate"``, ``"nearest"``, or ``"none"``.

    Returns:
        ``(H, W, D)`` float32 for GCS source; ``(D,)`` float32 for HF source.

    Raises:
        FileNotFoundError: if the embedding file does not exist.
        ValueError: if ``source`` is not ``"gcs"`` or ``"hf"``, or if an
                    unsupported resampling strategy is requested.
    """
    aef_dir = Path(aef_dir)
    if source == "gcs":
        path = aef_dir / f"{tile_id}.tif"
        if not path.exists():
            raise FileNotFoundError(f"GCS embedding not found: {path}")
        if all(p is not None for p in [dst_height, dst_width, dst_transform, dst_crs]):
            return resample_aef_raster_to_grid(
                str(path),
                dst_height,
                dst_width,
                dst_transform,
                dst_crs,
                strategy=aef_resampling,
            )
        data, _, _ = read_aef_raster(str(path))
        if np.issubdtype(data.dtype, np.floating) and np.isnan(data).any():
            data = normalize_aef_vectors(data, axis=0)
        return np.transpose(data, (1, 2, 0)).astype(np.float32)  # (H, W, D)
    elif source == "hf":
        path = aef_dir / f"{tile_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"HF embedding not found: {path}")
        return np.load(path).astype(np.float32)  # (D,)
    else:
        raise ValueError(f"Unknown source: '{source}'. Must be 'gcs' or 'hf'.")


def compute_w_embed(
    embedding: np.ndarray,
    p_soft: np.ndarray,
    class_centroids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute embedding-based confidence weight from AEF embeddings.

    Two modes are supported:

    **GCS per-pixel mode** — ``embedding`` shape ``(H, W, D)``:
        Within-tile class centroids are derived from ``p_soft``'s argmax.
        Each pixel's cosine similarity to its class centroid is computed and
        mapped to ``[0, 1]`` via ``(cosine + 1) / 2``.

    **HF patch-level mode** — ``embedding`` shape ``(D,)``:
        ``class_centroids`` (C, D) must be provided (cross-tile averages).
        The tile's dominant class (highest mean ``p_soft``) is used to select
        the matching centroid, and the scalar cosine similarity is broadcast
        to all pixels.

    Args:
        embedding: (H, W, D) for GCS source or (D,) for HF source.
        p_soft: (C, H, W) soft label distribution.
        class_centroids: (C, D) cross-tile centroids, required for HF mode.

    Returns:
        w_embed: (1, H, W) float32 in [0, 1].

    Raises:
        ValueError: if HF mode is used without ``class_centroids``, or if
                    ``embedding`` has an unsupported number of dimensions.
    """
    eps = 1e-8
    num_classes, h, w = p_soft.shape

    if embedding.ndim == 3:
        # GCS per-pixel mode: (H, W, D)
        argmax_map = p_soft.argmax(axis=0)  # (H, W)
        D = embedding.shape[2]
        centroids = np.zeros((num_classes, D), dtype=np.float32)
        valid_vectors = valid_aef_vector_mask(embedding, axis=2, eps=eps)
        for c in range(num_classes):
            mask = (argmax_map == c) & valid_vectors
            if mask.any():
                centroids[c] = embedding[mask].mean(axis=0)

        emb_norm = normalize_aef_vectors(embedding, axis=2, eps=eps)
        centroid_for_pixel = centroids[argmax_map]  # (H, W, D)
        cent_norm = normalize_aef_vectors(centroid_for_pixel, axis=2, eps=eps)
        cos_sim = (emb_norm * cent_norm).sum(axis=2)  # (H, W)
        w_embed = (cos_sim + 1.0) / 2.0  # map [-1,1] → [0,1]
        w_embed = np.where(valid_vectors, w_embed, 0.0)

    elif embedding.ndim == 1:
        # HF patch-level mode: (D,)
        if class_centroids is None:
            raise ValueError(
                "class_centroids (C, D) is required for HF patch-level embeddings."
            )
        dominant_class = int(p_soft.mean(axis=(1, 2)).argmax())
        centroid = class_centroids[dominant_class]  # (D,)
        emb_n = embedding / (np.linalg.norm(embedding) + eps)
        cent_n = centroid / (np.linalg.norm(centroid) + eps)
        cos_sim = float(np.dot(emb_n, cent_n))
        w_embed = np.full((h, w), (cos_sim + 1.0) / 2.0, dtype=np.float32)

    else:
        raise ValueError(
            f"Unexpected embedding shape: {embedding.shape}. "
            "Expected (H, W, D) for GCS or (D,) for HF."
        )

    return np.nan_to_num(w_embed[np.newaxis], nan=0.0).astype(np.float32)


def _compute_hf_class_centroids(
    manifest: pd.DataFrame,
    aef_dir: Path,
) -> np.ndarray:
    """Compute cross-tile class centroids from HF patch-level embeddings.

    Two-pass: loads all tile embeddings and assigns each to its dominant class
    (highest mean P_soft), then averages embeddings within each class.

    Args:
        manifest: DataFrame with ``tile_id`` and ``p_soft_path`` columns.
        aef_dir: Directory containing ``{tile_id}.npy`` HF embedding files.

    Returns:
        (C, D) float32 array of per-class embedding centroids.

    Raises:
        RuntimeError: if no valid tiles are found.
    """
    aef_dir = Path(aef_dir)
    class_embeddings: Dict[int, List[np.ndarray]] = {}
    num_classes: Optional[int] = None

    for _, tile in manifest.iterrows():
        tile_id = str(tile["tile_id"])
        emb_path = aef_dir / f"{tile_id}.npy"
        if not emb_path.exists():
            logger.warning("HF embedding missing for tile %s — skipping", tile_id)
            continue

        embedding = np.load(emb_path).astype(np.float32)

        with rasterio.open(tile["p_soft_path"]) as src:
            p_soft = src.read().astype(np.float32)  # (C, H, W)

        if num_classes is None:
            num_classes = p_soft.shape[0]

        dominant_class = int(p_soft.mean(axis=(1, 2)).argmax())
        class_embeddings.setdefault(dominant_class, []).append(embedding)

    if num_classes is None or not class_embeddings:
        raise RuntimeError("No valid tiles found for computing HF class centroids.")

    D = next(iter(class_embeddings.values()))[0].shape[0]
    centroids = np.zeros((num_classes, D), dtype=np.float32)
    for c, embs in class_embeddings.items():
        centroids[c] = np.stack(embs).mean(axis=0)

    return centroids


def _write_geotiff(
    path: Path,
    data: np.ndarray,
    profile: dict,
    transform: Affine,
    crs,
) -> None:
    """Write a (C, H, W) float32 array as a multi-band GeoTIFF."""
    c, h, w = data.shape
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        dtype="float32",
        count=c,
        height=h,
        width=w,
        transform=transform,
        crs=crs,
        compress="lzw",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data)


# ---------------------------------------------------------------------------
# Per-tile processing
# ---------------------------------------------------------------------------


def process_tile(
    tile_id: str,
    rows: pd.DataFrame,
    output_dir: Path,
    num_classes: int,
    alpha: float,
    aef_dir: Optional[Path] = None,
    aef_source: str = "gcs",
    beta: float = 0.0,
    class_centroids: Optional[np.ndarray] = None,
    use_border: bool = True,
    aef_resampling: AEFResamplingStrategy = "auto",
    entropy_norm: str = "max_entropy",
    border_radius: int = 10,
) -> Tuple[str, str, Path, Path]:
    """Build P_soft and W_conf rasters for one tile.

    Uses the tile's image as the spatial reference grid.  All LULC sources are
    reprojected to the image CRS, transform, and pixel dimensions before
    computing P_soft, so the outputs are always pixel-aligned with the image.

    Args:
        tile_id: Tile identifier (used to name output files).
        rows: Normalized dataframe rows for this tile.
              Required columns: image_path, source_name, lulc_path.
              All rows for the same tile_id must share the same image_path.
        output_dir: Root output directory.
        num_classes: Number of land-cover classes (0-indexed).
        alpha: Entropy component weight for W_conf.
        aef_dir: Directory with pre-downloaded AEF embeddings.  When set,
                 ``w_embed`` is computed and blended into W_conf with weight
                 ``beta``.
        aef_source: ``"gcs"`` (per-pixel GeoTIFF) or ``"hf"`` (patch .npy).
        beta: AEF embedding component weight (ignored when ``aef_dir`` is None).
        class_centroids: (C, D) centroids for HF mode; computed externally in a
                         two-pass step before dispatching workers.
        use_border: When True (default), include the border-distance component in
                    W_conf (user contribution). Set to False for the original
                    paper formula (entropy-only or entropy + embed).
        aef_resampling: AEF raster alignment strategy for GCS mode. ``"auto"``
                        aggregates when target pixels are coarser than AEF and
                        uses nearest-neighbor when target pixels are finer.
        entropy_norm: Entropy normalization strategy passed to ``compute_w_conf``.
                      ``"max_entropy"`` (default) or ``"minmax"``.
        The cartographic boundary source is taken from the CSV's ``mask_path``,
            ``cartographic_path``, or ``bags_path`` column.
        border_radius: Fixed radius *R* in pixels for the ``w_border_carta``
            normalization (default 10, matching the 2.5 m/pixel paper setup).
            Only used when the cartographic source is present.

    Returns:
        Tuple of (tile_id, image_path, p_soft_path, w_conf_path).
    """
    image_path = rows.iloc[0]["image_path"]
    img_h, img_w, img_transform, img_crs, img_profile = _get_image_grid(image_path)

    lulc_maps = []
    for _, row in rows.iterrows():
        lulc = _reproject_lulc_to_image_grid(
            row["lulc_path"], img_h, img_w, img_transform, img_crs
        )
        lulc_maps.append(lulc)

    p_soft = compute_p_soft(lulc_maps, num_classes)

    bags_mask: Optional[np.ndarray] = None
    if use_border:
        bags_rows = rows[rows["source_name"] == _CARTOGRAPHIC_SOURCE_NAME]
        if not bags_rows.empty:
            bags_mask = _reproject_lulc_to_image_grid(
                bags_rows.iloc[0]["lulc_path"], img_h, img_w, img_transform, img_crs
            )
        else:
            logger.warning(
                "cartographic mask source not found in tile %s; "
                "falling back to argmax-based border",
                tile_id,
            )

    w_embed: Optional[np.ndarray] = None
    if aef_dir is not None and beta > 0.0:
        try:
            embedding = load_aef_embedding(
                tile_id,
                aef_dir,
                source=aef_source,
                dst_height=img_h,
                dst_width=img_w,
                dst_transform=img_transform,
                dst_crs=img_crs,
                aef_resampling=aef_resampling,
            )
            w_embed = compute_w_embed(
                embedding, p_soft, class_centroids=class_centroids
            )
        except FileNotFoundError:
            logger.warning(
                "AEF embedding not found for tile %s — skipping w_embed", tile_id
            )
        except ValueError as exc:
            logger.error(
                "AEF embedding error for tile %s — skipping w_embed: %s",
                tile_id,
                exc,
            )

    w_conf = compute_w_conf(
        p_soft,
        alpha=alpha,
        w_embed=w_embed,
        beta=beta,
        use_border=use_border,
        entropy_norm=entropy_norm,
        bags_mask=bags_mask,
        border_radius=border_radius,
    )

    transform, crs, profile = img_transform, img_crs, img_profile

    p_soft_path = output_dir / "p_soft" / f"{tile_id}.tif"
    w_conf_path = output_dir / "w_conf" / f"{tile_id}.tif"

    _write_geotiff(p_soft_path, p_soft, profile, transform, crs)
    _write_geotiff(w_conf_path, w_conf, profile, transform, crs)

    logger.info("Tile %s done → p_soft=%s w_conf=%s", tile_id, p_soft_path, w_conf_path)
    return tile_id, image_path, p_soft_path, w_conf_path


# ---------------------------------------------------------------------------
# Patch generation (for SoftLabelWindowedDataset)
# ---------------------------------------------------------------------------


def generate_patch_rows(
    height: int,
    width: int,
    patch_size: int,
    stride: int,
) -> List[Tuple[int, int]]:
    """Return all valid (row_off, col_off) pairs for a sliding window.

    Patches that would exceed the raster boundary are excluded.

    Args:
        height: Raster height in pixels.
        width: Raster width in pixels.
        patch_size: Square patch side length in pixels.
        stride: Step between consecutive patches in pixels.

    Returns:
        List of ``(row_off, col_off)`` tuples, ordered row-major.
    """
    offsets = []
    for r in range(0, height - patch_size + 1, stride):
        for c in range(0, width - patch_size + 1, stride):
            offsets.append((r, c))
    return offsets


def expand_manifest_to_patches(
    manifest: pd.DataFrame,
    patch_size: int,
    stride: int,
) -> pd.DataFrame:
    """Expand a tile manifest into one row per patch for SoftLabelWindowedDataset.

    Reads each tile's image dimensions from ``image_path`` and generates
    all valid sliding-window patch offsets.

    Args:
        manifest: DataFrame with at least ``tile_id``, ``image_path``,
                  ``p_soft_path`` columns.  ``w_conf_path`` is optional.
        patch_size: Square patch side length in pixels.
        stride: Sliding-window step in pixels.

    Returns:
        DataFrame with columns:
        ``tile_id``, ``image_path``, ``p_soft_path``, [``w_conf_path``],
        ``row_off``, ``col_off``, ``patch_size``.
    """
    has_wconf = "w_conf_path" in manifest.columns
    patch_rows = []
    for _, tile in manifest.iterrows():
        with rasterio.open(tile["image_path"]) as src:
            h, w = src.height, src.width
        for r_off, c_off in generate_patch_rows(h, w, patch_size, stride):
            row: Dict[str, Any] = {
                "tile_id": tile["tile_id"],
                "image_path": tile["image_path"],
                "p_soft_path": tile["p_soft_path"],
                "row_off": r_off,
                "col_off": c_off,
                "patch_size": patch_size,
            }
            if has_wconf:
                row["w_conf_path"] = tile["w_conf_path"]
            patch_rows.append(row)
    return pd.DataFrame(patch_rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    input_csv: str,
    output_dir: Path,
    num_classes: int = 4,
    alpha: float = 0.6,
    max_workers: int = 4,
    patch_size: Optional[int] = None,
    stride: Optional[int] = None,
    aef_embeddings_dir: Optional[str] = None,
    aef_source: str = "gcs",
    beta: float = 0.0,
    use_border: bool = True,
    aef_resampling: AEFResamplingStrategy = "auto",
    entropy_norm: str = "max_entropy",
    image_key: str = "image_path",
    mask_key: str = "mask_path",
    lulc_keys: Optional[List[str]] = None,
    border_radius: int = 10,
) -> Path:
    """Build P_soft and W_conf rasters for all tiles in *input_csv*.

    Args:
        input_csv: CSV in wide format with one row per tile.
        output_dir: Root directory for output p_soft/ and w_conf/ sub-directories.
        num_classes: Number of land-cover classes (0-indexed).
        alpha: Entropy/border blend weight for W_conf.
        max_workers: Number of parallel worker processes.
        patch_size: When set, expand the manifest into patch rows for
                    SoftLabelWindowedDataset.
        stride: Sliding-window stride (default: same as patch_size, no overlap).
        aef_embeddings_dir: Directory with pre-downloaded AEF embeddings.
        aef_source: ``"gcs"`` or ``"hf"``.
        beta: AEF embedding blend weight (alpha + beta must be <= 1.0 when
              ``use_border=True``).
        use_border: When True (default), include the border-distance component in
                    W_conf. Set to False to replicate the original paper formula.
        aef_resampling: AEF raster alignment strategy for GCS mode.
        entropy_norm: Entropy normalization strategy. ``"max_entropy"`` (default)
                      normalizes by ``log(C)``; ``"minmax"`` applies per-tile
                      min-max normalization (LaTeX Eq. 9-10, Experiment E4).
        image_key: CSV column with image paths.
        mask_key: CSV column with BAGS cartographic mask paths.
        lulc_keys: CSV columns with external LULC source paths. The soft label
                   is an equal vote over ``mask_key`` plus these columns.
        border_radius: Fixed radius *R* in pixels for ``w_border_carta``
            normalization (default 10). Only used when the cartographic source
            is present.

    Returns:
        Path to the written manifest CSV.
    """
    df = _normalize_sources_dataframe(
        pd.read_csv(input_csv),
        image_key=image_key,
        mask_key=mask_key,
        lulc_keys=lulc_keys,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aef_dir = Path(aef_embeddings_dir) if aef_embeddings_dir else None

    groups = {tid: grp for tid, grp in df.groupby("tile_id")}
    results = []

    class_centroids: Optional[np.ndarray] = None
    if aef_dir is not None and aef_source == "hf" and beta > 0.0:
        logger.info("Computing HF class centroids (two-pass)…")
        logger.warning(
            "HF centroid pre-computation requires a completed p_soft pass.  "
            "Run once without --aef-embeddings-dir to build p_soft, then run again "
            "with --aef-embeddings-dir to blend w_embed.  "
            "Continuing without class centroids (w_embed will use dominant-class broadcast)."
        )

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                process_tile,
                tile_id,
                grp,
                output_dir,
                num_classes,
                alpha,
                aef_dir,
                aef_source,
                beta,
                class_centroids=class_centroids,
                use_border=use_border,
                aef_resampling=aef_resampling,
                entropy_norm=entropy_norm,
                border_radius=border_radius,
            ): tile_id
            for tile_id, grp in groups.items()
        }
        for future in as_completed(futures):
            tile_id = futures[future]
            try:
                results.append(future.result())
            except Exception:
                logger.exception("Tile %s failed", tile_id)

    manifest = pd.DataFrame(
        results, columns=["tile_id", "image_path", "p_soft_path", "w_conf_path"]
    )

    if patch_size is not None:
        _stride = stride if stride is not None else patch_size
        manifest = expand_manifest_to_patches(manifest, patch_size, _stride)
        manifest_path = output_dir / "soft_label_patches.csv"
        logger.info(
            "Patch manifest written to %s (%d patches, patch_size=%d, stride=%d)",
            manifest_path,
            len(manifest),
            patch_size,
            _stride,
        )
    else:
        manifest_path = output_dir / "soft_label_manifest.csv"
        logger.info("Manifest written to %s (%d tiles)", manifest_path, len(manifest))

    manifest.to_csv(manifest_path, index=False)
    return manifest_path
