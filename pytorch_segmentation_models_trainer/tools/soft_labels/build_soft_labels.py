# -*- coding: utf-8 -*-
"""Build P_soft and W_conf rasters from multiple LULC source products.

Pipeline (Xiao et al. 2026, JSTARS):

1. Load M source LULC rasters and reproject each to the image grid.
2. Apply temporal weight w_t for each source (derived from acquisition date).
3. Compute P_soft(i, c) = Σ_t w_t · 1[source_t(i) == c] / Σ_t w_t
4. Compute entropy-based uncertainty:
     H(i) = -Σ_c P_soft(i,c) · log(P_soft(i,c))
     w_entropy(i) = 1 - H(i) / log(C)
5. Compute border distance weight:
     argmax_map = argmax(P_soft, axis=0)
     border mask = morphological boundary of argmax_map
     w_border(i) = distance_transform_edt(~border)(i) / max_dist
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _get_image_grid(image_path: str):
    """Return (height, width, transform, crs, profile) of the reference image."""
    with rasterio.open(image_path) as src:
        return src.height, src.width, src.transform, src.crs, src.profile


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
    weights: List[float],
    num_classes: int,
) -> np.ndarray:
    """Compute P_soft from weighted LULC maps.

    Args:
        lulc_maps: List of (H, W) integer arrays, one per source.
        weights: Scalar weight for each source (temporal or reliability-based).
        num_classes: Number of land-cover classes (0-indexed).

    Returns:
        p_soft: (C, H, W) float32 array summing to 1 along axis 0.
    """
    h, w = lulc_maps[0].shape
    p_soft = np.zeros((num_classes, h, w), dtype=np.float32)
    total_weight = sum(weights)

    for lulc, wt in zip(lulc_maps, weights):
        for c in range(num_classes):
            p_soft[c] += wt * (lulc == c)

    p_soft /= total_weight
    return p_soft


def compute_w_conf(
    p_soft: np.ndarray,
    alpha: float = 0.6,
    w_embed: Optional[np.ndarray] = None,
    beta: float = 0.0,
    use_border: bool = True,
) -> np.ndarray:
    """Compute W_conf confidence weight map.

    Two formulas are available, controlled by *use_border*:

    * ``use_border=True`` (default, with the border-mitigation contribution):
      ``W_conf = alpha·w_entropy + (1-alpha-beta)·w_border + beta·w_embed``
      Constraint: ``alpha + beta <= 1.0``.

    * ``use_border=False`` (original paper formula — no border component):
      ``W_conf = (alpha·w_entropy + beta·w_embed) / (alpha + beta_eff)``
      where ``beta_eff = beta`` when *w_embed* is provided, else 0.
      No constraint on ``alpha + beta``.

    Args:
        p_soft: (C, H, W) float32 probability distribution.
        alpha: Entropy component weight (default 0.6).
        w_embed: (1, H, W) float32 embedding similarity weight, or None.
        beta: Embedding component weight (default 0.0, ignored when w_embed is None).
        use_border: When True (default) include the border-distance component,
            which is the user's contribution beyond the original paper.
            Set to False to replicate the original paper formula.

    Returns:
        w_conf: (1, H, W) float32 array with values in [0, 1].

    Raises:
        ValueError: if ``use_border=True`` and ``alpha + beta > 1.0``.
    """
    if use_border and alpha + beta > 1.0 + 1e-8:
        raise ValueError(
            f"alpha + beta must be <= 1.0, got alpha={alpha}, beta={beta}."
        )

    num_classes = p_soft.shape[0]
    eps = 1e-8

    log_p = np.log(np.clip(p_soft, eps, 1.0))
    entropy = -(p_soft * log_p).sum(axis=0)  # (H, W)
    max_entropy = np.log(num_classes)
    w_entropy = 1.0 - entropy / max_entropy  # (H, W)

    if use_border:
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
    eps = 1e-8
    with rasterio.open(aef_path) as src:
        src_pixel_area = abs(src.transform.a * src.transform.e)
        dst_pixel_area = abs(dst_transform.a * dst_transform.e)
        if dst_pixel_area < src_pixel_area * 0.99:
            raise ValueError(
                f"AEF aggregation only supports downscaling (target pixel area must "
                f"be >= source pixel area). Source: {src_pixel_area:.6g}, "
                f"target: {dst_pixel_area:.6g}. For finer target resolutions, "
                f"download AEF at the native resolution matching the training image."
            )

        # Step 1: read and dequantise all bands
        raw = src.read().astype(np.float32)  # (D, H_src, W_src)
        dequant = np.sign(raw) * (raw / 127.5) ** 2  # (D, H_src, W_src)

        num_bands = src.count
        dst_sum = np.zeros((num_bands, dst_height, dst_width), dtype=np.float32)

        # Step 2: sum each dequantised band to the target grid
        for band_idx in range(num_bands):
            band_dst = np.zeros((dst_height, dst_width), dtype=np.float32)
            warp_reproject(
                source=dequant[band_idx],
                destination=band_dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.sum,
            )
            dst_sum[band_idx] = band_dst

    # Step 3: L2 normalise per pixel
    norm = np.linalg.norm(dst_sum, axis=0, keepdims=True)  # (1, H, W)
    normalised = dst_sum / (norm + eps)  # (D, H, W)

    return np.transpose(normalised, (1, 2, 0)).astype(np.float32)  # (H, W, D)


def load_aef_embedding(
    tile_id: str,
    aef_dir: Path,
    source: str = "gcs",
    dst_height: Optional[int] = None,
    dst_width: Optional[int] = None,
    dst_transform: Optional[Affine] = None,
    dst_crs=None,
) -> np.ndarray:
    """Load a pre-downloaded AlphaEarth Foundation embedding for a tile.

    For GCS source, when ``dst_height``, ``dst_width``, ``dst_transform``, and
    ``dst_crs`` are all provided, the official Google aggregation algorithm is
    applied (dequantise → element-wise vector sum → L2 normalise).  This is
    valid only for downscaling; attempting to produce a finer grid than the
    source raises ``ValueError``.

    When the dst parameters are omitted, the raw array is returned at native
    resolution without any transformation.

    **Standard spatial resampling (nearest-neighbour, bilinear, average) must
    NOT be used on AEF embeddings.**  The 64-D vectors are learned manifold
    representations; pixel-level interpolation produces off-manifold synthetic
    vectors that corrupt cosine similarities.

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

    Returns:
        ``(H, W, D)`` float32 for GCS source; ``(D,)`` float32 for HF source.

    Raises:
        FileNotFoundError: if the embedding file does not exist.
        ValueError: if ``source`` is not ``"gcs"`` or ``"hf"``, or if
                    upscaling is requested (target finer than source).
    """
    aef_dir = Path(aef_dir)
    if source == "gcs":
        path = aef_dir / f"{tile_id}.tif"
        if not path.exists():
            raise FileNotFoundError(f"GCS embedding not found: {path}")
        if all(p is not None for p in [dst_height, dst_width, dst_transform, dst_crs]):
            return aggregate_aef_to_image_grid(
                str(path), dst_height, dst_width, dst_transform, dst_crs
            )
        with rasterio.open(path) as src:
            data = src.read()  # (D, H, W)
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
        for c in range(num_classes):
            mask = argmax_map == c
            if mask.any():
                centroids[c] = embedding[mask].mean(axis=0)

        emb_norm = embedding / (np.linalg.norm(embedding, axis=2, keepdims=True) + eps)
        centroid_for_pixel = centroids[argmax_map]  # (H, W, D)
        cent_norm = centroid_for_pixel / (
            np.linalg.norm(centroid_for_pixel, axis=2, keepdims=True) + eps
        )
        cos_sim = (emb_norm * cent_norm).sum(axis=2)  # (H, W)
        w_embed = (cos_sim + 1.0) / 2.0  # map [-1,1] → [0,1]

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

    return w_embed[np.newaxis].astype(np.float32)  # (1, H, W)


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
) -> Tuple[str, str, Path, Path]:
    """Build P_soft and W_conf rasters for one tile.

    Uses the tile's image as the spatial reference grid.  All LULC sources are
    reprojected to the image CRS, transform, and pixel dimensions before
    computing P_soft, so the outputs are always pixel-aligned with the image.

    Args:
        tile_id: Tile identifier (used to name output files).
        rows: DataFrame rows for this tile.
              Required columns: image_path, lulc_path, weight.
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

    Returns:
        Tuple of (tile_id, image_path, p_soft_path, w_conf_path).
    """
    image_path = rows.iloc[0]["image_path"]
    img_h, img_w, img_transform, img_crs, img_profile = _get_image_grid(image_path)

    lulc_maps, weights = [], []
    for _, row in rows.iterrows():
        lulc = _reproject_lulc_to_image_grid(
            row["lulc_path"], img_h, img_w, img_transform, img_crs
        )
        lulc_maps.append(lulc)
        weights.append(float(row["weight"]))

    p_soft = compute_p_soft(lulc_maps, weights, num_classes)

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
        p_soft, alpha=alpha, w_embed=w_embed, beta=beta, use_border=use_border
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
) -> Path:
    """Build P_soft and W_conf rasters for all tiles in *input_csv*.

    Args:
        input_csv: CSV with columns tile_id, image_path, source_name, lulc_path, weight.
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

    Returns:
        Path to the written manifest CSV.
    """
    df = pd.read_csv(input_csv)
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
                class_centroids,
                use_border,
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
