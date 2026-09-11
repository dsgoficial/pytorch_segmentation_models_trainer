# -*- coding: utf-8 -*-
"""SLICO-based label correction for noisy segmentation masks.

Pre-processing tool that corrects noisy GeoTIFF masks using SLICO
(zero-parameter SLIC superpixels, Achanta & Susstrunk 2012) + majority vote
within each superpixel across the original topographic vector source and
optional LULC auxiliary maps.

This is a region-generation ablation of ``tools.sam_correction``: stages 2/3
(majority-vote consensus, per-class eligibility rule) are identical — both
call :func:`apply_region_correction` — only stage 1 (region partition) differs.
SLICO produces a strict, non-overlapping label map instead of SAM AMG's
possibly-overlapping mask list, so it is generated with a deliberately fine,
fixed oversegmentation (``n_segments`` superpixels per 256x256 patch, default
1000, ~65 px/segment) rather than tuned to match SAM's region count — this
keeps the two region generators independent and comparable without one
depending on the other having already run.
"""

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from pytorch_segmentation_models_trainer.tools.region_correction.correction import (
    apply_region_correction,
    chunk_cache_key,
    parse_correction_targets,
)
from pytorch_segmentation_models_trainer.tools.region_correction.segment_cache import (
    LabelMapCache,
    SegmentListCache,
)

_PATCH_AREA_PX = 256 * 256


@dataclass
class SLICOLabelCorrectionConfig:
    """Configuration for SLICO-based label correction.

    Args:
        coreset_csv: Path to CSV with columns mask_path, row_off, col_off, patch_size.
        masks_dir: Directory containing original GeoTIFF mask files.
        targets: List of dicts with keys ``classes`` (list of int) and
            ``output_dir`` (str). Each target produces a separate output directory
            with SLICO-corrected masks for the specified class set.
        mbtiles_path: Path to MBTiles file used as RGB source imagery for SLICO.
        lulc_paths: Paths to auxiliary LULC rasters/VRTs included in the
            majority vote alongside the topographic vector source.
        include_bags: Whether to include the original topographic vector source
            mask in the majority vote (default True).
        num_classes: Number of valid semantic classes (default 6).
        nodata_val: Pixel value treated as nodata/invalid (default 255).
        chunk_size: Tile processing chunk size in pixels (default 1024).
        n_segments: Target superpixel count per 256x256 patch (default 1000,
            ~65 px/segment — deliberate oversegmentation, see module docstring).
            Scaled proportionally to the actual chunk area processed, so the
            per-pixel superpixel density stays constant regardless of
            ``chunk_size``.
        match_sam_cache_dir: Optional path to an existing SAM segment NPZ cache
            (``tools.sam_correction`` ``cache_dir``). When a chunk has a cached
            SAM entry, its mask count is used as ``n_segments`` for that chunk
            instead of the density-based default — secondary sensitivity
            analysis only (see module docstring); leave empty for the primary,
            SAM-independent comparison.
        cache_dir: Directory for NPZ label-map cache. Empty string disables cache.
        start_idx: First tile index to process (inclusive, for multi-worker splits).
        end_idx: Last tile index to process (exclusive).

    Example YAML:

    .. code-block:: yaml

        coreset_csv: /data/coreset.csv
        masks_dir: /data/masks
        targets:
          - classes: [3, 5]
            output_dir: /data/masks_slico_gc
        mbtiles_path: /data/images/tiles.mbtiles
        lulc_paths:
          - /data/lulc/mapbiomas.vrt
          - /data/lulc/esri.vrt
        n_segments: 1000
        cache_dir: /data/slico_cache
    """

    coreset_csv: str
    masks_dir: str
    targets: list
    mbtiles_path: str
    lulc_paths: list = field(default_factory=list)
    include_bags: bool = True
    num_classes: int = 6
    nodata_val: int = 255
    chunk_size: int = 1024
    n_segments: int = 1000
    match_sam_cache_dir: str = ""
    cache_dir: str = ""
    start_idx: int = 0
    end_idx: int = 999999


class SlicoLabelCorrector:
    """Orchestrates SLICO-based label correction across a coreset of tiles.

    Loads the coreset CSV and processes tiles in order. For each tile, chunks
    are processed; SLICO is run once per chunk (or loaded from cache) and the
    result is applied to every registered target. Mirrors
    :class:`pytorch_segmentation_models_trainer.tools.sam_correction.SamLabelCorrector`
    structurally — see that class and the module docstring for how the two relate.

    Args:
        config: :class:`SLICOLabelCorrectionConfig` instance.

    Example::

        config = SLICOLabelCorrectionConfig(
            coreset_csv="/data/coreset.csv",
            masks_dir="/data/masks",
            targets=[{"classes": [3, 5], "output_dir": "/data/out"}],
            mbtiles_path="/data/tiles.mbtiles",
        )
        corrector = SlicoLabelCorrector(config)
        stats = corrector.run()
    """

    def __init__(self, config: SLICOLabelCorrectionConfig) -> None:
        self._cfg = config
        self._cache = LabelMapCache(config.cache_dir)
        self._sam_cache = (
            SegmentListCache(config.match_sam_cache_dir)
            if config.match_sam_cache_dir
            else None
        )
        self._targets = parse_correction_targets(config.targets)
        self._lulc_paths = [Path(p) for p in config.lulc_paths]
        self._masks_dir = Path(config.masks_dir)
        self._mbtiles = Path(config.mbtiles_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Run SLICO correction over all tiles in the coreset CSV.

        Returns:
            Summary dict with keys ``n_tiles``, ``elapsed_s``, and per-target
            ``total_changed`` / ``total_target`` counters.
        """
        df = pd.read_csv(self._cfg.coreset_csv, low_memory=False)
        if "patch_size" not in df.columns:
            df["patch_size"] = 256

        all_tiles = sorted(df["mask_path"].unique())
        tiles = all_tiles[self._cfg.start_idx : self._cfg.end_idx]

        for _, out_dir in self._targets:
            out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        t_start = time.time()

        for tile_name in tqdm(tiles, desc="Tiles", unit="tile"):
            patch_rows = df[df["mask_path"] == tile_name].copy()
            stats = self._process_tile(tile_name, patch_rows)
            results.append(stats)

        elapsed = time.time() - t_start
        return {"n_tiles": len(tiles), "elapsed_s": round(elapsed, 1), "tiles": results}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_cache_key(self, tile_name: str, window, n_segments: int) -> str:
        return f"{chunk_cache_key(tile_name, window)}_n{n_segments}"

    def _resolve_n_segments(self, tile_name: str, window) -> int:
        if self._sam_cache is not None:
            sam_masks = self._sam_cache.get(chunk_cache_key(tile_name, window))
            if sam_masks:
                return len(sam_masks)
        density = self._cfg.n_segments / _PATCH_AREA_PX
        area = window.height * window.width
        return max(1, round(density * area))

    def _process_tile(self, tile_name: str, patch_rows: pd.DataFrame) -> Dict:
        import rasterio
        from rasterio.windows import Window

        src_path = self._masks_dir / tile_name
        corrected_paths = []
        for _, out_dir in self._targets:
            cp = out_dir / tile_name
            if not cp.exists():
                shutil.copy2(src_path, cp)
            corrected_paths.append(cp)

        chunk_targets = [
            (cls, corrected_paths[i]) for i, (cls, _) in enumerate(self._targets)
        ]

        row_min = int(patch_rows["row_off"].min())
        col_min = int(patch_rows["col_off"].min())
        row_max = int((patch_rows["row_off"] + patch_rows["patch_size"]).max())
        col_max = int((patch_rows["col_off"] + patch_rows["patch_size"]).max())

        tile_changed = [0] * len(self._targets)
        tile_target = [0] * len(self._targets)
        n_chunks = 0
        n_skipped = 0

        chunk = self._cfg.chunk_size
        with rasterio.open(src_path) as mask_src:
            for r0 in range(row_min, row_max, chunk):
                for c0 in range(col_min, col_max, chunk):
                    r1 = min(r0 + chunk, row_max, mask_src.height)
                    c1 = min(c0 + chunk, col_max, mask_src.width)
                    if r1 <= r0 or c1 <= c0:
                        continue
                    window = Window(c0, r0, c1 - c0, r1 - r0)
                    chunk_results = self._process_chunk(mask_src, chunk_targets, window)
                    n_chunks += 1
                    if all(t == 0 for _, t in chunk_results):
                        n_skipped += 1
                    for i, (changed, target) in enumerate(chunk_results):
                        tile_changed[i] += changed
                        tile_target[i] += target

        return {
            "tile": tile_name,
            "n_chunks": n_chunks,
            "n_skipped": n_skipped,
            "per_target": [
                {
                    "output_dir": str(self._targets[i][1]),
                    "classes": sorted(self._targets[i][0]),
                    "n_target": tile_target[i],
                    "n_changed": tile_changed[i],
                    "pct_changed": round(
                        100.0 * tile_changed[i] / max(1, tile_target[i]), 2
                    ),
                }
                for i in range(len(self._targets))
            ],
        }

    def _process_chunk(
        self,
        mask_src,
        chunk_targets: List[Tuple[FrozenSet[int], Path]],
        window,
    ) -> List[Tuple[int, int]]:
        import rasterio
        from skimage.segmentation import slic

        from pytorch_segmentation_models_trainer.tools.mbtiles.alignment import (
            read_source_aligned_to_mask_window,
        )

        bags_raw = mask_src.read(1, window=window)

        has_any = any(np.isin(bags_raw, list(cls)).any() for cls, _ in chunk_targets)
        if not has_any:
            return [(0, 0)] * len(chunk_targets)

        try:
            image_chw = read_source_aligned_to_mask_window(
                source_path=self._mbtiles,
                mask_src=mask_src,
                window=window,
                selected_bands=[1, 2, 3],
                image_dtype="uint8",
                image_resampling="bilinear",
            )
            lulc_maps = [
                read_source_aligned_to_mask_window(
                    source_path=p,
                    mask_src=mask_src,
                    window=window,
                    selected_bands=[1],
                    image_dtype="uint8",
                    image_resampling="nearest",
                )[0]
                for p in self._lulc_paths
            ]
        except Exception:
            return [(0, 0)] * len(chunk_targets)

        image_hwc = np.ascontiguousarray(image_chw.transpose(1, 2, 0))

        n_segments = self._resolve_n_segments(mask_src.name, window)
        cache_key = self._chunk_cache_key(mask_src.name, window, n_segments)
        label_map = self._cache.get(cache_key)
        if label_map is None:
            label_map = slic(
                image_hwc,
                n_segments=n_segments,
                slic_zero=True,
                channel_axis=-1,
                start_label=0,
            ).astype(np.int32)
            self._cache.put(cache_key, label_map)

        results = []
        for classes_to_correct, corrected_path in chunk_targets:
            if not np.isin(bags_raw, list(classes_to_correct)).any():
                results.append((0, 0))
                continue

            corrected = apply_region_correction(
                bags_raw=bags_raw,
                segments=label_map,
                lulc_maps=lulc_maps,
                classes_to_correct=classes_to_correct,
                num_classes=self._cfg.num_classes,
                include_bags=self._cfg.include_bags,
            )
            n_target = int(np.isin(bags_raw, list(classes_to_correct)).sum())
            n_changed = int((corrected != bags_raw).sum())

            with rasterio.open(corrected_path, "r+") as dst:
                dst.write(corrected[np.newaxis, :, :], window=window)

            results.append((n_changed, n_target))

        return results
