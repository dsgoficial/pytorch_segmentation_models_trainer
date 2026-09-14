# -*- coding: utf-8 -*-
"""SAM-based label correction for noisy segmentation masks.

Pre-processing tool that corrects noisy GeoTIFF masks using SAM AMG
(Automatic Mask Generation) + majority vote within SAM segments across the
mask being corrected (whatever raster ``masks_dir`` holds — not tied to any
particular dataset) and optional LULC auxiliary maps.
"""

import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from pytorch_segmentation_models_trainer.tools.mbtiles.image_source import (
    ImageSourceSpec,
    resolve_image_source,
)
from pytorch_segmentation_models_trainer.tools.region_correction.correction import (
    apply_region_correction,
    chunk_cache_key,
    parse_correction_targets,
)
from pytorch_segmentation_models_trainer.tools.region_correction.segment_cache import (
    SegmentListCache as SAMSegmentCache,
)


@dataclass
class SAMLabelCorrectionConfig:
    """Configuration for SAM-based label correction.

    Args:
        coreset_csv: Path to CSV with columns mask_path, row_off, col_off, patch_size.
        masks_dir: Directory containing original GeoTIFF mask files.
        targets: List of dicts with keys ``classes`` (list of int) and
            ``output_dir`` (str). Each target produces a separate output directory
            with SAM-corrected masks for the specified class set.
        sam_checkpoint: Path to SAM ViT checkpoint (.pth).
        mbtiles_path: RGB source imagery for SAM. Either a path to a single file
            (MBTiles, VRT, or any rasterio-readable raster — existing behavior),
            or a directory spec (``{directory, extensions, recursive}``) to search
            a folder of per-tile raster files instead — see
            ``tools.mbtiles.image_source.resolve_image_source``.
        lulc_paths: Auxiliary LULC rasters included in the majority vote
            alongside the mask being corrected. Each entry accepts the
            same single-file-or-directory forms as ``mbtiles_path``.
        include_base_mask: Whether the mask being corrected (whatever raster
            ``masks_dir`` holds) counts as one vote source alongside
            ``lulc_paths``, rather than only being the thing corrected
            (default True).
        sam_model_type: SAM model registry key (default ``"vit_b"``).
        device: Torch device string (default ``"cuda:0"``).
        num_classes: Number of valid semantic classes (default 6).
        nodata_val: Pixel value treated as nodata/invalid (default 255).
        chunk_size: Tile processing chunk size in pixels (default 1024).
        points_per_side: SAM AMG grid density (default 32).
        points_per_batch: Number of point prompts run through SAM in one forward
            pass (default 64). SAM AMG's own internal batching knob — raising it
            trades GPU memory for fewer forward passes per chunk (e.g.
            ``points_per_side=32`` gives 1024 points/chunk = 16 passes at the
            default 64/batch).
        pred_iou_thresh: SAM AMG predicted-IoU threshold (default 0.80).
        stability_score_thresh: SAM AMG stability-score threshold (default 0.90).
        min_mask_region_area: Minimum SAM segment area in pixels (default 200).
        cache_dir: Directory for NPZ segment cache. Empty string disables cache.
        prefetch: When True (default), the next chunk's aligned imagery/LULC
            arrays are read on a background thread while SAM processes the
            current chunk, overlapping I/O with GPU compute. SAM itself is not
            batched across chunks (its public API processes one image per call);
            this only hides the I/O wait, not the GPU compute time.
        start_idx: First tile index to process (inclusive, for multi-GPU splits).
        end_idx: Last tile index to process (exclusive).

    Example YAML:

    .. code-block:: yaml

        coreset_csv: /data/coreset.csv
        masks_dir: /data/masks
        targets:
          - classes: [3, 5]
            output_dir: /data/masks_sam_gc
        sam_checkpoint: /models/sam_vit_b_01ec64.pth
        mbtiles_path: /data/images/tiles.mbtiles
        lulc_paths:
          - /data/lulc/mapbiomas.vrt
          - /data/lulc/esri.vrt
        cache_dir: /data/sam_cache
    """

    coreset_csv: str
    masks_dir: str
    targets: list
    sam_checkpoint: str
    mbtiles_path: ImageSourceSpec
    lulc_paths: list = field(default_factory=list)
    include_base_mask: bool = True
    sam_model_type: str = "vit_b"
    device: str = "cuda:0"
    num_classes: int = 6
    nodata_val: int = 255
    chunk_size: int = 1024
    points_per_side: int = 32
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.80
    stability_score_thresh: float = 0.90
    min_mask_region_area: int = 200
    cache_dir: str = ""
    prefetch: bool = True
    start_idx: int = 0
    end_idx: int = 999999


def apply_sam_correction(
    base_mask: np.ndarray,
    sam_masks: List[Dict],
    lulc_maps: List[np.ndarray],
    classes_to_correct: FrozenSet[int],
    num_classes: int = 6,
    include_base_mask: bool = True,
) -> np.ndarray:
    """Apply SAM-based majority-vote correction to a mask array.

    For each SAM segment that contains at least one pixel from
    ``classes_to_correct``, the winning class is determined by majority vote
    across all sources (optionally ``base_mask`` itself + each LULC map). The
    winner is written to every pixel in the segment. Non-target class pixels
    are always restored from ``base_mask`` after processing.

    SAM masks are processed in ascending (predicted_iou, area) order so that
    higher-confidence, larger segments overwrite smaller/less-confident ones in
    overlapping regions.

    Args:
        base_mask: The mask being corrected (H, W) uint8 — whatever raster
            ``masks_dir`` holds; not tied to any particular dataset.
        sam_masks: List of SAM segment dicts with keys ``segmentation``
            (bool H×W), ``predicted_iou`` (float), ``area`` (int).
        lulc_maps: List of auxiliary class arrays (H, W) uint8 used as extra
            votes alongside ``base_mask`` — never corrected themselves, only
            consulted for consensus.
        classes_to_correct: Set of class indices eligible for correction.
        num_classes: Number of valid class indices (values >= num_classes ignored).
        include_base_mask: If True, ``base_mask`` is counted as one vote source.

    Returns:
        Corrected mask array (H, W) uint8, same shape as ``base_mask``.
    """
    return apply_region_correction(
        base_mask=base_mask,
        segments=sam_masks,
        lulc_maps=lulc_maps,
        classes_to_correct=classes_to_correct,
        num_classes=num_classes,
        include_base_mask=include_base_mask,
    )


class SamLabelCorrector:
    """Orchestrates SAM-based label correction across a coreset of tiles.

    Loads the coreset CSV, initialises SAM, and processes tiles in order.
    For each tile, chunks are processed; SAM is run once per chunk (or
    loaded from cache) and the result is applied to every registered target.

    Args:
        config: :class:`SAMLabelCorrectionConfig` instance.

    Example::

        config = SAMLabelCorrectionConfig(
            coreset_csv="/data/coreset.csv",
            masks_dir="/data/masks",
            targets=[{"classes": [3, 5], "output_dir": "/data/out"}],
            sam_checkpoint="/models/sam_vit_b.pth",
            mbtiles_path="/data/tiles.mbtiles",
        )
        corrector = SamLabelCorrector(config)
        stats = corrector.run()
    """

    def __init__(self, config: SAMLabelCorrectionConfig) -> None:
        self._cfg = config
        self._cache = SAMSegmentCache(config.cache_dir)
        self._targets = self._parse_targets(config.targets)
        self._lulc_paths = [resolve_image_source(p) for p in config.lulc_paths]
        self._masks_dir = Path(config.masks_dir)
        self._mbtiles = resolve_image_source(config.mbtiles_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Run SAM correction over all tiles in the coreset CSV.

        Returns:
            Summary dict with keys ``n_tiles``, ``elapsed_s``, and per-target
            ``total_changed`` / ``total_target`` counters.
        """
        mask_generator = self._load_sam()

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
            stats = self._process_tile(tile_name, patch_rows, mask_generator)
            results.append(stats)

        elapsed = time.time() - t_start
        return {"n_tiles": len(tiles), "elapsed_s": round(elapsed, 1), "tiles": results}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_targets(raw: list) -> List[Tuple[FrozenSet[int], Path]]:
        return parse_correction_targets(raw)

    def _load_sam(self):
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "segment_anything is required for SAM label correction. "
                "Install it with: pip install segment-anything"
            ) from exc

        sam = sam_model_registry[self._cfg.sam_model_type](
            checkpoint=self._cfg.sam_checkpoint
        )
        sam.to(self._cfg.device)
        return SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=self._cfg.points_per_side,
            points_per_batch=self._cfg.points_per_batch,
            pred_iou_thresh=self._cfg.pred_iou_thresh,
            stability_score_thresh=self._cfg.stability_score_thresh,
            min_mask_region_area=self._cfg.min_mask_region_area,
        )

    def _chunk_cache_key(self, tile_name: str, window) -> str:
        return chunk_cache_key(tile_name, window)

    def _process_tile(
        self, tile_name: str, patch_rows: pd.DataFrame, mask_generator
    ) -> Dict:
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

        windows = self._enumerate_windows(src_path, row_min, row_max, col_min, col_max)

        tile_changed = [0] * len(self._targets)
        tile_target = [0] * len(self._targets)
        n_chunks = 0
        n_skipped = 0

        for chunk_results in self._iter_chunk_results(
            src_path, windows, chunk_targets, mask_generator
        ):
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

    def _enumerate_windows(
        self, src_path: Path, row_min: int, row_max: int, col_min: int, col_max: int
    ) -> List["Window"]:
        import rasterio
        from rasterio.windows import Window

        chunk = self._cfg.chunk_size
        with rasterio.open(src_path) as mask_src:
            mask_height, mask_width = mask_src.height, mask_src.width

        windows = []
        for r0 in range(row_min, row_max, chunk):
            for c0 in range(col_min, col_max, chunk):
                r1 = min(r0 + chunk, row_max, mask_height)
                c1 = min(c0 + chunk, col_max, mask_width)
                if r1 <= r0 or c1 <= c0:
                    continue
                windows.append(Window(c0, r0, c1 - c0, r1 - r0))
        return windows

    def _iter_chunk_results(
        self,
        src_path: Path,
        windows: List["Window"],
        chunk_targets: List[Tuple[FrozenSet[int], Path]],
        mask_generator,
    ):
        """Yield per-chunk ``(changed, target)`` results, one window at a time.

        When ``config.prefetch`` is enabled and there is more than one window,
        the next window's imagery/LULC read runs on a background thread while
        SAM processes the current window (I/O-GPU overlap; SAM itself still
        processes one window per call — see ``SAMLabelCorrectionConfig.prefetch``).
        """
        if not self._cfg.prefetch or len(windows) <= 1:
            for window in windows:
                inputs = self._read_chunk_inputs(src_path, window, chunk_targets)
                yield self._run_sam_and_correct(
                    src_path, inputs, chunk_targets, window, mask_generator
                )
            return

        with ThreadPoolExecutor(max_workers=1) as io_pool:
            next_future = io_pool.submit(
                self._read_chunk_inputs, src_path, windows[0], chunk_targets
            )
            for i, window in enumerate(windows):
                inputs = next_future.result()
                if i + 1 < len(windows):
                    next_future = io_pool.submit(
                        self._read_chunk_inputs, src_path, windows[i + 1], chunk_targets
                    )
                yield self._run_sam_and_correct(
                    src_path, inputs, chunk_targets, window, mask_generator
                )

    def _read_chunk_inputs(
        self,
        src_path: Path,
        window,
        chunk_targets: List[Tuple[FrozenSet[int], Path]],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, List[np.ndarray]]]:
        """Read and align one window's mask/imagery/LULC arrays.

        Opens its own dataset handles (safe to call from a background thread —
        no rasterio handle is shared with the caller). Returns ``None`` when the
        window has no target-class pixels, or the aligned read fails.
        """
        import rasterio

        from pytorch_segmentation_models_trainer.tools.mbtiles.alignment import (
            read_source_aligned_to_mask_window,
        )

        with rasterio.open(src_path) as mask_src:
            base_mask = mask_src.read(1, window=window)

            has_any = any(
                np.isin(base_mask, list(cls)).any() for cls, _ in chunk_targets
            )
            if not has_any:
                return None

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
                return None

        image_hwc = np.ascontiguousarray(image_chw.transpose(1, 2, 0))
        return base_mask, image_hwc, lulc_maps

    def _run_sam_and_correct(
        self,
        src_path: Path,
        inputs: Optional[Tuple[np.ndarray, np.ndarray, List[np.ndarray]]],
        chunk_targets: List[Tuple[FrozenSet[int], Path]],
        window,
        mask_generator,
    ) -> List[Tuple[int, int]]:
        import rasterio
        import torch

        if inputs is None:
            return [(0, 0)] * len(chunk_targets)
        base_mask, image_hwc, lulc_maps = inputs

        cache_key = self._chunk_cache_key(str(src_path), window)
        sam_masks = self._cache.get(cache_key)
        if sam_masks is None:
            with torch.no_grad():
                sam_masks = mask_generator.generate(image_hwc)
            self._cache.put(cache_key, sam_masks)

        results = []
        for classes_to_correct, corrected_path in chunk_targets:
            if not np.isin(base_mask, list(classes_to_correct)).any():
                results.append((0, 0))
                continue

            corrected = apply_sam_correction(
                base_mask=base_mask,
                sam_masks=sam_masks,
                lulc_maps=lulc_maps,
                classes_to_correct=classes_to_correct,
                num_classes=self._cfg.num_classes,
                include_base_mask=self._cfg.include_base_mask,
            )
            n_target = int(np.isin(base_mask, list(classes_to_correct)).sum())
            n_changed = int((corrected != base_mask).sum())

            with rasterio.open(corrected_path, "r+") as dst:
                dst.write(corrected[np.newaxis, :, :], window=window)

            results.append((n_changed, n_target))

        return results
