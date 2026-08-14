# -*- coding: utf-8 -*-
"""SAM-based label correction for noisy segmentation masks.

Pre-processing tool that corrects noisy GeoTIFF masks using SAM AMG
(Automatic Mask Generation) + majority vote within SAM segments across
the original topographic vector source and optional LULC auxiliary maps.
"""

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


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
        mbtiles_path: Path to MBTiles file used as RGB source imagery for SAM.
        lulc_paths: Paths to auxiliary LULC rasters/VRTs included in the
            majority vote alongside the topographic vector source.
        include_bags: Whether to include the original topographic vector source
            mask in the majority vote (default True).
        sam_model_type: SAM model registry key (default ``"vit_b"``).
        device: Torch device string (default ``"cuda:0"``).
        num_classes: Number of valid semantic classes (default 6).
        nodata_val: Pixel value treated as nodata/invalid (default 255).
        chunk_size: Tile processing chunk size in pixels (default 1024).
        points_per_side: SAM AMG grid density (default 32).
        pred_iou_thresh: SAM AMG predicted-IoU threshold (default 0.80).
        stability_score_thresh: SAM AMG stability-score threshold (default 0.90).
        min_mask_region_area: Minimum SAM segment area in pixels (default 200).
        cache_dir: Directory for NPZ segment cache. Empty string disables cache.
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
    mbtiles_path: str
    lulc_paths: list = field(default_factory=list)
    include_bags: bool = True
    sam_model_type: str = "vit_b"
    device: str = "cuda:0"
    num_classes: int = 6
    nodata_val: int = 255
    chunk_size: int = 1024
    points_per_side: int = 32
    pred_iou_thresh: float = 0.80
    stability_score_thresh: float = 0.90
    min_mask_region_area: int = 200
    cache_dir: str = ""
    start_idx: int = 0
    end_idx: int = 999999


def apply_sam_correction(
    bags_raw: np.ndarray,
    sam_masks: List[Dict],
    lulc_maps: List[np.ndarray],
    classes_to_correct: FrozenSet[int],
    num_classes: int = 6,
    include_bags: bool = True,
) -> np.ndarray:
    """Apply SAM-based majority-vote correction to a mask array.

    For each SAM segment that contains at least one pixel from
    ``classes_to_correct``, the winning class is determined by majority vote
    across all sources (optionally the original mask + each LULC map). The
    winner is written to every pixel in the segment. Non-target class pixels
    are always restored from the original mask after processing.

    SAM masks are processed in ascending (predicted_iou, area) order so that
    higher-confidence, larger segments overwrite smaller/less-confident ones in
    overlapping regions.

    Args:
        bags_raw: Original mask array (H, W) uint8 — topographic vector source.
        sam_masks: List of SAM segment dicts with keys ``segmentation``
            (bool H×W), ``predicted_iou`` (float), ``area`` (int).
        lulc_maps: List of auxiliary class arrays (H, W) uint8 used as extra
            votes alongside ``bags_raw``.
        classes_to_correct: Set of class indices eligible for correction.
        num_classes: Number of valid class indices (values >= num_classes ignored).
        include_bags: If True, ``bags_raw`` is counted as one vote source.

    Returns:
        Corrected mask array (H, W) uint8, same shape as ``bags_raw``.
    """
    backup = bags_raw.copy()
    corrected = bags_raw.copy()

    # Sort ascending so higher-confidence/larger masks overwrite last.
    sorted_masks = sorted(sam_masks, key=lambda m: (m["predicted_iou"], m["area"]))

    all_sources: List[np.ndarray] = []
    if include_bags:
        all_sources.append(bags_raw)
    all_sources.extend(lulc_maps)

    for m in sorted_masks:
        seg: np.ndarray = m["segmentation"]
        if not seg.any():
            continue
        if not np.isin(bags_raw[seg], list(classes_to_correct)).any():
            continue

        vote_counts = np.zeros(num_classes, dtype=np.int64)
        for src in all_sources:
            vals = src[seg]
            valid = vals[vals < num_classes]
            np.add.at(vote_counts, valid.astype(int), 1)

        if vote_counts.sum() == 0:
            continue

        winner = int(vote_counts.argmax())
        corrected[seg] = winner

    # Restore pixels that should not have been touched.
    non_target = ~np.isin(backup, list(classes_to_correct))
    corrected[non_target] = backup[non_target]
    return corrected


class SAMSegmentCache:
    """NPZ-based cache for SAM segment arrays.

    Stores SAM segment masks keyed by a string identifier. Each entry
    contains per-segment boolean arrays (``seg_N``), predicted IoU scalars
    (``iou_N``), and area scalars (``area_N``).

    Args:
        cache_dir: Directory for NPZ files. Pass ``""`` or ``None`` to disable.

    Example::

        cache = SAMSegmentCache(cache_dir="/data/sam_cache")
        masks = cache.get("tile_r0_c0_h1024_w1024")
        if masks is None:
            masks = mask_generator.generate(image)
            cache.put("tile_r0_c0_h1024_w1024", masks)
    """

    def __init__(self, cache_dir: Optional[str]) -> None:
        self._enabled = bool(cache_dir)
        self._root = Path(cache_dir) if cache_dir else None

    def get(self, key: str) -> Optional[List[Dict]]:
        """Load cached SAM masks for ``key``, or None if absent/invalid.

        Returns None when:
        - cache is disabled
        - the NPZ file does not exist
        - the file was written with an old format lacking ``iou_N`` arrays
        """
        if not self._enabled:
            return None
        path = self._root / f"{key}.npz"
        if not path.exists():
            return None

        data = np.load(path, allow_pickle=False)
        n = int(data["n_masks"])
        if n > 0 and "iou_0" not in data:
            return None

        masks = []
        for i in range(n):
            masks.append(
                {
                    "segmentation": data[f"seg_{i}"].astype(bool),
                    "predicted_iou": float(data[f"iou_{i}"]),
                    "area": int(data[f"area_{i}"]),
                }
            )
        return masks

    def put(self, key: str, masks: List[Dict]) -> None:
        """Persist SAM masks to disk under ``key``.

        No-op when cache is disabled.

        Args:
            key: Cache key (should be unique per chunk/patch).
            masks: List of SAM mask dicts (``segmentation``, ``predicted_iou``, ``area``).
        """
        if not self._enabled:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{key}.npz"
        arrays: Dict[str, np.ndarray] = {"n_masks": np.array(len(masks))}
        for i, m in enumerate(masks):
            arrays[f"seg_{i}"] = m["segmentation"].astype(np.uint8)
            arrays[f"iou_{i}"] = np.array(m["predicted_iou"])
            arrays[f"area_{i}"] = np.array(m["area"])
        np.savez_compressed(path, **arrays)


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
        self._lulc_paths = [Path(p) for p in config.lulc_paths]
        self._masks_dir = Path(config.masks_dir)
        self._mbtiles = Path(config.mbtiles_path)

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
        parsed = []
        for t in raw:
            classes = frozenset(int(c) for c in t["classes"])
            out_dir = Path(t["output_dir"])
            parsed.append((classes, out_dir))
        return parsed

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
            pred_iou_thresh=self._cfg.pred_iou_thresh,
            stability_score_thresh=self._cfg.stability_score_thresh,
            min_mask_region_area=self._cfg.min_mask_region_area,
        )

    def _chunk_cache_key(self, tile_name: str, window) -> str:
        stem = Path(tile_name).stem
        return (
            f"{stem}_r{window.row_off}_c{window.col_off}"
            f"_h{window.height}_w{window.width}"
        )

    def _process_tile(
        self, tile_name: str, patch_rows: pd.DataFrame, mask_generator
    ) -> Dict:
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
                    chunk_results = self._process_chunk(
                        mask_src, chunk_targets, window, mask_generator
                    )
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
        mask_generator,
    ) -> List[Tuple[int, int]]:
        import rasterio
        import torch

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

        cache_key = self._chunk_cache_key(mask_src.name, window)
        sam_masks = self._cache.get(cache_key)
        if sam_masks is None:
            with torch.no_grad():
                sam_masks = mask_generator.generate(image_hwc)
            self._cache.put(cache_key, sam_masks)

        results = []
        for classes_to_correct, corrected_path in chunk_targets:
            if not np.isin(bags_raw, list(classes_to_correct)).any():
                results.append((0, 0))
                continue

            corrected = apply_sam_correction(
                bags_raw=bags_raw,
                sam_masks=sam_masks,
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
