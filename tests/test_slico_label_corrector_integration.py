# -*- coding: utf-8 -*-
"""Real (unmocked) end-to-end test: sequential vs. threaded SLICO correction
must produce byte-identical output, since tile-level parallelism only changes
*when* each tile is processed, never *how*.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import skimage  # noqa: F401

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

pytestmark = pytest.mark.skipif(
    not (HAS_RASTERIO and HAS_SKIMAGE), reason="rasterio and scikit-image required"
)

from pytorch_segmentation_models_trainer.tools.slico_correction.slico_label_corrector import (
    SLICOLabelCorrectionConfig,
    SlicoLabelCorrector,
)


def _write_mask(path: Path, seed: int, size=16):
    rng = np.random.default_rng(seed)
    mask = rng.choice([3, 4, 5], size=(size, size)).astype(np.uint8)
    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": from_origin(0, size, 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask, 1)


def _write_imagery(path: Path, size=16):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 255, size=(3, size, size), dtype=np.uint8)
    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": from_origin(0, size, 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _build_dataset(tmp_path: Path, n_tiles: int = 6, size: int = 16):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    rows = []
    for i in range(n_tiles):
        name = f"tile_{i}.tif"
        _write_mask(masks_dir / name, seed=i, size=size)
        rows.append({"mask_path": name, "row_off": 0, "col_off": 0, "patch_size": size})
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "coreset.csv"
    df.to_csv(csv_path, index=False)

    imagery_path = tmp_path / "imagery.tif"
    _write_imagery(imagery_path, size=size)

    return csv_path, masks_dir, imagery_path


def _run(tmp_path, csv_path, masks_dir, imagery_path, out_dir, n_workers):
    cfg = SLICOLabelCorrectionConfig(
        coreset_csv=str(csv_path),
        masks_dir=str(masks_dir),
        targets=[{"classes": [3, 5], "output_dir": str(out_dir)}],
        mbtiles_path=str(imagery_path),
        n_segments=8,
        chunk_size=1024,
        n_workers=n_workers,
        cache_dir="",
    )
    return SlicoLabelCorrector(cfg).run()


class TestSlicoThreadedMatchesSequential:
    def test_output_masks_are_byte_identical(self, tmp_path):
        csv_path, masks_dir, imagery_path = _build_dataset(tmp_path, n_tiles=6)

        out_sequential = tmp_path / "out_sequential"
        out_threaded = tmp_path / "out_threaded"

        stats_seq = _run(
            tmp_path, csv_path, masks_dir, imagery_path, out_sequential, n_workers=1
        )
        stats_thr = _run(
            tmp_path, csv_path, masks_dir, imagery_path, out_threaded, n_workers=4
        )

        assert stats_seq["n_tiles"] == stats_thr["n_tiles"] == 6

        for i in range(6):
            name = f"tile_{i}.tif"
            with (
                rasterio.open(out_sequential / name) as a,
                rasterio.open(out_threaded / name) as b,
            ):
                np.testing.assert_array_equal(a.read(), b.read())

    def test_per_tile_stats_match_regardless_of_worker_count(self, tmp_path):
        csv_path, masks_dir, imagery_path = _build_dataset(tmp_path, n_tiles=5)

        out_sequential = tmp_path / "out_sequential"
        out_threaded = tmp_path / "out_threaded"

        stats_seq = _run(
            tmp_path, csv_path, masks_dir, imagery_path, out_sequential, n_workers=1
        )
        stats_thr = _run(
            tmp_path, csv_path, masks_dir, imagery_path, out_threaded, n_workers=3
        )

        def _strip_output_dir(per_target):
            return [
                {k: v for k, v in pt.items() if k != "output_dir"} for pt in per_target
            ]

        by_tile_seq = {
            t["tile"]: _strip_output_dir(t["per_target"]) for t in stats_seq["tiles"]
        }
        by_tile_thr = {
            t["tile"]: _strip_output_dir(t["per_target"]) for t in stats_thr["tiles"]
        }
        assert by_tile_seq == by_tile_thr

        # Tile order in the summary matches the coreset CSV's order either way.
        assert [t["tile"] for t in stats_seq["tiles"]] == [
            t["tile"] for t in stats_thr["tiles"]
        ]
