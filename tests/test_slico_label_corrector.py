# -*- coding: utf-8 -*-
"""Tests for pytorch_segmentation_models_trainer.tools.slico_correction.slico_label_corrector."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pytorch_segmentation_models_trainer.tools.slico_correction.slico_label_corrector import (
    SLICOLabelCorrectionConfig,
    SlicoLabelCorrector,
)

NUM_CLASSES = 6


def _make_config(tmp_path, **overrides):
    defaults = dict(
        coreset_csv=str(tmp_path / "coreset.csv"),
        masks_dir=str(tmp_path / "masks"),
        targets=[{"classes": [3], "output_dir": str(tmp_path / "out")}],
        mbtiles_path=str(tmp_path / "tiles.mbtiles"),
        cache_dir="",
    )
    defaults.update(overrides)
    return SLICOLabelCorrectionConfig(**defaults)


def _make_window(row_off=0, col_off=0, height=4, width=4):
    w = MagicMock()
    w.row_off = row_off
    w.col_off = col_off
    w.height = height
    w.width = width
    return w


# ---------------------------------------------------------------------------
# SLICOLabelCorrectionConfig defaults
# ---------------------------------------------------------------------------


class TestSLICOLabelCorrectionConfig:
    def test_required_fields(self):
        cfg = SLICOLabelCorrectionConfig(
            coreset_csv="/data/coreset.csv",
            masks_dir="/data/masks",
            targets=[{"classes": [3, 5], "output_dir": "/data/out"}],
            mbtiles_path="/data/tiles.mbtiles",
        )
        assert cfg.coreset_csv == "/data/coreset.csv"
        assert cfg.masks_dir == "/data/masks"

    def test_default_values(self):
        cfg = SLICOLabelCorrectionConfig(
            coreset_csv="a", masks_dir="b", targets=[], mbtiles_path="d"
        )
        assert cfg.lulc_paths == []
        assert cfg.include_base_mask is True
        assert cfg.num_classes == 6
        assert cfg.nodata_val == 255
        assert cfg.chunk_size == 1024
        assert cfg.n_segments == 1000
        assert cfg.match_sam_cache_dir == ""
        assert cfg.cache_dir == ""
        assert cfg.start_idx == 0
        assert cfg.end_idx == 999999


# ---------------------------------------------------------------------------
# SlicoLabelCorrector — targets / cache keys
# ---------------------------------------------------------------------------


class TestSlicoLabelCorrectorParseTargets:
    def test_parse_single_target(self, tmp_path):
        cfg = _make_config(tmp_path)
        corrector = SlicoLabelCorrector(cfg)
        classes, out_dir = corrector._targets[0]
        assert classes == frozenset([3])
        assert out_dir == tmp_path / "out"


class TestSlicoLabelCorrectorImageSourceResolution:
    def test_mbtiles_path_string_stays_a_path(self, tmp_path):
        cfg = _make_config(tmp_path, mbtiles_path="/data/tiles.mbtiles")
        corrector = SlicoLabelCorrector(cfg)
        assert corrector._mbtiles == Path("/data/tiles.mbtiles")

    def test_mbtiles_path_directory_spec_resolves_to_tiled_source(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin

        from pytorch_segmentation_models_trainer.tools.mbtiles.image_source import (
            TiledImageSource,
        )

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()
        profile = {
            "driver": "GTiff",
            "height": 4,
            "width": 4,
            "count": 1,
            "dtype": "uint8",
            "crs": "EPSG:3857",
            "transform": from_origin(0, 4, 1, 1),
        }
        with rasterio.open(tiles_dir / "a.tif", "w", **profile) as dst:
            dst.write(np.zeros((4, 4), dtype=np.uint8), 1)

        cfg = _make_config(tmp_path, mbtiles_path={"directory": str(tiles_dir)})
        corrector = SlicoLabelCorrector(cfg)
        assert isinstance(corrector._mbtiles, TiledImageSource)


class TestSlicoLabelCorrectorChunkCacheKey:
    def test_key_format_includes_n_segments(self, tmp_path):
        cfg = _make_config(tmp_path)
        corrector = SlicoLabelCorrector(cfg)
        window = _make_window(row_off=10, col_off=20, height=64, width=128)
        key = corrector._chunk_cache_key("some/path/tile_001.tif", window, 500)
        assert key == "tile_001_r10_c20_h64_w128_n500"


class TestSlicoLabelCorrectorEstimateNChunks:
    def test_single_chunk(self, tmp_path):
        cfg = _make_config(tmp_path, chunk_size=1024)
        corrector = SlicoLabelCorrector(cfg)
        rows = pd.DataFrame({"row_off": [0], "col_off": [0], "patch_size": [256]})
        assert corrector._estimate_n_chunks(rows) == 1

    def test_multiple_chunks_both_axes(self, tmp_path):
        cfg = _make_config(tmp_path, chunk_size=1024)
        corrector = SlicoLabelCorrector(cfg)
        # Spans 2048x2048 total extent -> 2x2 = 4 chunks at chunk_size=1024.
        rows = pd.DataFrame(
            {
                "row_off": [0, 1792],
                "col_off": [0, 1792],
                "patch_size": [256, 256],
            }
        )
        assert corrector._estimate_n_chunks(rows) == 4


class TestSlicoLabelCorrectorResolveNSegments:
    def test_density_scales_with_chunk_area(self, tmp_path):
        """n_segments=1000 per 256x256 patch → 4x area gives 4x segments."""
        cfg = _make_config(tmp_path, n_segments=1000)
        corrector = SlicoLabelCorrector(cfg)
        window_1x = _make_window(height=256, width=256)
        window_4x = _make_window(height=512, width=512)
        n1 = corrector._resolve_n_segments("tile.tif", window_1x)
        n4 = corrector._resolve_n_segments("tile.tif", window_4x)
        assert n1 == 1000
        assert n4 == 4000

    def test_minimum_one_segment(self, tmp_path):
        cfg = _make_config(tmp_path, n_segments=1000)
        corrector = SlicoLabelCorrector(cfg)
        window = _make_window(height=1, width=1)
        assert corrector._resolve_n_segments("tile.tif", window) >= 1

    def test_match_sam_cache_dir_uses_sam_mask_count(self, tmp_path):
        cfg = _make_config(
            tmp_path, match_sam_cache_dir=str(tmp_path / "sam_cache"), n_segments=1000
        )
        corrector = SlicoLabelCorrector(cfg)
        window = _make_window(row_off=0, col_off=0, height=256, width=256)
        sam_masks = [
            {
                "segmentation": np.zeros((4, 4), dtype=bool),
                "predicted_iou": 0.9,
                "area": 1,
            }
        ] * 42
        corrector._sam_cache = MagicMock()
        corrector._sam_cache.get.return_value = sam_masks
        assert corrector._resolve_n_segments("tile.tif", window) == 42

    def test_match_sam_cache_dir_falls_back_when_miss(self, tmp_path):
        cfg = _make_config(
            tmp_path, match_sam_cache_dir=str(tmp_path / "sam_cache"), n_segments=1000
        )
        corrector = SlicoLabelCorrector(cfg)
        window = _make_window(height=256, width=256)
        corrector._sam_cache = MagicMock()
        corrector._sam_cache.get.return_value = None
        assert corrector._resolve_n_segments("tile.tif", window) == 1000

    def test_no_match_sam_cache_dir_configured(self, tmp_path):
        cfg = _make_config(tmp_path)
        corrector = SlicoLabelCorrector(cfg)
        assert corrector._sam_cache is None


# ---------------------------------------------------------------------------
# SlicoLabelCorrector.run
# ---------------------------------------------------------------------------


class TestSlicoLabelCorrectorRun:
    def _setup_csv(self, tmp_path):
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        (masks_dir / "tile.tif").touch()
        df = pd.DataFrame(
            {
                "mask_path": ["tile.tif"],
                "row_off": [0],
                "col_off": [0],
                "patch_size": [4],
            }
        )
        csv_path = tmp_path / "coreset.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_run_returns_summary(self, tmp_path):
        csv_path = self._setup_csv(tmp_path)
        cfg = _make_config(tmp_path, coreset_csv=str(csv_path))
        corrector = SlicoLabelCorrector(cfg)

        tile_stats = {
            "tile": "tile.tif",
            "n_chunks": 1,
            "n_skipped": 0,
            "per_target": [],
        }
        with patch.object(corrector, "_process_tile", return_value=tile_stats):
            result = corrector.run()

        assert result["n_tiles"] == 1
        assert "elapsed_s" in result
        assert result["tiles"] == [tile_stats]

    def test_run_threaded_preserves_csv_tile_order(self, tmp_path):
        """n_workers > 1 dispatches via ThreadPoolExecutor but the returned
        `tiles` list still matches the coreset CSV's (sorted) tile order,
        regardless of completion order."""
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        for name in ["a.tif", "b.tif", "c.tif"]:
            (masks_dir / name).touch()
        df = pd.DataFrame(
            {
                "mask_path": ["a.tif", "b.tif", "c.tif"],
                "row_off": [0, 0, 0],
                "col_off": [0, 0, 0],
                "patch_size": [4, 4, 4],
            }
        )
        csv_path = tmp_path / "coreset.csv"
        df.to_csv(csv_path, index=False)

        cfg = _make_config(tmp_path, coreset_csv=str(csv_path), n_workers=4)
        corrector = SlicoLabelCorrector(cfg)

        def fake_process(tile_name, patch_rows):
            return {"tile": tile_name, "n_chunks": 1, "n_skipped": 0, "per_target": []}

        with patch.object(corrector, "_process_tile", side_effect=fake_process):
            result = corrector.run()

        assert [t["tile"] for t in result["tiles"]] == ["a.tif", "b.tif", "c.tif"]
        assert result["n_tiles"] == 3

    def test_run_threaded_single_tile_falls_back_to_sequential(self, tmp_path):
        """n_workers > 1 with only one tile skips the thread pool entirely."""
        csv_path = self._setup_csv(tmp_path)
        cfg = _make_config(tmp_path, coreset_csv=str(csv_path), n_workers=4)
        corrector = SlicoLabelCorrector(cfg)

        tile_stats = {
            "tile": "tile.tif",
            "n_chunks": 1,
            "n_skipped": 0,
            "per_target": [],
        }
        with patch.object(
            corrector, "_process_tile", return_value=tile_stats
        ) as mock_pt:
            result = corrector.run()

        mock_pt.assert_called_once()
        assert result["tiles"] == [tile_stats]

    def test_run_respects_start_end_idx(self, tmp_path):
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        for name in ["a.tif", "b.tif", "c.tif"]:
            (masks_dir / name).touch()
        df = pd.DataFrame(
            {
                "mask_path": ["a.tif", "b.tif", "c.tif"],
                "row_off": [0, 0, 0],
                "col_off": [0, 0, 0],
                "patch_size": [4, 4, 4],
            }
        )
        csv_path = tmp_path / "coreset.csv"
        df.to_csv(csv_path, index=False)

        cfg = _make_config(tmp_path, coreset_csv=str(csv_path), start_idx=1, end_idx=2)
        corrector = SlicoLabelCorrector(cfg)

        processed = []

        def fake_process(tile_name, patch_rows):
            processed.append(tile_name)
            return {"tile": tile_name, "n_chunks": 0, "n_skipped": 0, "per_target": []}

        with patch.object(corrector, "_process_tile", side_effect=fake_process):
            corrector.run()

        assert processed == ["b.tif"]

    def test_run_adds_patch_size_column_when_missing(self, tmp_path):
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        (masks_dir / "tile.tif").touch()
        df = pd.DataFrame({"mask_path": ["tile.tif"], "row_off": [0], "col_off": [0]})
        csv_path = tmp_path / "coreset.csv"
        df.to_csv(csv_path, index=False)

        cfg = _make_config(tmp_path, coreset_csv=str(csv_path))
        corrector = SlicoLabelCorrector(cfg)

        tile_stats = {
            "tile": "tile.tif",
            "n_chunks": 0,
            "n_skipped": 0,
            "per_target": [],
        }
        with patch.object(
            corrector, "_process_tile", return_value=tile_stats
        ) as mock_pt:
            corrector.run()

        patch_rows_arg = mock_pt.call_args[0][1]
        assert "patch_size" in patch_rows_arg.columns


# ---------------------------------------------------------------------------
# SlicoLabelCorrector._process_tile / _process_chunk
# ---------------------------------------------------------------------------


def _mock_rasterio(open_ctx=None):
    mock_rio = MagicMock()
    mock_windows_mod = MagicMock()
    mock_rio.windows = mock_windows_mod
    if open_ctx is not None:
        mock_rio.open.return_value = open_ctx
    mods = {"rasterio": mock_rio, "rasterio.windows": mock_windows_mod}
    return mods, mock_rio


def _mock_chunk_imports(
    read_return=None, read_side_effect=None, open_ctx=None, slic_return=None
):
    mock_rio = MagicMock()
    if open_ctx is not None:
        mock_rio.open.return_value = open_ctx
    mock_alignment = MagicMock()
    if read_side_effect is not None:
        mock_alignment.read_source_aligned_to_mask_window.side_effect = read_side_effect
    elif read_return is not None:
        mock_alignment.read_source_aligned_to_mask_window.return_value = read_return
    mock_skimage_seg = MagicMock()
    if slic_return is not None:
        mock_skimage_seg.slic.return_value = slic_return
    mods = {
        "rasterio": mock_rio,
        "rasterio.windows": MagicMock(),
        "skimage.segmentation": mock_skimage_seg,
        "pytorch_segmentation_models_trainer.tools.mbtiles.alignment": mock_alignment,
    }
    return mods, mock_rio, mock_alignment, mock_skimage_seg


class TestSlicoLabelCorrectorProcessTile:
    def _make_mock_src(self, height=4, width=4):
        src = MagicMock()
        src.height = height
        src.width = width
        src.name = "tile.tif"
        return src

    def _make_open_ctx(self, mock_src):
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_src)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    def test_process_tile_copies_source_to_targets(self, tmp_path):
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        (masks_dir / "tile.tif").write_bytes(b"\x00" * 16)
        (tmp_path / "out").mkdir(parents=True)

        cfg = _make_config(tmp_path, masks_dir=str(masks_dir))
        corrector = SlicoLabelCorrector(cfg)

        patch_rows = pd.DataFrame({"row_off": [0], "col_off": [0], "patch_size": [4]})

        mock_src = self._make_mock_src()
        open_ctx = self._make_open_ctx(mock_src)
        sys_mods, _ = _mock_rasterio(open_ctx=open_ctx)

        with patch.dict("sys.modules", sys_mods):
            with patch.object(corrector, "_process_chunk", return_value=[(2, 16)]):
                result = corrector._process_tile("tile.tif", patch_rows)

        assert result["tile"] == "tile.tif"
        assert result["n_chunks"] == 1
        assert result["per_target"][0]["n_changed"] == 2

    def test_process_tile_degenerate_chunk_skipped(self, tmp_path):
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        (masks_dir / "tile.tif").write_bytes(b"\x00" * 16)
        (tmp_path / "out").mkdir(parents=True)

        cfg = _make_config(tmp_path, masks_dir=str(masks_dir))
        corrector = SlicoLabelCorrector(cfg)

        patch_rows = pd.DataFrame({"row_off": [0], "col_off": [0], "patch_size": [4]})
        mock_src = self._make_mock_src(height=0, width=4)
        open_ctx = self._make_open_ctx(mock_src)
        sys_mods, _ = _mock_rasterio(open_ctx=open_ctx)

        with patch.dict("sys.modules", sys_mods):
            with patch.object(
                corrector, "_process_chunk", return_value=[(0, 0)]
            ) as mock_pc:
                result = corrector._process_tile("tile.tif", patch_rows)

        mock_pc.assert_not_called()
        assert result["n_chunks"] == 0

    def test_process_tile_skipped_chunk_counted(self, tmp_path):
        """Chunk with all-zero target counts is counted as skipped."""
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        (masks_dir / "tile.tif").write_bytes(b"\x00" * 16)
        (tmp_path / "out").mkdir(parents=True)

        cfg = _make_config(tmp_path, masks_dir=str(masks_dir))
        corrector = SlicoLabelCorrector(cfg)

        patch_rows = pd.DataFrame({"row_off": [0], "col_off": [0], "patch_size": [4]})
        mock_src = self._make_mock_src()
        open_ctx = self._make_open_ctx(mock_src)
        sys_mods, _ = _mock_rasterio(open_ctx=open_ctx)

        with patch.dict("sys.modules", sys_mods):
            with patch.object(corrector, "_process_chunk", return_value=[(0, 0)]):
                result = corrector._process_tile("tile.tif", patch_rows)

        assert result["n_skipped"] == 1


class TestSlicoLabelCorrectorProcessChunk:
    def _corrector(self, tmp_path):
        return SlicoLabelCorrector(_make_config(tmp_path))

    def _mock_src(self, array):
        src = MagicMock()
        src.read.return_value = array
        src.name = "tile.tif"
        return src

    def test_no_target_pixels_returns_zeros(self, tmp_path):
        corrector = self._corrector(tmp_path)
        bags = np.full((4, 4), 0, dtype=np.uint8)  # class 0, not in targets
        mask_src = self._mock_src(bags)
        chunk_targets = [(frozenset([3]), tmp_path / "out" / "tile.tif")]
        sys_mods, _, _, _ = _mock_chunk_imports()

        with patch.dict("sys.modules", sys_mods):
            result = corrector._process_chunk(mask_src, chunk_targets, _make_window())

        assert result == [(0, 0)]

    def test_image_read_exception_returns_zeros(self, tmp_path):
        corrector = self._corrector(tmp_path)
        bags = np.full((4, 4), 3, dtype=np.uint8)
        mask_src = self._mock_src(bags)
        chunk_targets = [(frozenset([3]), tmp_path / "out" / "tile.tif")]
        sys_mods, _, _, _ = _mock_chunk_imports(read_side_effect=RuntimeError("IO"))

        with patch.dict("sys.modules", sys_mods):
            result = corrector._process_chunk(mask_src, chunk_targets, _make_window())

        assert result == [(0, 0)]

    def test_cache_hit_skips_slic(self, tmp_path):
        corrector = self._corrector(tmp_path)
        bags = np.full((4, 4), 3, dtype=np.uint8)
        mask_src = self._mock_src(bags)
        chunk_targets = [(frozenset([3]), tmp_path / "out" / "tile.tif")]

        corrector._cache = MagicMock()
        corrector._cache.get.return_value = np.zeros((4, 4), dtype=np.int32)

        image_chw = np.zeros((3, 4, 4), dtype=np.uint8)
        sys_mods, _, _, mock_skimage_seg = _mock_chunk_imports(read_return=image_chw)

        with patch.dict("sys.modules", sys_mods):
            corrector._process_chunk(mask_src, chunk_targets, _make_window())

        mock_skimage_seg.slic.assert_not_called()
        corrector._cache.put.assert_not_called()

    def test_cache_miss_runs_slic_and_stores(self, tmp_path):
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True)
        corrected_file = out_dir / "tile.tif"
        corrected_file.write_bytes(b"\x00" * 16)

        corrector = self._corrector(tmp_path)
        bags = np.full((4, 4), 3, dtype=np.uint8)
        mask_src = self._mock_src(bags)
        mask_src.name = str(tmp_path / "masks" / "tile.tif")
        chunk_targets = [(frozenset([3]), corrected_file)]

        corrector._cache = MagicMock()
        corrector._cache.get.return_value = None  # cache miss

        mock_dst = MagicMock()
        dst_ctx = MagicMock()
        dst_ctx.__enter__ = MagicMock(return_value=mock_dst)
        dst_ctx.__exit__ = MagicMock(return_value=False)

        image_chw = np.zeros((3, 4, 4), dtype=np.uint8)
        label_map = np.zeros((4, 4), dtype=np.int64)
        sys_mods, mock_rio, _, mock_skimage_seg = _mock_chunk_imports(
            read_return=image_chw, open_ctx=dst_ctx, slic_return=label_map
        )

        with patch.dict("sys.modules", sys_mods):
            corrector._process_chunk(mask_src, chunk_targets, _make_window())

        mock_skimage_seg.slic.assert_called_once()
        corrector._cache.put.assert_called_once()

    def test_chunk_target_without_target_pixels_skipped(self, tmp_path):
        corrector = self._corrector(tmp_path)
        bags = np.full((4, 4), 3, dtype=np.uint8)
        mask_src = self._mock_src(bags)
        chunk_targets = [
            (frozenset([3]), tmp_path / "out" / "tile.tif"),
            (frozenset([5]), tmp_path / "out" / "tile.tif"),
        ]

        corrector._cache = MagicMock()
        corrector._cache.get.return_value = np.zeros((4, 4), dtype=np.int32)

        image_chw = np.zeros((3, 4, 4), dtype=np.uint8)
        sys_mods, _, _, _ = _mock_chunk_imports(read_return=image_chw)

        with patch.dict("sys.modules", sys_mods):
            result = corrector._process_chunk(mask_src, chunk_targets, _make_window())

        assert result[1] == (0, 0)  # second target (class 5, absent) skipped
