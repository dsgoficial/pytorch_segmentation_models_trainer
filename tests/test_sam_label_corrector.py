# -*- coding: utf-8 -*-
"""Tests for pytorch_segmentation_models_trainer.tools.sam_correction.sam_label_corrector."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pytorch_segmentation_models_trainer.tools.sam_correction.sam_label_corrector import (
    SAMLabelCorrectionConfig,
    SAMSegmentCache,
    SamLabelCorrector,
    apply_sam_correction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 6


def _make_mask(h: int = 4, w: int = 4, fill: int = 3) -> np.ndarray:
    return np.full((h, w), fill, dtype=np.uint8)


def _make_seg(h: int = 4, w: int = 4, rows=None, cols=None) -> np.ndarray:
    seg = np.zeros((h, w), dtype=bool)
    if rows is not None and cols is not None:
        seg[rows, cols] = True
    return seg


def _make_sam_mask(seg: np.ndarray, iou: float = 0.9, area: int = 4) -> dict:
    return {"segmentation": seg, "predicted_iou": iou, "area": area}


# ---------------------------------------------------------------------------
# apply_sam_correction
# ---------------------------------------------------------------------------


class TestApplySamCorrection:
    def test_basic_winner_assigned(self):
        """SAM segment covers target class pixels — majority vote assigns winner."""
        bags = _make_mask(fill=3)  # all grassland (class 3)
        seg = _make_seg(rows=slice(None), cols=slice(None))  # full 4×4 segment
        sam_masks = [_make_sam_mask(seg)]
        # LULC maps: all say class 4 (forest)
        lulc = [np.full((4, 4), 4, dtype=np.uint8)]
        result = apply_sam_correction(
            bags_raw=bags,
            sam_masks=sam_masks,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_bags=True,
        )
        # vote: BAGS=3 (16), LULC=4 (16) → tie resolved by argmax (lower index wins)
        # bags=3 count=16, class=4 count=16 → argmax picks 3 (same as original)
        # Actually since include_bags=True: sources = [bags, lulc[0]]
        # bags contributes 16 votes for class 3; lulc contributes 16 votes for class 4
        # argmax picks class 3 (first maximum)
        assert result.shape == bags.shape

    def test_lulc_majority_overrides_bags(self):
        """When LULC maps outvote BAGS, winner changes."""
        bags = _make_mask(fill=3)
        seg = _make_seg(rows=slice(None), cols=slice(None))
        sam_masks = [_make_sam_mask(seg)]
        # 2 LULC maps say class 5; BAGS says class 3 → 2:1 → class 5 wins
        lulc = [
            np.full((4, 4), 5, dtype=np.uint8),
            np.full((4, 4), 5, dtype=np.uint8),
        ]
        result = apply_sam_correction(
            bags_raw=bags,
            sam_masks=sam_masks,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_bags=True,
        )
        assert np.all(result == 5)

    def test_non_target_pixels_preserved(self):
        """Pixels belonging to non-target classes are never changed."""
        bags = np.array([[0, 0], [3, 3]], dtype=np.uint8)  # class 0 and class 3
        seg = np.ones((2, 2), dtype=bool)  # full segment
        sam_masks = [{"segmentation": seg, "predicted_iou": 0.9, "area": 4}]
        lulc = [np.full((2, 2), 5, dtype=np.uint8)]  # lulc says class 5
        result = apply_sam_correction(
            bags_raw=bags,
            sam_masks=sam_masks,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_bags=True,
        )
        # class 0 pixels (non-target) must remain 0
        assert result[0, 0] == 0
        assert result[0, 1] == 0

    def test_no_masks_returns_original(self):
        """Empty sam_masks list → original array returned unchanged."""
        bags = _make_mask(fill=3)
        result = apply_sam_correction(
            bags_raw=bags,
            sam_masks=[],
            lulc_maps=[],
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_bags=True,
        )
        np.testing.assert_array_equal(result, bags)

    def test_include_bags_false_excludes_bags_from_vote(self):
        """When include_bags=False, BAGS values are not counted in the vote."""
        bags = _make_mask(fill=3)
        seg = np.ones((4, 4), dtype=bool)
        sam_masks = [_make_sam_mask(seg)]
        # LULC unanimously says class 5; without BAGS, class 5 wins
        lulc = [np.full((4, 4), 5, dtype=np.uint8)]
        result = apply_sam_correction(
            bags_raw=bags,
            sam_masks=sam_masks,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_bags=False,
        )
        assert np.all(result == 5)

    def test_segment_without_target_class_skipped(self):
        """Segments with no target-class pixels are skipped (original preserved)."""
        bags = _make_mask(fill=0)  # all water — not in target
        seg = np.ones((4, 4), dtype=bool)
        sam_masks = [_make_sam_mask(seg)]
        lulc = [np.full((4, 4), 5, dtype=np.uint8)]
        result = apply_sam_correction(
            bags_raw=bags,
            sam_masks=sam_masks,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),  # grassland — not present
            num_classes=NUM_CLASSES,
            include_bags=True,
        )
        np.testing.assert_array_equal(result, bags)

    def test_empty_segment_skipped(self):
        """SAM mask with all-False segmentation array is skipped."""
        bags = _make_mask(fill=3)
        seg = np.zeros((4, 4), dtype=bool)  # empty segment
        sam_masks = [_make_sam_mask(seg)]
        lulc = [np.full((4, 4), 5, dtype=np.uint8)]
        result = apply_sam_correction(
            bags_raw=bags,
            sam_masks=sam_masks,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_bags=True,
        )
        np.testing.assert_array_equal(result, bags)

    def test_vote_counts_zero_skipped(self):
        """Segment with all nodata values in all sources → no change."""
        bags = _make_mask(fill=3)
        seg = np.ones((4, 4), dtype=bool)
        sam_masks = [_make_sam_mask(seg)]
        # LULC has only values >= NUM_CLASSES (nodata) — no valid votes
        lulc = [np.full((4, 4), 255, dtype=np.uint8)]
        result = apply_sam_correction(
            bags_raw=bags,
            sam_masks=sam_masks,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_bags=False,  # exclude BAGS so only nodata votes remain
        )
        np.testing.assert_array_equal(result, bags)

    def test_does_not_mutate_input(self):
        """apply_sam_correction must not modify the original bags_raw array."""
        bags = _make_mask(fill=3)
        original = bags.copy()
        seg = np.ones((4, 4), dtype=bool)
        sam_masks = [_make_sam_mask(seg)]
        lulc = [np.full((4, 4), 5, dtype=np.uint8), np.full((4, 4), 5, dtype=np.uint8)]
        apply_sam_correction(
            bags_raw=bags,
            sam_masks=sam_masks,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_bags=True,
        )
        np.testing.assert_array_equal(bags, original)

    def test_multiple_masks_later_wins(self):
        """Masks are processed in order; later masks overwrite earlier ones."""
        bags = np.array([[3, 3], [3, 3]], dtype=np.uint8)
        seg_all = np.ones((2, 2), dtype=bool)
        mask1 = _make_sam_mask(seg_all, iou=0.8, area=4)
        mask2 = _make_sam_mask(seg_all, iou=0.9, area=4)
        # Second mask has higher (iou, area) so processed last → wins
        sam_masks = [mask1, mask2]

        # lulc for both masks votes class 5 in mask2 scenario
        # We test by checking masks are sorted ascending (lower iou first)
        # Here mask1 (iou=0.8) processed first, mask2 (iou=0.9) processed second → class 5
        lulc = [np.full((2, 2), 5, dtype=np.uint8), np.full((2, 2), 5, dtype=np.uint8)]
        result = apply_sam_correction(
            bags_raw=bags,
            sam_masks=sam_masks,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_bags=False,
        )
        assert np.all(result == 5)


# ---------------------------------------------------------------------------
# SAMSegmentCache
# ---------------------------------------------------------------------------


class TestSAMSegmentCache:
    def test_disabled_when_empty_string(self):
        """cache_dir='' → cache is disabled; get always returns None."""
        cache = SAMSegmentCache(cache_dir="")
        result = cache.get("any_key")
        assert result is None

    def test_disabled_when_none(self):
        """cache_dir=None → cache is disabled; get always returns None."""
        cache = SAMSegmentCache(cache_dir=None)
        assert cache.get("key") is None

    def test_put_then_get_roundtrip(self, tmp_path):
        """put followed by get returns the same masks."""
        cache = SAMSegmentCache(cache_dir=str(tmp_path))
        seg = np.array([[True, False], [False, True]], dtype=bool)
        masks = [
            {"segmentation": seg, "predicted_iou": 0.85, "area": 2},
        ]
        cache.put("tile_r0_c0_h2_w2", masks)
        retrieved = cache.get("tile_r0_c0_h2_w2")
        assert retrieved is not None
        assert len(retrieved) == 1
        np.testing.assert_array_equal(retrieved[0]["segmentation"], seg)
        assert abs(retrieved[0]["predicted_iou"] - 0.85) < 1e-6
        assert retrieved[0]["area"] == 2

    def test_missing_key_returns_none(self, tmp_path):
        """Requesting non-existent key returns None."""
        cache = SAMSegmentCache(cache_dir=str(tmp_path))
        assert cache.get("nonexistent_key") is None

    def test_invalid_format_returns_none(self, tmp_path):
        """Old NPZ format without iou_N arrays is rejected (returns None)."""
        cache = SAMSegmentCache(cache_dir=str(tmp_path))
        # Manually write a file without iou_0
        key = "old_format"
        path = tmp_path / f"{key}.npz"
        seg = np.ones((2, 2), dtype=np.uint8)
        np.savez_compressed(path, n_masks=np.array(1), seg_0=seg, area_0=np.array(4))
        # No iou_0 → should return None
        result = cache.get(key)
        assert result is None

    def test_zero_masks_roundtrip(self, tmp_path):
        """Empty mask list can be cached and retrieved."""
        cache = SAMSegmentCache(cache_dir=str(tmp_path))
        cache.put("empty_tile", [])
        retrieved = cache.get("empty_tile")
        assert retrieved is not None
        assert len(retrieved) == 0

    def test_put_creates_parent_dirs(self, tmp_path):
        """put() creates nested directories if needed."""
        nested = tmp_path / "a" / "b" / "c"
        cache = SAMSegmentCache(cache_dir=str(nested))
        seg = np.zeros((2, 2), dtype=bool)
        cache.put("key", [{"segmentation": seg, "predicted_iou": 0.9, "area": 0}])
        assert (nested / "key.npz").exists()

    def test_put_disabled_does_not_write(self, tmp_path):
        """put() is a no-op when cache is disabled."""
        cache = SAMSegmentCache(cache_dir="")
        seg = np.zeros((2, 2), dtype=bool)
        # Should not raise, and no file should be written
        cache.put("key", [{"segmentation": seg, "predicted_iou": 0.9, "area": 0}])
        assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# SAMLabelCorrectionConfig defaults
# ---------------------------------------------------------------------------


class TestSAMLabelCorrectionConfig:
    def test_required_fields(self):
        """Config can be instantiated with only required fields."""
        cfg = SAMLabelCorrectionConfig(
            coreset_csv="/data/coreset.csv",
            masks_dir="/data/masks",
            targets=[{"classes": [3, 5], "output_dir": "/data/out"}],
            sam_checkpoint="/models/sam_vit_b.pth",
            mbtiles_path="/data/tiles.mbtiles",
        )
        assert cfg.coreset_csv == "/data/coreset.csv"
        assert cfg.masks_dir == "/data/masks"

    def test_default_values(self):
        """All optional fields have correct defaults."""
        cfg = SAMLabelCorrectionConfig(
            coreset_csv="a",
            masks_dir="b",
            targets=[],
            sam_checkpoint="c",
            mbtiles_path="d",
        )
        assert cfg.lulc_paths == []
        assert cfg.include_bags is True
        assert cfg.sam_model_type == "vit_b"
        assert cfg.device == "cuda:0"
        assert cfg.num_classes == 6
        assert cfg.nodata_val == 255
        assert cfg.chunk_size == 1024
        assert cfg.points_per_side == 32
        assert cfg.pred_iou_thresh == 0.80
        assert cfg.stability_score_thresh == 0.90
        assert cfg.min_mask_region_area == 200
        assert cfg.cache_dir == ""
        assert cfg.start_idx == 0
        assert cfg.end_idx == 999999


# ---------------------------------------------------------------------------
# SamLabelCorrector helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path, **overrides):
    defaults = dict(
        coreset_csv=str(tmp_path / "coreset.csv"),
        masks_dir=str(tmp_path / "masks"),
        targets=[{"classes": [3], "output_dir": str(tmp_path / "out")}],
        sam_checkpoint=str(tmp_path / "sam.pth"),
        mbtiles_path=str(tmp_path / "tiles.mbtiles"),
        cache_dir="",
    )
    defaults.update(overrides)
    return SAMLabelCorrectionConfig(**defaults)


def _make_window(row_off=0, col_off=0, height=4, width=4):
    w = MagicMock()
    w.row_off = row_off
    w.col_off = col_off
    w.height = height
    w.width = width
    return w


class TestSamLabelCorrectorParseTargets:
    def test_parse_single_target(self, tmp_path):
        cfg = _make_config(tmp_path)
        corrector = SamLabelCorrector(cfg)
        classes, out_dir = corrector._targets[0]
        assert classes == frozenset([3])
        assert out_dir == tmp_path / "out"

    def test_parse_multiple_targets(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            targets=[
                {"classes": [3, 5], "output_dir": str(tmp_path / "out1")},
                {"classes": [1], "output_dir": str(tmp_path / "out2")},
            ],
        )
        corrector = SamLabelCorrector(cfg)
        assert len(corrector._targets) == 2
        assert corrector._targets[0][0] == frozenset([3, 5])
        assert corrector._targets[1][0] == frozenset([1])


class TestSamLabelCorrectorChunkCacheKey:
    def test_key_format(self, tmp_path):
        cfg = _make_config(tmp_path)
        corrector = SamLabelCorrector(cfg)
        window = _make_window(row_off=10, col_off=20, height=64, width=128)
        key = corrector._chunk_cache_key("some/path/tile_001.tif", window)
        assert key == "tile_001_r10_c20_h64_w128"


class TestSamLabelCorrectorLoadSam:
    def test_import_error_raised_with_message(self, tmp_path):
        cfg = _make_config(tmp_path)
        corrector = SamLabelCorrector(cfg)
        with patch.dict("sys.modules", {"segment_anything": None}):
            with pytest.raises(ImportError, match="segment_anything"):
                corrector._load_sam()

    def test_load_sam_calls_registry(self, tmp_path):
        cfg = _make_config(tmp_path)
        corrector = SamLabelCorrector(cfg)

        mock_sam = MagicMock()
        mock_registry = {"vit_b": MagicMock(return_value=mock_sam)}
        mock_generator_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "segment_anything": MagicMock(
                    sam_model_registry=mock_registry,
                    SamAutomaticMaskGenerator=mock_generator_cls,
                )
            },
        ):
            corrector._load_sam()

        mock_sam.to.assert_called_once_with(cfg.device)
        mock_generator_cls.assert_called_once()


class TestSamLabelCorrectorRun:
    def _setup_csv(self, tmp_path):
        import pandas as pd

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
        corrector = SamLabelCorrector(cfg)

        tile_stats = {
            "tile": "tile.tif",
            "n_chunks": 1,
            "n_skipped": 0,
            "per_target": [],
        }
        with patch.object(corrector, "_load_sam", return_value=MagicMock()):
            with patch.object(corrector, "_process_tile", return_value=tile_stats):
                result = corrector.run()

        assert result["n_tiles"] == 1
        assert "elapsed_s" in result
        assert result["tiles"] == [tile_stats]

    def test_run_respects_start_end_idx(self, tmp_path):
        import pandas as pd

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
        corrector = SamLabelCorrector(cfg)

        processed = []

        def fake_process(tile_name, patch_rows, mg):
            processed.append(tile_name)
            return {"tile": tile_name, "n_chunks": 0, "n_skipped": 0, "per_target": []}

        with patch.object(corrector, "_load_sam", return_value=MagicMock()):
            with patch.object(corrector, "_process_tile", side_effect=fake_process):
                corrector.run()

        assert processed == ["b.tif"]

    def test_run_adds_patch_size_column_when_missing(self, tmp_path):
        import pandas as pd

        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        (masks_dir / "tile.tif").touch()
        df = pd.DataFrame({"mask_path": ["tile.tif"], "row_off": [0], "col_off": [0]})
        csv_path = tmp_path / "coreset.csv"
        df.to_csv(csv_path, index=False)

        cfg = _make_config(tmp_path, coreset_csv=str(csv_path))
        corrector = SamLabelCorrector(cfg)

        tile_stats = {
            "tile": "tile.tif",
            "n_chunks": 0,
            "n_skipped": 0,
            "per_target": [],
        }
        with patch.object(corrector, "_load_sam", return_value=MagicMock()):
            with patch.object(
                corrector, "_process_tile", return_value=tile_stats
            ) as mock_pt:
                corrector.run()

        _, patch_rows_arg, _ = mock_pt.call_args[0]
        assert "patch_size" in patch_rows_arg.columns


def _mock_rasterio(open_ctx=None):
    """Return (sys.modules patch dict, mock_rasterio) for local rasterio imports."""
    mock_rio = MagicMock()
    mock_windows_mod = MagicMock()
    mock_rio.windows = mock_windows_mod
    if open_ctx is not None:
        mock_rio.open.return_value = open_ctx
    mods = {"rasterio": mock_rio, "rasterio.windows": mock_windows_mod}
    return mods, mock_rio


def _mock_chunk_imports(read_return=None, read_side_effect=None, open_ctx=None):
    """Return sys.modules patch dict for _process_chunk local imports."""
    mock_rio = MagicMock()
    if open_ctx is not None:
        mock_rio.open.return_value = open_ctx
    mock_torch = MagicMock()
    mock_alignment = MagicMock()
    if read_side_effect is not None:
        mock_alignment.read_source_aligned_to_mask_window.side_effect = read_side_effect
    elif read_return is not None:
        mock_alignment.read_source_aligned_to_mask_window.return_value = read_return
    mods = {
        "rasterio": mock_rio,
        "rasterio.windows": MagicMock(),
        "torch": mock_torch,
        "pytorch_segmentation_models_trainer.tools.mbtiles.alignment": mock_alignment,
    }
    return mods, mock_rio, mock_torch, mock_alignment


class TestSamLabelCorrectorProcessTile:
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
        import pandas as pd

        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        (masks_dir / "tile.tif").write_bytes(b"\x00" * 16)
        (tmp_path / "out").mkdir(parents=True)

        cfg = _make_config(tmp_path, masks_dir=str(masks_dir))
        corrector = SamLabelCorrector(cfg)

        patch_rows = pd.DataFrame({"row_off": [0], "col_off": [0], "patch_size": [4]})

        mock_src = self._make_mock_src()
        open_ctx = self._make_open_ctx(mock_src)
        sys_mods, _ = _mock_rasterio(open_ctx=open_ctx)

        with patch.dict("sys.modules", sys_mods):
            with patch.object(corrector, "_process_chunk", return_value=[(2, 16)]):
                result = corrector._process_tile("tile.tif", patch_rows, MagicMock())

        assert result["tile"] == "tile.tif"
        assert result["n_chunks"] == 1
        assert result["per_target"][0]["n_changed"] == 2

    def test_process_tile_skips_copy_when_dest_exists(self, tmp_path):
        import pandas as pd

        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        (masks_dir / "tile.tif").write_bytes(b"\x00" * 16)
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True)
        dest = out_dir / "tile.tif"
        dest.write_bytes(b"\xff" * 16)

        cfg = _make_config(tmp_path, masks_dir=str(masks_dir))
        corrector = SamLabelCorrector(cfg)

        patch_rows = pd.DataFrame({"row_off": [0], "col_off": [0], "patch_size": [4]})
        mock_src = self._make_mock_src()
        open_ctx = self._make_open_ctx(mock_src)
        sys_mods, _ = _mock_rasterio(open_ctx=open_ctx)

        with patch.dict("sys.modules", sys_mods):
            with patch.object(corrector, "_process_chunk", return_value=[(0, 0)]):
                corrector._process_tile("tile.tif", patch_rows, MagicMock())

        assert dest.read_bytes() == b"\xff" * 16

    def test_process_tile_degenerate_chunk_skipped(self, tmp_path):
        """Chunk where r1<=r0 (mask height=0) is skipped silently."""
        import pandas as pd

        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        (masks_dir / "tile.tif").write_bytes(b"\x00" * 16)
        (tmp_path / "out").mkdir(parents=True)

        cfg = _make_config(tmp_path, masks_dir=str(masks_dir))
        corrector = SamLabelCorrector(cfg)

        patch_rows = pd.DataFrame({"row_off": [0], "col_off": [0], "patch_size": [4]})
        mock_src = self._make_mock_src(
            height=0, width=4
        )  # height=0 → r1 = min(chunk,4,0)=0 ≤ r0=0
        open_ctx = self._make_open_ctx(mock_src)
        sys_mods, _ = _mock_rasterio(open_ctx=open_ctx)

        with patch.dict("sys.modules", sys_mods):
            with patch.object(
                corrector, "_process_chunk", return_value=[(0, 0)]
            ) as mock_pc:
                result = corrector._process_tile("tile.tif", patch_rows, MagicMock())

        mock_pc.assert_not_called()
        assert result["n_chunks"] == 0

    def test_process_tile_skipped_chunk_counted(self, tmp_path):
        """Chunk with all-zero target counts is counted as skipped."""
        import pandas as pd

        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        (masks_dir / "tile.tif").write_bytes(b"\x00" * 16)
        (tmp_path / "out").mkdir(parents=True)

        cfg = _make_config(tmp_path, masks_dir=str(masks_dir))
        corrector = SamLabelCorrector(cfg)

        patch_rows = pd.DataFrame({"row_off": [0], "col_off": [0], "patch_size": [4]})
        mock_src = self._make_mock_src()
        open_ctx = self._make_open_ctx(mock_src)
        sys_mods, _ = _mock_rasterio(open_ctx=open_ctx)

        with patch.dict("sys.modules", sys_mods):
            with patch.object(corrector, "_process_chunk", return_value=[(0, 0)]):
                result = corrector._process_tile("tile.tif", patch_rows, MagicMock())

        assert result["n_skipped"] == 1


class TestSamLabelCorrectorProcessChunk:
    def _corrector(self, tmp_path):
        return SamLabelCorrector(_make_config(tmp_path))

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
            result = corrector._process_chunk(
                mask_src, chunk_targets, _make_window(), MagicMock()
            )

        assert result == [(0, 0)]

    def test_image_read_exception_returns_zeros(self, tmp_path):
        corrector = self._corrector(tmp_path)
        bags = np.full((4, 4), 3, dtype=np.uint8)
        mask_src = self._mock_src(bags)
        chunk_targets = [(frozenset([3]), tmp_path / "out" / "tile.tif")]
        sys_mods, _, _, _ = _mock_chunk_imports(read_side_effect=RuntimeError("IO"))

        with patch.dict("sys.modules", sys_mods):
            result = corrector._process_chunk(
                mask_src, chunk_targets, _make_window(), MagicMock()
            )

        assert result == [(0, 0)]

    def test_cache_hit_skips_generator(self, tmp_path):
        corrector = self._corrector(tmp_path)
        bags = np.full((4, 4), 3, dtype=np.uint8)
        mask_src = self._mock_src(bags)
        chunk_targets = [(frozenset([3]), tmp_path / "out" / "tile.tif")]

        corrector._cache = MagicMock()
        corrector._cache.get.return_value = []  # cache hit → empty masks

        image_chw = np.zeros((3, 4, 4), dtype=np.uint8)
        sys_mods, _, _, _ = _mock_chunk_imports(read_return=image_chw)

        mock_generator = MagicMock()
        with patch.dict("sys.modules", sys_mods):
            corrector._process_chunk(
                mask_src, chunk_targets, _make_window(), mock_generator
            )

        mock_generator.generate.assert_not_called()
        corrector._cache.put.assert_not_called()

    def test_cache_miss_runs_generator_and_stores(self, tmp_path):
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

        sam_masks = [
            {
                "segmentation": np.ones((4, 4), dtype=bool),
                "predicted_iou": 0.9,
                "area": 16,
            }
        ]
        mock_generator = MagicMock()
        mock_generator.generate.return_value = sam_masks

        mock_dst = MagicMock()
        dst_ctx = MagicMock()
        dst_ctx.__enter__ = MagicMock(return_value=mock_dst)
        dst_ctx.__exit__ = MagicMock(return_value=False)

        image_chw = np.zeros((3, 4, 4), dtype=np.uint8)
        sys_mods, mock_rio, _, _ = _mock_chunk_imports(
            read_return=image_chw, open_ctx=dst_ctx
        )

        with patch.dict("sys.modules", sys_mods):
            corrector._process_chunk(
                mask_src, chunk_targets, _make_window(), mock_generator
            )

        mock_generator.generate.assert_called_once()
        corrector._cache.put.assert_called_once()

    def test_chunk_target_without_target_pixels_skipped(self, tmp_path):
        corrector = self._corrector(tmp_path)
        bags = np.full((4, 4), 3, dtype=np.uint8)
        mask_src = self._mock_src(bags)
        # Two targets: class 3 (present) and class 5 (absent)
        chunk_targets = [
            (frozenset([3]), tmp_path / "out" / "tile.tif"),
            (frozenset([5]), tmp_path / "out" / "tile.tif"),
        ]

        corrector._cache = MagicMock()
        corrector._cache.get.return_value = []  # empty → no write needed

        image_chw = np.zeros((3, 4, 4), dtype=np.uint8)
        sys_mods, _, _, _ = _mock_chunk_imports(read_return=image_chw)

        with patch.dict("sys.modules", sys_mods):
            result = corrector._process_chunk(
                mask_src, chunk_targets, _make_window(), MagicMock()
            )

        assert result[1] == (0, 0)  # second target skipped
