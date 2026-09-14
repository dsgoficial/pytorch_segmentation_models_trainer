# -*- coding: utf-8 -*-
"""Tests for pytorch_segmentation_models_trainer.tools.region_correction."""

import numpy as np
import pytest

from pytorch_segmentation_models_trainer.tools.region_correction.correction import (
    apply_region_correction,
    chunk_cache_key,
    parse_correction_targets,
)
from pytorch_segmentation_models_trainer.tools.region_correction.segment_cache import (
    LabelMapCache,
    SegmentListCache,
)

NUM_CLASSES = 6


def _make_mask(h: int = 4, w: int = 4, fill: int = 3) -> np.ndarray:
    return np.full((h, w), fill, dtype=np.uint8)


# ---------------------------------------------------------------------------
# apply_region_correction — mask-list path (SAM AMG style)
# ---------------------------------------------------------------------------


class TestApplyRegionCorrectionMaskList:
    def test_lulc_majority_overrides_bags(self):
        bags = _make_mask(fill=3)
        seg = np.ones((4, 4), dtype=bool)
        sam_masks = [{"segmentation": seg, "predicted_iou": 0.9, "area": 16}]
        lulc = [
            np.full((4, 4), 5, dtype=np.uint8),
            np.full((4, 4), 5, dtype=np.uint8),
        ]
        result = apply_region_correction(
            base_mask=bags,
            segments=sam_masks,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_base_mask=True,
        )
        assert np.all(result == 5)

    def test_non_target_pixels_preserved(self):
        bags = np.array([[0, 0], [3, 3]], dtype=np.uint8)
        seg = np.ones((2, 2), dtype=bool)
        sam_masks = [{"segmentation": seg, "predicted_iou": 0.9, "area": 4}]
        lulc = [np.full((2, 2), 5, dtype=np.uint8)]
        result = apply_region_correction(
            base_mask=bags,
            segments=sam_masks,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_base_mask=True,
        )
        assert result[0, 0] == 0
        assert result[0, 1] == 0

    def test_does_not_mutate_input(self):
        bags = _make_mask(fill=3)
        original = bags.copy()
        seg = np.ones((4, 4), dtype=bool)
        sam_masks = [{"segmentation": seg, "predicted_iou": 0.9, "area": 16}]
        lulc = [np.full((4, 4), 5, dtype=np.uint8), np.full((4, 4), 5, dtype=np.uint8)]
        apply_region_correction(
            base_mask=bags,
            segments=sam_masks,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_base_mask=True,
        )
        np.testing.assert_array_equal(bags, original)


# ---------------------------------------------------------------------------
# apply_region_correction — label-map path (SLICO style)
# ---------------------------------------------------------------------------


class TestApplyRegionCorrectionLabelMap:
    def test_basic_majority_vote(self):
        """One superpixel (label 0) covers the whole array; LULC outvotes bags."""
        bags = _make_mask(fill=3)
        label_map = np.zeros((4, 4), dtype=np.int32)
        lulc = [
            np.full((4, 4), 5, dtype=np.uint8),
            np.full((4, 4), 5, dtype=np.uint8),
        ]
        result = apply_region_correction(
            base_mask=bags,
            segments=label_map,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_base_mask=True,
        )
        assert np.all(result == 5)

    def test_two_disjoint_segments_vote_independently(self):
        """Left half (label 0) and right half (label 1) get different winners."""
        bags = _make_mask(fill=3)
        label_map = np.zeros((4, 4), dtype=np.int32)
        label_map[:, 2:] = 1
        lulc_left_right = np.array(
            [[5, 5, 4, 4]] * 4, dtype=np.uint8
        )  # left votes 5, right votes 4
        result = apply_region_correction(
            base_mask=bags,
            segments=label_map,
            lulc_maps=[lulc_left_right],
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_base_mask=False,
        )
        assert np.all(result[:, :2] == 5)
        assert np.all(result[:, 2:] == 4)

    def test_non_target_pixels_preserved(self):
        bags = np.array([[0, 0], [3, 3]], dtype=np.uint8)
        label_map = np.zeros((2, 2), dtype=np.int32)
        lulc = [np.full((2, 2), 5, dtype=np.uint8)]
        result = apply_region_correction(
            base_mask=bags,
            segments=label_map,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_base_mask=True,
        )
        assert result[0, 0] == 0
        assert result[0, 1] == 0

    def test_segment_without_target_class_skipped(self):
        bags = _make_mask(fill=0)  # all water — not in target
        label_map = np.zeros((4, 4), dtype=np.int32)
        lulc = [np.full((4, 4), 5, dtype=np.uint8)]
        result = apply_region_correction(
            base_mask=bags,
            segments=label_map,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_base_mask=True,
        )
        np.testing.assert_array_equal(result, bags)

    def test_no_target_pixels_anywhere_returns_original(self):
        bags = _make_mask(fill=0)
        label_map = np.zeros((4, 4), dtype=np.int32)
        result = apply_region_correction(
            base_mask=bags,
            segments=label_map,
            lulc_maps=[],
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_base_mask=True,
        )
        np.testing.assert_array_equal(result, bags)

    def test_vote_counts_zero_skipped(self):
        """All sources nodata (>= num_classes) in the target segment → no change."""
        bags = _make_mask(fill=3)
        label_map = np.zeros((4, 4), dtype=np.int32)
        lulc = [np.full((4, 4), 255, dtype=np.uint8)]
        result = apply_region_correction(
            base_mask=bags,
            segments=label_map,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_base_mask=False,
        )
        np.testing.assert_array_equal(result, bags)

    def test_include_base_mask_false_excludes_bags_from_vote(self):
        bags = _make_mask(fill=3)
        label_map = np.zeros((4, 4), dtype=np.int32)
        lulc = [np.full((4, 4), 5, dtype=np.uint8)]
        result = apply_region_correction(
            base_mask=bags,
            segments=label_map,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_base_mask=False,
        )
        assert np.all(result == 5)

    def test_does_not_mutate_input(self):
        bags = _make_mask(fill=3)
        original = bags.copy()
        label_map = np.zeros((4, 4), dtype=np.int32)
        lulc = [np.full((4, 4), 5, dtype=np.uint8), np.full((4, 4), 5, dtype=np.uint8)]
        apply_region_correction(
            base_mask=bags,
            segments=label_map,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_base_mask=True,
        )
        np.testing.assert_array_equal(bags, original)

    def test_equivalent_to_mask_list_for_single_full_segment(self):
        """Label-map path and mask-list path agree when there is one full segment."""
        bags = _make_mask(fill=3)
        lulc = [np.full((4, 4), 5, dtype=np.uint8), np.full((4, 4), 4, dtype=np.uint8)]

        label_map = np.zeros((4, 4), dtype=np.int32)
        result_label_map = apply_region_correction(
            base_mask=bags,
            segments=label_map,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_base_mask=True,
        )

        seg = np.ones((4, 4), dtype=bool)
        sam_masks = [{"segmentation": seg, "predicted_iou": 0.9, "area": 16}]
        result_mask_list = apply_region_correction(
            base_mask=bags,
            segments=sam_masks,
            lulc_maps=lulc,
            classes_to_correct=frozenset([3]),
            num_classes=NUM_CLASSES,
            include_base_mask=True,
        )
        np.testing.assert_array_equal(result_label_map, result_mask_list)


# ---------------------------------------------------------------------------
# parse_correction_targets
# ---------------------------------------------------------------------------


class TestParseCorrectionTargets:
    def test_parse_single_target(self):
        parsed = parse_correction_targets(
            [{"classes": [3, 5], "output_dir": "/data/out"}]
        )
        assert len(parsed) == 1
        classes, out_dir = parsed[0]
        assert classes == frozenset([3, 5])
        assert str(out_dir) == "/data/out"

    def test_parse_multiple_targets(self):
        parsed = parse_correction_targets(
            [
                {"classes": [3, 5], "output_dir": "/data/out1"},
                {"classes": [1], "output_dir": "/data/out2"},
            ]
        )
        assert len(parsed) == 2
        assert parsed[0][0] == frozenset([3, 5])
        assert parsed[1][0] == frozenset([1])


# ---------------------------------------------------------------------------
# chunk_cache_key
# ---------------------------------------------------------------------------


class TestChunkCacheKey:
    def test_key_format(self):
        from unittest.mock import MagicMock

        window = MagicMock()
        window.row_off, window.col_off, window.height, window.width = 10, 20, 64, 128
        key = chunk_cache_key("some/path/tile_001.tif", window)
        assert key == "tile_001_r10_c20_h64_w128"


# ---------------------------------------------------------------------------
# SegmentListCache
# ---------------------------------------------------------------------------


class TestSegmentListCache:
    def test_disabled_when_empty_string(self):
        cache = SegmentListCache(cache_dir="")
        assert cache.get("any_key") is None

    def test_put_then_get_roundtrip(self, tmp_path):
        cache = SegmentListCache(cache_dir=str(tmp_path))
        seg = np.array([[True, False], [False, True]], dtype=bool)
        masks = [{"segmentation": seg, "predicted_iou": 0.85, "area": 2}]
        cache.put("key", masks)
        retrieved = cache.get("key")
        assert retrieved is not None
        assert len(retrieved) == 1
        np.testing.assert_array_equal(retrieved[0]["segmentation"], seg)

    def test_missing_key_returns_none(self, tmp_path):
        cache = SegmentListCache(cache_dir=str(tmp_path))
        assert cache.get("nonexistent") is None


# ---------------------------------------------------------------------------
# LabelMapCache
# ---------------------------------------------------------------------------


class TestLabelMapCache:
    def test_disabled_when_empty_string(self):
        cache = LabelMapCache(cache_dir="")
        assert cache.get("any_key") is None

    def test_disabled_put_does_not_write(self, tmp_path):
        cache = LabelMapCache(cache_dir="")
        cache.put("key", np.zeros((4, 4), dtype=np.int32))
        assert not any(tmp_path.iterdir())

    def test_put_then_get_roundtrip(self, tmp_path):
        cache = LabelMapCache(cache_dir=str(tmp_path))
        label_map = np.arange(16, dtype=np.int32).reshape(4, 4)
        cache.put("key", label_map)
        retrieved = cache.get("key")
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, label_map)

    def test_missing_key_returns_none(self, tmp_path):
        cache = LabelMapCache(cache_dir=str(tmp_path))
        assert cache.get("nonexistent") is None

    def test_put_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b"
        cache = LabelMapCache(cache_dir=str(nested))
        cache.put("key", np.zeros((2, 2), dtype=np.int32))
        assert (nested / "key.npz").exists()
