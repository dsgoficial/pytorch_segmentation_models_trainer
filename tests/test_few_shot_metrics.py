# -*- coding: utf-8 -*-
"""Tests for the GFSS confusion-matrix metrics."""

import pytest
import torch

from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy
from pytorch_segmentation_models_trainer.few_shot.metrics import GFSSMetrics


@pytest.fixture
def h():
    # base: 0, 1 (untouched), 2 (mother); novel 3 = child of 2
    return ClassHierarchy({2: [2, 3]}, num_base_classes=3)


def _t(x):
    return torch.tensor(x).view(1, 1, -1)


class TestConfusion:
    def test_confusion_matrices_and_ignore(self, h):
        m = GFSSMetrics(h)
        target = _t([0, 1, 2, 3, 3, 255])
        pred = _t([0, 1, 3, 3, 2, 0])
        base_pred = _t([0, 0, 2, 2, 1, 0])
        m.update(pred, target, base_pred)
        assert m.confmat.shape == (4, 4)
        assert m.base_confmat.shape == (4, 3)
        assert m.confmat.sum() == 5 and m.base_confmat.sum() == 5
        assert m.confmat[3, 3] == 1 and m.confmat[3, 2] == 1 and m.confmat[2, 3] == 1
        assert m.base_confmat[3, 2] == 1 and m.base_confmat[3, 1] == 1

    def test_base_pred_optional(self, h):
        m = GFSSMetrics(h)
        m.update(_t([0, 1]), _t([0, 1]))
        out = m.compute()
        assert "locality" not in out and "split_ceiling/3" not in out
        assert out["iou/0"] == 1.0


class TestCompute:
    def test_values(self, h):
        m = GFSSMetrics(h, class_names=["water", "forest", "grass", "crop"])
        target = _t([0, 0, 1, 1, 2, 2, 3, 3])
        pred = _t([0, 0, 1, 0, 2, 3, 3, 2])
        base_pred = _t([0, 0, 1, 1, 2, 2, 2, 1])
        m.update(pred, target, base_pred)
        out = m.compute()
        # crop: TP 1, FP 1 (grass->crop), FN 1 -> IoU 1/3, P 0.5, R 0.5
        assert out["iou/crop"] == pytest.approx(1 / 3)
        assert out["precision/crop"] == pytest.approx(0.5)
        assert out["recall/crop"] == pytest.approx(0.5)
        # water: TP 2, FP 1 -> 2/3 ; forest: TP 1, FN 1 -> 1/2 ; grass: 1/3
        assert out["miou_base"] == pytest.approx((2 / 3 + 1 / 2 + 1 / 3) / 3)
        assert out["miou_novel"] == pytest.approx(1 / 3)
        assert out["miou"] == pytest.approx((2 / 3 + 1 / 2 + 1 / 3 + 1 / 3) / 4)
        assert out["oem_score"] == pytest.approx(
            0.4 * out["miou_base"] + 0.6 * out["miou_novel"]
        )
        # untouched classes (0, 1): acc before = 4/4, after = 3/4
        assert out["locality"] == pytest.approx(0.75)
        # split ceiling: fraction of true crop the base put in the mother (2)
        assert out["split_ceiling/crop"] == pytest.approx(0.5)
        assert out["split_ceiling/grass"] == pytest.approx(1.0)
        assert all(isinstance(v, torch.Tensor) and v.dim() == 0 for v in out.values())

    def test_absent_class_is_nan_and_skipped_in_means(self, h):
        m = GFSSMetrics(h)
        m.update(_t([0, 0, 3]), _t([0, 0, 3]))
        out = m.compute()
        assert torch.isnan(out["iou/1"]) and torch.isnan(out["iou/2"])
        assert out["miou_base"] == pytest.approx(1.0)
        assert out["miou_novel"] == pytest.approx(1.0)

    def test_accumulates_and_resets(self, h):
        m = GFSSMetrics(h)
        m.update(_t([0]), _t([0]))
        m.update(_t([1]), _t([0]))
        assert m.confmat[0].sum() == 2
        m.reset()
        assert m.confmat.sum() == 0

    def test_prefix(self, h):
        m = GFSSMetrics(h, prefix="test/")
        m.update(_t([0]), _t([0]))
        assert "test/iou/0" in m.compute()

    def test_wrong_number_of_names_raises(self, h):
        with pytest.raises(ValueError, match="class_names"):
            GFSSMetrics(h, class_names=["a"])

    def test_locality_undefined_without_untouched_pixels(self, h):
        m = GFSSMetrics(h)
        m.update(_t([2, 3]), _t([2, 3]), _t([2, 2]))
        assert torch.isnan(m.compute()["locality"])


class TestBoundaryBand:
    def test_disabled_by_default(self, h):
        m = GFSSMetrics(h)
        m.update(_t([0, 0, 1, 1]), _t([0, 0, 1, 1]))
        assert not any(k.startswith("boundary/") for k in m.compute())

    def test_band_marks_pixels_near_label_changes(self):
        from pytorch_segmentation_models_trainer.few_shot.metrics import (
            boundary_band,
        )

        target = _t([0, 0, 0, 1, 1, 1, 1])
        band = boundary_band(target, width=1, ignore_index=255)
        assert band.tolist() == [[[False, False, True, True, False, False, False]]]
        band2 = boundary_band(target, width=2, ignore_index=255)
        assert band2.tolist() == [[[False, True, True, True, True, False, False]]]

    def test_ignore_and_tile_border_are_not_boundaries(self):
        from pytorch_segmentation_models_trainer.few_shot.metrics import (
            boundary_band,
        )

        target = _t([255, 255, 1, 1, 1])
        band = boundary_band(target, width=1, ignore_index=255)
        assert not band.any()

    def test_2d_band(self):
        from pytorch_segmentation_models_trainer.few_shot.metrics import (
            boundary_band,
        )

        target = torch.zeros(1, 5, 5, dtype=torch.long)
        target[:, :, 3:] = 1
        band = boundary_band(target, width=1, ignore_index=255)
        assert band[0, :, 2:4].all() and not band[0, :, :2].any()
        assert not band[0, :, 4].any()

    def test_boundary_iou_only_counts_band(self, h):
        m = GFSSMetrics(h, boundary_width=1)
        target = _t([2, 2, 2, 3, 3, 3])
        # far-from-edge error (pos 0) must not count; edge pixel 2 wrong
        pred = _t([0, 2, 3, 3, 3, 3])
        m.update(pred, target)
        out = m.compute()
        assert m.boundary_confmat.sum() == 2
        assert out["boundary/iou/2"] == 0.0
        assert out["boundary/iou/3"] == pytest.approx(0.5)
        assert out["boundary/miou_novel"] == pytest.approx(0.5)
        assert torch.isnan(out["boundary/iou/0"])
        assert out["iou/2"] == pytest.approx(1 / 3)

    def test_negative_width_rejected(self, h):
        with pytest.raises(ValueError):
            GFSSMetrics(h, boundary_width=-1)
