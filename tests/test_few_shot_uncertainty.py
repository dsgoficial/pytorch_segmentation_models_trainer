# -*- coding: utf-8 -*-
"""Tests for subjective-logic uncertainty measures and their evaluation."""

import math

import pytest
import torch

from pytorch_segmentation_models_trainer.few_shot.uncertainty import (
    GFSSUncertaintyMetrics,
    dissonance,
    normalized_entropy,
    vacuity,
)


class TestMeasures:
    def test_vacuity(self):
        alpha = torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1)
        assert vacuity(alpha).item() == pytest.approx(1.0)
        alpha = torch.tensor([10.0, 1.0, 1.0]).view(1, 3, 1, 1)
        assert vacuity(alpha).item() == pytest.approx(3 / 12)
        assert vacuity(alpha).shape == (1, 1, 1)

    def test_dissonance_extremes(self):
        # equal beliefs in two classes -> maximal conflict between them
        b = torch.tensor([0.5, 0.5, 0.0]).view(1, 3, 1, 1)
        assert dissonance(b).item() == pytest.approx(1.0)
        # belief in one class only -> no conflict
        b = torch.tensor([0.9, 0.0, 0.0]).view(1, 3, 1, 1)
        assert dissonance(b).item() == pytest.approx(0.0)
        # no belief at all -> 0 (pure vacuity)
        assert dissonance(torch.zeros(1, 3, 1, 1)).item() == 0.0

    def test_dissonance_known_value(self):
        b = torch.tensor([0.6, 0.2, 0.0]).view(1, 3, 1, 1)
        bal = 1 - abs(0.6 - 0.2) / 0.8
        expected = 0.6 * (0.2 * bal) / 0.2 + 0.2 * (0.6 * bal) / 0.6
        assert dissonance(b).item() == pytest.approx(expected)

    def test_normalized_entropy(self):
        q = torch.tensor([0.5, 0.5]).view(1, 2, 1, 1)
        assert normalized_entropy(q).item() == pytest.approx(1.0)
        q = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1)
        assert normalized_entropy(q).item() == pytest.approx(0.0, abs=1e-6)
        assert normalized_entropy(torch.ones(1, 1, 2, 2)).abs().max() == 0


class TestUncertaintyMetrics:
    def test_aurc_perfect_and_random_ordering(self):
        m = GFSSUncertaintyMetrics(["u"], n_bins=10)
        correct = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        region = torch.ones_like(correct)
        # errors have the highest uncertainty -> perfect ranking
        m.update(
            {"u": torch.tensor([[0.05, 0.15, 0.85, 0.95]])}, region, correct, region
        )
        out = m.compute()
        # coverage 1/4: 0, 2/4: 0, 3/4: 1/3, 4/4: 1/2 -> mean 0.2083
        assert out["aurc/u"].item() == pytest.approx((0 + 0 + 1 / 3 + 1 / 2) / 4)

    def test_region_mask_and_reset(self):
        m = GFSSUncertaintyMetrics(["u"], n_bins=4, prefix="test/")
        correct = torch.tensor([[1, 0]], dtype=torch.bool)
        region = torch.tensor([[1, 0]], dtype=torch.bool)
        m.update(
            {"u": torch.tensor([[0.1, 0.9]])}, region, correct, torch.ones_like(region)
        )
        assert m.compute()["test/aurc/u"].item() == 0.0
        m.reset()
        assert torch.isnan(m.compute()["test/aurc/u"])

    def test_tile_spearman(self):
        m = GFSSUncertaintyMetrics(["u"], n_bins=4)
        for err in [0.0, 0.5, 1.0]:
            correct = torch.zeros(1, 2, dtype=torch.bool)
            if err == 0.0:
                correct[:] = True
            elif err == 0.5:
                correct[0, 0] = True
            u = torch.full((1, 2), err * 0.8 + 0.1)
            m.update(
                {"u": u}, torch.ones_like(correct), correct, torch.ones_like(correct)
            )
        out = m.compute()
        assert out["tile_spearman/u"].item() == pytest.approx(1.0)
        assert out["tile_mean/u"].item() == pytest.approx(0.5)

    def test_spearman_undefined_with_one_tile(self):
        m = GFSSUncertaintyMetrics(["u"], n_bins=4)
        c = torch.ones(1, 2, dtype=torch.bool)
        m.update({"u": torch.rand(1, 2)}, c, c, c)
        assert math.isnan(m.compute()["tile_spearman/u"].item())

    def test_unknown_measure_raises(self):
        m = GFSSUncertaintyMetrics(["u"])
        c = torch.ones(1, 2, dtype=torch.bool)
        with pytest.raises(KeyError, match="v"):
            m.update({"v": torch.rand(1, 2)}, c, c, c)


class TestAbstentionAndCalibration:
    def _filled(self, **kw):
        # 4 decisions: u = 0.05 (ok), 0.15 (ok), 0.85 (error), 0.95 (error)
        m = GFSSUncertaintyMetrics(["u"], n_bins=10, **kw)
        correct = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        region = torch.ones_like(correct)
        m.update(
            {"u": torch.tensor([[0.05, 0.15, 0.85, 0.95]])}, region, correct, region
        )
        return m.compute()

    def test_operating_points(self):
        out = self._filled(abstain_thresholds=[0.5, 0.9, 1.0])
        assert out["coverage@0.5/u"].item() == pytest.approx(0.5)
        assert out["risk@0.5/u"].item() == pytest.approx(0.0)
        assert out["coverage@0.9/u"].item() == pytest.approx(0.75)
        assert out["risk@0.9/u"].item() == pytest.approx(1 / 3)
        assert out["coverage@1/u"].item() == pytest.approx(1.0)
        assert out["risk@1/u"].item() == pytest.approx(0.5)

    def test_risk_undefined_when_nothing_retained(self):
        out = self._filled(abstain_thresholds=[0.0])
        assert out["coverage@0/u"].item() == 0.0
        assert math.isnan(out["risk@0/u"].item())

    def test_ece(self):
        # confidence = 1 - u: 0.95 and 0.85 (correct), 0.15 and 0.05 (wrong)
        out = self._filled(ece_bins=10)
        expected = (0.05 + 0.15 + 0.15 + 0.05) / 4
        assert out["ece/u"].item() == pytest.approx(expected, abs=1e-6)

    def test_ece_undefined_without_pixels(self):
        m = GFSSUncertaintyMetrics(["u"])
        assert math.isnan(m.compute()["ece/u"].item())

    def test_ece_with_coarse_bins_not_dividing_histogram(self):
        # 3 coarse bins over 10 fine ones: [0.05, 0.15] | - | [0.85, 0.95]
        out = self._filled(ece_bins=3)
        expected = 0.5 * abs(1.0 - 0.9) + 0.5 * abs(0.0 - 0.1)
        assert out["ece/u"].item() == pytest.approx(expected, abs=1e-6)
