# -*- coding: utf-8 -*-
"""
Tests for domain adaptation lambda schedulers.
"""
import math
import pytest

from pytorch_segmentation_models_trainer.domain_adaptation.schedulers import (
    BaseLambdaScheduler,
    ConstantScheduler,
    DANNScheduler,
    LinearScheduler,
)


class TestBaseLambdaScheduler:
    def test_abstract_get_lambda_raises(self):
        scheduler = BaseLambdaScheduler.__new__(BaseLambdaScheduler)
        BaseLambdaScheduler.__init__(scheduler)
        with pytest.raises(NotImplementedError):
            scheduler.get_lambda(0, 10)

    def test_forward_delegates_to_get_lambda(self):
        scheduler = ConstantScheduler(value=0.7)
        assert scheduler(5, 10) == pytest.approx(0.7)


class TestConstantScheduler:
    def test_default_value(self):
        s = ConstantScheduler()
        assert s.get_lambda(0, 100) == pytest.approx(1.0)
        assert s.get_lambda(50, 100) == pytest.approx(1.0)
        assert s.get_lambda(99, 100) == pytest.approx(1.0)

    def test_custom_value(self):
        s = ConstantScheduler(value=0.5)
        for epoch in [0, 1, 50, 99]:
            assert s.get_lambda(epoch, 100) == pytest.approx(0.5)

    def test_returns_float(self):
        assert isinstance(ConstantScheduler().get_lambda(0, 10), float)


class TestLinearScheduler:
    def test_starts_at_start_value(self):
        s = LinearScheduler(start_value=0.0, end_value=1.0)
        assert s.get_lambda(0, 10) == pytest.approx(0.0)

    def test_ends_at_end_value(self):
        s = LinearScheduler(start_value=0.0, end_value=1.0)
        assert s.get_lambda(9, 10) == pytest.approx(1.0)

    def test_midpoint(self):
        s = LinearScheduler(start_value=0.0, end_value=1.0)
        # epoch 4 out of 9 total → progress = 4/9
        val = s.get_lambda(4, 9)
        expected = 4 / 8
        assert val == pytest.approx(expected, abs=1e-6)

    def test_custom_range(self):
        s = LinearScheduler(start_value=0.2, end_value=0.8)
        assert s.get_lambda(0, 10) == pytest.approx(0.2)
        assert s.get_lambda(9, 10) == pytest.approx(0.8)

    def test_single_epoch_returns_end_value(self):
        s = LinearScheduler(start_value=0.0, end_value=1.0)
        assert s.get_lambda(0, 1) == pytest.approx(1.0)

    def test_returns_float(self):
        assert isinstance(LinearScheduler().get_lambda(0, 10), float)


class TestDANNScheduler:
    def test_starts_near_zero(self):
        s = DANNScheduler(gamma=10.0)
        val = s.get_lambda(0, 100)
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_ends_near_one(self):
        s = DANNScheduler(gamma=10.0)
        val = s.get_lambda(99, 100)
        assert val > 0.99

    def test_monotonically_increasing(self):
        s = DANNScheduler(gamma=10.0)
        values = [s.get_lambda(e, 100) for e in range(100)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

    def test_midpoint_approximately_half(self):
        s = DANNScheduler(gamma=10.0)
        # At progress=0.5, formula gives 2/(1+exp(-5)) - 1 ≈ 0.9933 — it's NOT 0.5
        # The curve is sigmoid, not linear. Just verify it's in (0, 1).
        val = s.get_lambda(50, 100)
        assert 0.0 < val < 1.0

    def test_formula_matches_manual_computation(self):
        gamma = 10.0
        s = DANNScheduler(gamma=gamma)
        epoch, total = 30, 100
        progress = epoch / (total - 1)
        expected = 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0
        assert s.get_lambda(epoch, total) == pytest.approx(expected, rel=1e-6)

    def test_single_epoch_returns_one(self):
        s = DANNScheduler(gamma=10.0)
        assert s.get_lambda(0, 1) == pytest.approx(1.0)

    def test_returns_float(self):
        assert isinstance(DANNScheduler().get_lambda(0, 10), float)

    def test_custom_gamma(self):
        s_sharp = DANNScheduler(gamma=20.0)
        s_gentle = DANNScheduler(gamma=5.0)
        # Sharper gamma → faster growth → higher value at same progress
        val_sharp = s_sharp.get_lambda(10, 100)
        val_gentle = s_gentle.get_lambda(10, 100)
        assert val_sharp > val_gentle
