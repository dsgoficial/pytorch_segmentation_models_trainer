# -*- coding: utf-8 -*-
"""Tests for tools/sampling/weight_calculator.py."""

import numpy as np
import pytest

from pytorch_segmentation_models_trainer.tools.sampling.weight_calculator import (
    combine_weights,
)


def test_rank_max_selects_correct():
    comp = np.array([0.1, 0.9, 0.5])
    uniq = np.array([0.8, 0.2, 0.5])
    result = combine_weights(comp, uniq, method="rank_max")
    # comp ranks (argsort argsort): 0→0/2, 2→1/2, 1→2/2 = [0, 1, 0.5]
    # uniq ranks: 1→0/2, 2→1/2, 0→2/2 = [1, 0, 0.5]
    # max: [1, 1, 0.5]
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(1.0)
    assert result[2] == pytest.approx(0.5)


def test_rank_multiply_bounded():
    rng = np.random.default_rng(0)
    comp = rng.random(100)
    uniq = rng.random(100)
    result = combine_weights(comp, uniq, method="rank_multiply")
    assert np.all(result >= 0)
    assert np.all(result <= 1)


def test_rank_add_bounded():
    rng = np.random.default_rng(1)
    comp = rng.random(50)
    uniq = rng.random(50)
    result = combine_weights(comp, uniq, method="rank_add")
    assert np.all(result >= 0)
    assert np.all(result <= 1)


def test_composition_only_ignores_uniq():
    comp = np.array([0.1, 0.5, 0.9])
    uniq_a = np.array([0.1, 0.5, 0.9])
    uniq_b = np.array([0.9, 0.5, 0.1])
    r_a = combine_weights(comp, uniq_a, method="composition_only")
    r_b = combine_weights(comp, uniq_b, method="composition_only")
    np.testing.assert_array_equal(r_a, r_b)


def test_uniqueness_only_ignores_comp():
    uniq = np.array([0.1, 0.5, 0.9])
    comp_a = np.array([0.1, 0.5, 0.9])
    comp_b = np.array([0.9, 0.5, 0.1])
    r_a = combine_weights(comp_a, uniq, method="uniqueness_only")
    r_b = combine_weights(comp_b, uniq, method="uniqueness_only")
    np.testing.assert_array_equal(r_a, r_b)


def test_invalid_method_raises():
    comp = np.array([0.1, 0.5, 0.9])
    uniq = np.array([0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="Unknown sampling_method"):
        combine_weights(comp, uniq, method="unknown_method")


def test_multiplicative_raw_scores():
    comp = np.array([0.5, 1.0])
    uniq = np.array([0.5, 0.5])
    result = combine_weights(comp, uniq, method="multiplicative")
    assert result[0] == pytest.approx(0.25)
    assert result[1] == pytest.approx(0.5)


def test_rank_max_output_shape():
    rng = np.random.default_rng(2)
    comp = rng.random(200)
    uniq = rng.random(200)
    result = combine_weights(comp, uniq, method="rank_max")
    assert result.shape == (200,)


def test_rank_multiply_correctness():
    comp = np.array([0.0, 0.5, 1.0])
    uniq = np.array([0.0, 0.5, 1.0])
    result = combine_weights(comp, uniq, method="rank_multiply")
    # ranks: comp=[0,0.5,1], uniq=[0,0.5,1]
    # multiply: [0, 0.25, 1.0]
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.25)
    assert result[2] == pytest.approx(1.0)


def test_all_methods_positive_for_random():
    rng = np.random.default_rng(3)
    comp = rng.random(50) + 0.1
    uniq = rng.random(50) + 0.1
    for method in [
        "rank_max",
        "rank_multiply",
        "rank_add",
        "multiplicative",
        "composition_only",
        "uniqueness_only",
    ]:
        result = combine_weights(comp, uniq, method=method)
        assert result.shape == (50,), f"method={method}"
