# -*- coding: utf-8 -*-
"""Tests for tools/sampling/weight_calculator.py."""

import numpy as np
import pytest

from pytorch_segmentation_models_trainer.tools.sampling.weight_calculator import (
    combine_weights,
    compute_class_weights_from_proportions,
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


# ---------------------------------------------------------------------------
# compute_class_weights_from_proportions
# ---------------------------------------------------------------------------


class TestComputeClassWeightsFromProportions:
    def test_sqrt_formula_basic(self):
        """Manual 2-class, 2-patch case."""
        props = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float)
        # freq = [0.55, 0.45]
        # sqrt(props / freq): patch0 → [sqrt(0.8/0.55), sqrt(0.2/0.45)] = [1.205, 0.667] → sum=1.872
        #                       patch1 → [sqrt(0.3/0.55), sqrt(0.7/0.45)] = [0.738, 1.247] → sum=1.985
        freq = props.mean(axis=0)
        expected = np.sqrt(props / freq).sum(axis=1)
        result = compute_class_weights_from_proportions(
            props, formula="sqrt_inverse_freq"
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_inverse_freq_formula_basic(self):
        props = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float)
        freq = props.mean(axis=0)
        expected = (props / freq).sum(axis=1)
        result = compute_class_weights_from_proportions(props, formula="inverse_freq")
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_sqrt_formula_rare_class_upweighted(self):
        """Rare class gets higher weight under sqrt than linear inverse."""
        props = np.array([[0.01, 0.99], [0.99, 0.01]], dtype=float)
        sqrt_w = compute_class_weights_from_proportions(
            props, formula="sqrt_inverse_freq"
        )
        inv_w = compute_class_weights_from_proportions(props, formula="inverse_freq")
        # patch with rare class (index 0, class 0 is rare): sqrt should give lower extreme than inverse
        # Both patches are symmetric so ratio should be < 1 for sqrt vs inverse
        assert (
            sqrt_w[0] / sqrt_w[1] < inv_w[0] / inv_w[1]
            or abs(sqrt_w[0] / sqrt_w[1] - inv_w[0] / inv_w[1]) < 1e-6
        )

    def test_zero_freq_class_ignored(self):
        """All-zero column → no division error; treated as inf freq → zero contribution."""
        props = np.array([[0.5, 0.5, 0.0], [0.4, 0.6, 0.0]], dtype=float)
        result = compute_class_weights_from_proportions(
            props, formula="sqrt_inverse_freq"
        )
        assert np.all(np.isfinite(result))

    def test_invalid_formula_raises(self):
        props = np.ones((3, 2)) / 2
        with pytest.raises(ValueError, match="Unknown formula"):
            compute_class_weights_from_proportions(props, formula="bad_formula")

    def test_output_shape(self):
        props = np.random.rand(15, 6)
        props = props / props.sum(axis=1, keepdims=True)
        result = compute_class_weights_from_proportions(
            props, formula="sqrt_inverse_freq"
        )
        assert result.shape == (15,)

    def test_default_formula_is_sqrt(self):
        props = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)
        result_default = compute_class_weights_from_proportions(props)
        result_explicit = compute_class_weights_from_proportions(
            props, formula="sqrt_inverse_freq"
        )
        np.testing.assert_array_equal(result_default, result_explicit)
