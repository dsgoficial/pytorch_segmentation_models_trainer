# -*- coding: utf-8 -*-
"""Tests for tools/sampling/class_freq_clustering.py."""

import numpy as np
import pytest

from pytorch_segmentation_models_trainer.tools.sampling.class_freq_clustering import (
    clr_transform,
    compute_cluster_weights,
    fit_kmeans_on_clr,
    select_k_elbow,
)


def test_clr_transform_equal_proportions():
    x = np.array([[0.25, 0.25, 0.25, 0.25]])
    result = clr_transform(x)
    assert result.shape == (1, 4)
    np.testing.assert_allclose(result, np.zeros((1, 4)), atol=1e-5)


def test_clr_transform_known():
    x = np.array([[0.9, 0.1]])
    result = clr_transform(x)
    assert result.shape == (1, 2)
    log_vals = np.log(np.array([0.9, 0.1]))
    expected = log_vals - log_vals.mean()
    np.testing.assert_allclose(result[0], expected, atol=1e-5)


def test_clr_transform_zero_class_no_crash():
    x = np.array([[0.0, 0.5, 0.5]])
    result = clr_transform(x, eps=1e-6)
    assert np.all(np.isfinite(result))


def test_clr_transform_row_sums_zero():
    rng = np.random.default_rng(42)
    x = rng.dirichlet([1, 1, 1, 1], size=10)
    result = clr_transform(x)
    assert result.shape == (10, 4)
    np.testing.assert_allclose(result.sum(axis=1), np.zeros(10), atol=1e-5)


def test_clr_transform_multiple_rows_shape():
    x = np.ones((5, 6)) / 6
    result = clr_transform(x)
    assert result.shape == (5, 6)


def test_kmeans_cluster_count_fixed():
    rng = np.random.default_rng(0)
    freq = rng.dirichlet([1, 2, 3, 4, 5, 6], size=200)
    result = fit_kmeans_on_clr(freq, k_min=3, k_max=6, k_selection="fixed", k=4)
    assert "cluster_ids" in result
    assert "cluster_sizes" in result
    assert "cluster_weights" in result
    unique_ids = np.unique(result["cluster_ids"])
    assert len(unique_ids) == 4


def test_kmeans_output_length():
    rng = np.random.default_rng(1)
    freq = rng.dirichlet([1, 1, 1, 1, 1, 1], size=50)
    result = fit_kmeans_on_clr(freq, k_min=2, k_max=4, k_selection="fixed", k=3)
    assert len(result["cluster_ids"]) == 50
    assert len(result["cluster_weights"]) == 50


def test_kmeans_elbow_selection():
    rng = np.random.default_rng(2)
    freq = rng.dirichlet([1, 2, 3, 4, 5, 6], size=120)
    result = fit_kmeans_on_clr(freq, k_min=3, k_max=8, k_selection="elbow")
    unique_ids = np.unique(result["cluster_ids"])
    assert 3 <= len(unique_ids) <= 8


def test_select_k_elbow_returns_valid_k():
    inertias = [1000, 600, 400, 300, 250, 220, 210, 205, 203]
    k_min = 2
    k = select_k_elbow(inertias, k_min=k_min)
    assert k_min <= k <= k_min + len(inertias) - 1


def test_select_k_elbow_single_element():
    k = select_k_elbow([500], k_min=3)
    assert k == 3


def test_cluster_weight_inverse_size():
    rng = np.random.default_rng(3)
    freq = rng.dirichlet([1, 1, 1, 1, 1, 1], size=90)
    result = fit_kmeans_on_clr(freq, k_min=3, k_max=3, k_selection="fixed", k=3)
    ids = result["cluster_ids"]
    weights = result["cluster_weights"]
    # For each pair of clusters: larger cluster → smaller weight
    unique_ids = np.unique(ids)
    cluster_sizes = {cid: int((ids == cid).sum()) for cid in unique_ids}
    for i, cid_a in enumerate(unique_ids):
        for cid_b in unique_ids[i + 1 :]:
            if cluster_sizes[cid_a] < cluster_sizes[cid_b]:
                assert weights[ids == cid_a][0] > weights[ids == cid_b][0]
            elif cluster_sizes[cid_a] > cluster_sizes[cid_b]:
                assert weights[ids == cid_a][0] < weights[ids == cid_b][0]


def test_compute_cluster_weights_basic():
    cluster_ids = np.array([0, 0, 1, 1, 1, 2])
    weights = compute_cluster_weights(cluster_ids)
    assert weights[0] == pytest.approx(0.5)
    assert weights[1] == pytest.approx(0.5)
    assert weights[2] == pytest.approx(1 / 3)
    assert weights[3] == pytest.approx(1 / 3)
    assert weights[4] == pytest.approx(1 / 3)
    assert weights[5] == pytest.approx(1.0)


def test_compute_cluster_weights_all_same():
    cluster_ids = np.array([0, 0, 0, 0])
    weights = compute_cluster_weights(cluster_ids)
    np.testing.assert_allclose(weights, 0.25)


def test_kmeans_weights_positive():
    rng = np.random.default_rng(4)
    freq = rng.dirichlet([1, 1, 1, 1, 1, 1], size=60)
    result = fit_kmeans_on_clr(freq, k_min=3, k_max=5, k_selection="fixed", k=3)
    assert np.all(result["cluster_weights"] > 0)
