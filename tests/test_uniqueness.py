# -*- coding: utf-8 -*-
"""Tests for tools/sampling/uniqueness.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pytorch_segmentation_models_trainer.tools.sampling.uniqueness import (
    compute_uniqueness,
    compute_uniqueness_sklearn,
)


def test_uniqueness_identical_embeddings_zero():
    emb = np.ones((20, 8), dtype=np.float32)
    scores = compute_uniqueness_sklearn(emb, k=5)
    assert scores.shape == (20,)
    np.testing.assert_allclose(scores, 0.0, atol=1e-5)


def test_uniqueness_output_shape():
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((50, 16)).astype(np.float32)
    scores = compute_uniqueness_sklearn(emb, k=5)
    assert scores.shape == (50,)


def test_uniqueness_orthogonal_vectors():
    n = 8
    emb = np.eye(n, dtype=np.float32)
    scores = compute_uniqueness_sklearn(emb, k=1)
    assert scores.shape == (n,)
    np.testing.assert_allclose(scores, 1.0, atol=1e-5)


def test_uniqueness_scores_non_negative():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((30, 16)).astype(np.float32)
    scores = compute_uniqueness_sklearn(emb, k=3)
    assert np.all(scores >= 0)


def test_uniqueness_default_backend_sklearn():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((30, 8)).astype(np.float32)
    scores = compute_uniqueness(emb, k=3, backend="sklearn")
    assert scores.shape == (30,)
    assert np.all(scores >= 0)


def test_uniqueness_invalid_backend_raises():
    emb = np.ones((10, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="backend"):
        compute_uniqueness(emb, k=3, backend="unknown_backend")


def test_uniqueness_k_smaller_than_n():
    rng = np.random.default_rng(5)
    emb = rng.standard_normal((10, 4)).astype(np.float32)
    scores = compute_uniqueness_sklearn(emb, k=3)
    assert scores.shape == (10,)


def test_compute_uniqueness_faiss_backend_dispatches():
    """compute_uniqueness with backend='faiss' calls compute_uniqueness_faiss."""
    n, k, d = 10, 5, 4
    rng = np.random.default_rng(14)
    emb = rng.standard_normal((n, d)).astype(np.float32)

    mock_faiss = MagicMock()
    index_mock = MagicMock()
    sims = np.full((n, k + 1), 0.8, dtype=np.float32)
    sims[:, 0] = 1.0
    index_mock.search.return_value = (sims, np.zeros((n, k + 1), dtype=np.int64))
    mock_faiss.IndexFlatIP.return_value = index_mock

    with patch.dict("sys.modules", {"faiss": mock_faiss}):
        scores = compute_uniqueness(emb, k=k, backend="faiss")

    assert scores.shape == (n,)


def test_uniqueness_faiss_import_error_propagated():
    """compute_uniqueness_faiss raises ImportError when faiss not installed."""
    from pytorch_segmentation_models_trainer.tools.sampling.uniqueness import (
        compute_uniqueness_faiss,
    )

    rng = np.random.default_rng(12)
    emb = rng.standard_normal((10, 4)).astype(np.float32)

    with patch.dict("sys.modules", {"faiss": None}):
        with pytest.raises(ImportError, match="faiss"):
            compute_uniqueness_faiss(emb, k=3)


def test_uniqueness_faiss_mocked():
    """compute_uniqueness_faiss runs correctly with a mocked faiss module."""
    from unittest.mock import patch

    from pytorch_segmentation_models_trainer.tools.sampling.uniqueness import (
        compute_uniqueness_faiss,
    )

    n, k, d = 10, 5, 4
    rng = np.random.default_rng(13)
    emb = rng.standard_normal((n, d)).astype(np.float32)

    mock_faiss = MagicMock()
    index_mock = MagicMock()
    # Simulated cosine similarities: column 0 = self (1.0), rest = 0.8
    sims = np.full((n, k + 1), 0.8, dtype=np.float32)
    sims[:, 0] = 1.0
    index_mock.search.return_value = (sims, np.zeros((n, k + 1), dtype=np.int64))
    mock_faiss.IndexFlatIP.return_value = index_mock

    with patch.dict("sys.modules", {"faiss": mock_faiss}):
        scores = compute_uniqueness_faiss(emb, k=k)

    assert scores.shape == (n,)
    # cosine_dist = 1 - 0.8 = 0.2 for all non-self neighbors
    np.testing.assert_allclose(scores, 0.2, atol=1e-5)


def test_uniqueness_faiss_matches_sklearn():
    pytest.importorskip("faiss")
    from pytorch_segmentation_models_trainer.tools.sampling.uniqueness import (
        compute_uniqueness_faiss,
    )

    rng = np.random.default_rng(7)
    emb = rng.standard_normal((40, 16)).astype(np.float32)
    s_faiss = compute_uniqueness_faiss(emb, k=5)
    s_sklearn = compute_uniqueness_sklearn(emb, k=5)
    np.testing.assert_allclose(s_faiss, s_sklearn, atol=1e-4)


def test_compute_uniqueness_hnsw_backend_dispatches():
    """compute_uniqueness with backend='hnsw' dispatches to compute_uniqueness_hnsw."""
    n, k, d = 10, 3, 4
    rng = np.random.default_rng(20)
    emb = rng.standard_normal((n, d)).astype(np.float32)

    mock_hnswlib = MagicMock()
    index_mock = MagicMock()
    fake_dists = np.full((n, k + 1), 0.3, dtype=np.float32)
    fake_dists[:, 0] = 0.0  # self-distance
    index_mock.knn_query.return_value = (
        np.zeros((n, k + 1), dtype=np.int64),
        fake_dists,
    )
    mock_hnswlib.Index.return_value = index_mock

    with patch.dict("sys.modules", {"hnswlib": mock_hnswlib}):
        scores = compute_uniqueness(emb, k=k, backend="hnsw")

    assert scores.shape == (n,)
    np.testing.assert_allclose(scores, 0.3, atol=1e-5)


def test_compute_uniqueness_hnsw_import_error():
    """compute_uniqueness_hnsw raises ImportError when hnswlib not installed."""
    from pytorch_segmentation_models_trainer.tools.sampling.uniqueness import (
        compute_uniqueness_hnsw,
    )

    emb = np.ones((10, 4), dtype=np.float32)

    with patch.dict("sys.modules", {"hnswlib": None}):
        with pytest.raises(ImportError, match="hnswlib"):
            compute_uniqueness_hnsw(emb, k=3)


def test_uniqueness_invalid_backend_error_message_lists_hnsw():
    """Error message for invalid backend now lists 'hnsw' as an option."""
    emb = np.ones((10, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="hnsw"):
        compute_uniqueness(emb, k=3, backend="bad")


def test_uniqueness_higher_for_outlier():
    # 99 vectors all pointing in roughly the +0 direction (high cosine similarity to each other)
    # 1 outlier pointing in the -0 direction (maximally dissimilar to the cluster)
    n = 99
    cluster = np.zeros((n, 8), dtype=np.float32)
    cluster[:, 0] = 1.0  # all point along +axis 0
    outlier = np.array(
        [[0.0] + [0.0] * 6 + [1.0]], dtype=np.float32
    )  # points along +axis 7
    emb = np.vstack([cluster, outlier])
    scores = compute_uniqueness_sklearn(emb, k=5)
    assert scores[-1] > scores[:-1].mean()
