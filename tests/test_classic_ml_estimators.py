# -*- coding: utf-8 -*-
"""Tests for pytorch_segmentation_models_trainer.classic_ml.estimators."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from pytorch_segmentation_models_trainer.classic_ml.estimators import (
    GPUAcceleratedKMeans,
    GPUAcceleratedRandomForest,
    GPUAcceleratedSVM,
    _to_numpy,
    enable_gpu_acceleration,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 30
N_FEATURES = 4
N_CLASSES = 3


def _classification_data():
    rng = np.random.default_rng(0)
    X = rng.random((N_SAMPLES, N_FEATURES)).astype(np.float32)
    y = rng.integers(0, N_CLASSES, size=N_SAMPLES)
    return X, y


# ---------------------------------------------------------------------------
# enable_gpu_acceleration
# ---------------------------------------------------------------------------


class TestEnableGpuAcceleration:
    def test_returns_false_when_cuml_missing(self):
        with patch.dict(sys.modules, {"cuml": None, "cuml.accel": None}):
            result = enable_gpu_acceleration()
        assert result is False

    def test_returns_true_when_cuml_available(self):
        mock_cuml = MagicMock()
        mock_cuml_accel = MagicMock()
        # `import cuml.accel` uses IMPORT_FROM bytecode, which resolves to
        # getattr(sys.modules["cuml"], "accel").  Explicitly link the attribute
        # so that cuml.accel.install() hits our mock.
        mock_cuml.accel = mock_cuml_accel
        with patch.dict(
            sys.modules, {"cuml": mock_cuml, "cuml.accel": mock_cuml_accel}
        ):
            result = enable_gpu_acceleration()
        assert result is True
        mock_cuml_accel.install.assert_called_once()


# ---------------------------------------------------------------------------
# _to_numpy
# ---------------------------------------------------------------------------


class TestToNumpy:
    def test_numpy_passthrough(self):
        arr = np.array([1.0, 2.0])
        result = _to_numpy(arr)
        assert result is arr

    def test_torch_tensor(self):
        t = torch.tensor([1.0, 2.0])
        result = _to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0])

    def test_cupy_array(self):
        class FakeCupyArray:
            pass

        expected = np.array([3.0, 4.0])
        mock_cp = MagicMock()
        mock_cp.ndarray = FakeCupyArray
        mock_cp.asnumpy.return_value = expected
        fake_arr = FakeCupyArray()

        with patch.dict(sys.modules, {"cupy": mock_cp}):
            result = _to_numpy(fake_arr)

        mock_cp.asnumpy.assert_called_once_with(fake_arr)
        np.testing.assert_array_equal(result, expected)

    def test_generic_array_like(self):
        result = _to_numpy([[1, 2], [3, 4]])
        assert isinstance(result, np.ndarray)

    def test_cupy_import_error_falls_through(self):
        with patch.dict(sys.modules, {"cupy": None}):
            result = _to_numpy([[1, 2]])
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# GPUAcceleratedRandomForest
# ---------------------------------------------------------------------------


class TestGPUAcceleratedRandomForest:
    def test_fit_predict(self):
        X, y = _classification_data()
        clf = GPUAcceleratedRandomForest(n_estimators=5, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (N_SAMPLES,)
        assert set(preds).issubset(set(range(N_CLASSES)))

    def test_predict_proba_shape(self):
        X, y = _classification_data()
        clf = GPUAcceleratedRandomForest(n_estimators=5, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (N_SAMPLES, N_CLASSES)

    def test_predict_proba_sums_to_one(self):
        X, y = _classification_data()
        clf = GPUAcceleratedRandomForest(n_estimators=5, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(N_SAMPLES), atol=1e-6)

    def test_extra_kwargs_ignored(self):
        clf = GPUAcceleratedRandomForest(n_estimators=3, random_state=0, unused=True)
        X, y = _classification_data()
        clf.fit(X, y)
        assert clf.predict(X).shape == (N_SAMPLES,)

    def test_max_depth_parameter(self):
        clf = GPUAcceleratedRandomForest(n_estimators=3, max_depth=3, random_state=0)
        X, y = _classification_data()
        clf.fit(X, y)
        assert clf._model.max_depth == 3

    def test_fit_accepts_torch_tensor(self):
        X, y = _classification_data()
        clf = GPUAcceleratedRandomForest(n_estimators=3, random_state=0)
        clf.fit(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)))
        preds = clf.predict(X)
        assert preds.shape == (N_SAMPLES,)

    def test_fit_returns_self(self):
        X, y = _classification_data()
        clf = GPUAcceleratedRandomForest(n_estimators=3, random_state=0)
        result = clf.fit(X, y)
        assert result is clf


# ---------------------------------------------------------------------------
# GPUAcceleratedSVM
# ---------------------------------------------------------------------------


class TestGPUAcceleratedSVM:
    def test_fit_predict(self):
        X, y = _classification_data()
        clf = GPUAcceleratedSVM(C=1.0, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (N_SAMPLES,)

    def test_predict_proba_shape(self):
        X, y = _classification_data()
        clf = GPUAcceleratedSVM(C=1.0, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (N_SAMPLES, N_CLASSES)

    def test_predict_proba_sums_to_one(self):
        X, y = _classification_data()
        clf = GPUAcceleratedSVM(C=1.0, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(N_SAMPLES), atol=1e-5)

    def test_kernel_parameter(self):
        clf = GPUAcceleratedSVM(kernel="linear")
        assert clf._model.kernel == "linear"

    def test_extra_kwargs_ignored(self):
        clf = GPUAcceleratedSVM(C=0.5, unused=True)
        assert clf._model.C == 0.5

    def test_fit_returns_self(self):
        X, y = _classification_data()
        clf = GPUAcceleratedSVM(random_state=0)
        result = clf.fit(X, y)
        assert result is clf


# ---------------------------------------------------------------------------
# GPUAcceleratedKMeans
# ---------------------------------------------------------------------------


class TestGPUAcceleratedKMeans:
    def test_fit_predict(self):
        X, _ = _classification_data()
        km = GPUAcceleratedKMeans(n_clusters=3, random_state=0)
        km.fit(X)
        labels = km.predict(X)
        assert labels.shape == (N_SAMPLES,)
        assert set(labels).issubset(set(range(3)))

    def test_fit_with_y_ignored(self):
        X, y = _classification_data()
        km = GPUAcceleratedKMeans(n_clusters=3, random_state=0)
        km.fit(X, y)
        assert km.predict(X).shape == (N_SAMPLES,)

    def test_predict_proba_shape(self):
        X, _ = _classification_data()
        km = GPUAcceleratedKMeans(n_clusters=3, random_state=0)
        km.fit(X)
        proba = km.predict_proba(X)
        assert proba.shape == (N_SAMPLES, 3)

    def test_predict_proba_sums_to_one(self):
        X, _ = _classification_data()
        km = GPUAcceleratedKMeans(n_clusters=3, random_state=0)
        km.fit(X)
        proba = km.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(N_SAMPLES), atol=1e-6)

    def test_predict_proba_non_negative(self):
        X, _ = _classification_data()
        km = GPUAcceleratedKMeans(n_clusters=3, random_state=0)
        km.fit(X)
        proba = km.predict_proba(X)
        assert (proba >= 0).all()

    def test_extra_kwargs_ignored(self):
        km = GPUAcceleratedKMeans(n_clusters=2, random_state=0, unused=True)
        X, _ = _classification_data()
        km.fit(X)
        assert km.predict(X).shape == (N_SAMPLES,)

    def test_fit_returns_self(self):
        X, _ = _classification_data()
        km = GPUAcceleratedKMeans(n_clusters=2, random_state=0)
        result = km.fit(X)
        assert result is km
