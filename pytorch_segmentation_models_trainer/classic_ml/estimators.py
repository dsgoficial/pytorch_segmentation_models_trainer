# -*- coding: utf-8 -*-
"""Classic ML estimators with optional NVIDIA RAPIDS GPU acceleration.

All wrappers expose the standard sklearn estimator API
(``fit``, ``predict``, ``predict_proba``) and are instantiable via Hydra.

GPU acceleration is opt-in.  Call :func:`enable_gpu_acceleration` once at
application startup to transparently route sklearn operations through cuml
backends.  This patches sklearn **globally** for the entire process — see the
function docstring for caveats.
"""

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def enable_gpu_acceleration() -> bool:
    """Patch sklearn with cuml backends for transparent GPU acceleration.

    Uses ``cuml.accel.install()`` (NVIDIA RAPIDS "Zero Code Change") to
    replace sklearn estimators with GPU-accelerated cuml equivalents.  Must
    be called **once** before creating any estimator instances.

    Warning:
        This patches sklearn **globally** for the entire Python process.
        All subsequent sklearn calls — including k-fold splitting and
        clustering metrics in other parts of this framework — will route
        through cuml backends where supported.  Only call this when the
        entire workload should run on GPU.

    Returns:
        ``True`` if cuml.accel was successfully installed, ``False`` if cuml
        is not available (CPU fallback remains active).

    Example::

        from pytorch_segmentation_models_trainer.classic_ml.estimators import (
            enable_gpu_acceleration,
        )
        enable_gpu_acceleration()
    """
    try:
        import cuml.accel

        cuml.accel.install()
        return True
    except ImportError:
        return False


def _to_numpy(arr) -> np.ndarray:
    """Convert arr to a NumPy array for sklearn compatibility.

    Args:
        arr: A NumPy ndarray, PyTorch tensor, or CuPy array.

    Returns:
        NumPy ndarray.
    """
    if isinstance(arr, np.ndarray):
        return arr
    import torch

    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    try:
        import cupy as cp

        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except ImportError:
        pass
    return np.asarray(arr)


class GPUAcceleratedRandomForest:
    """RandomForestClassifier with transparent GPU acceleration via cuml.accel.

    Thin wrapper around ``sklearn.ensemble.RandomForestClassifier`` that
    handles input conversion and is instantiable via Hydra.  Call
    :func:`enable_gpu_acceleration` before creating instances to use cuml.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree.  ``None`` means unlimited.
        random_state: Random seed for reproducibility.
        **kwargs: Ignored extra Hydra arguments.

    Example YAML:
        classifier:
          _target_: pytorch_segmentation_models_trainer.classic_ml.estimators.GPUAcceleratedRandomForest
          n_estimators: 100
          max_depth: null
          random_state: 42
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

    def fit(self, X, y) -> "GPUAcceleratedRandomForest":
        """Fit the forest on feature matrix *X* and labels *y*.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Label vector of shape ``(n_samples,)``.

        Returns:
            ``self``.
        """
        self._model.fit(_to_numpy(X), _to_numpy(y))
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels for *X*.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted labels of shape ``(n_samples,)``.
        """
        return self._model.predict(_to_numpy(X))

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for *X*.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Probability matrix of shape ``(n_samples, n_classes)``.
        """
        return self._model.predict_proba(_to_numpy(X))


class GPUAcceleratedSVM:
    """SVC with transparent GPU acceleration via cuml.accel.

    Wraps ``sklearn.svm.SVC`` with ``probability=True`` enabled so that
    ``predict_proba`` is always available.  Call :func:`enable_gpu_acceleration`
    before creating instances to use cuml.

    Args:
        C: Regularisation parameter.
        kernel: Kernel type — ``"rbf"``, ``"linear"``, etc.
        random_state: Random seed for probability calibration.
        **kwargs: Ignored extra Hydra arguments.

    Example YAML:
        classifier:
          _target_: pytorch_segmentation_models_trainer.classic_ml.estimators.GPUAcceleratedSVM
          C: 1.0
          kernel: rbf
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self._model = SVC(
            C=C, kernel=kernel, probability=True, random_state=random_state
        )

    def fit(self, X, y) -> "GPUAcceleratedSVM":
        """Fit the SVM on feature matrix *X* and labels *y*.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Label vector of shape ``(n_samples,)``.

        Returns:
            ``self``.
        """
        self._model.fit(_to_numpy(X), _to_numpy(y))
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels for *X*.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted labels of shape ``(n_samples,)``.
        """
        return self._model.predict(_to_numpy(X))

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for *X*.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Probability matrix of shape ``(n_samples, n_classes)``.
        """
        return self._model.predict_proba(_to_numpy(X))


class GPUAcceleratedKMeans:
    """KMeans clustering with transparent GPU acceleration via cuml.accel.

    Wraps ``sklearn.cluster.KMeans`` and adds a ``predict_proba`` method
    that returns soft cluster assignments based on inverse centroid distances.
    Call :func:`enable_gpu_acceleration` before creating instances to use cuml.

    Args:
        n_clusters: Number of clusters.
        max_iter: Maximum number of K-Means iterations.
        random_state: Random seed.
        **kwargs: Ignored extra Hydra arguments.

    Example YAML:
        classifier:
          _target_: pytorch_segmentation_models_trainer.classic_ml.estimators.GPUAcceleratedKMeans
          n_clusters: 8
          max_iter: 300
          random_state: 42
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self._model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state,
            n_init="auto",
        )

    def fit(self, X, y=None) -> "GPUAcceleratedKMeans":
        """Fit K-Means on feature matrix *X*.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Ignored.  Present for API compatibility.

        Returns:
            ``self``.
        """
        self._model.fit(_to_numpy(X))
        return self

    def predict(self, X) -> np.ndarray:
        """Predict nearest cluster label for each sample.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Cluster labels of shape ``(n_samples,)``.
        """
        return self._model.predict(_to_numpy(X))

    def predict_proba(self, X) -> np.ndarray:
        """Compute soft cluster assignments based on inverse centroid distances.

        Each sample's probability for cluster *k* is proportional to
        ``1 / (distance_to_centroid_k + eps)``, then row-normalised.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Soft-assignment matrix of shape ``(n_samples, n_clusters)``.
        """
        X_np = _to_numpy(X)
        centers = self._model.cluster_centers_
        # Euclidean distances: (n_samples, n_clusters)
        diffs = X_np[:, np.newaxis, :] - centers[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=-1)
        weights = 1.0 / (distances + 1e-10)
        return weights / weights.sum(axis=1, keepdims=True)
