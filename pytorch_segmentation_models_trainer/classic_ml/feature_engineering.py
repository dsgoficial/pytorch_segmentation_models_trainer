# -*- coding: utf-8 -*-
"""Feature engineering pipeline for classic ML segmentation.

Provides GPU-accelerated feature extractors via ``cucim``/``cupy`` with
automatic fallback to ``scikit-image``/``numpy`` on CPU-only machines.

All extractors accept arrays of shape ``(H, W)`` or ``(H, W, C)`` — either
``numpy.ndarray`` or ``cupy.ndarray`` — and always return ``numpy.ndarray``
feature maps of shape ``(H, W, n_features)``.  The
:class:`FeatureEngineeringPipeline` concatenates extractor outputs and
flattens to ``(H*W, total_features)`` for direct ingestion by sklearn-compatible
classifiers.
"""

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np
from skimage import filters as sk_filters

# ---------------------------------------------------------------------------
# Backend helpers (GPU / CPU dispatch)
# ---------------------------------------------------------------------------


def _get_xp(arr):
    """Return the array module (``cupy`` or ``numpy``) for *arr*.

    Falls back to ``numpy`` when ``cupy`` is not installed.

    Args:
        arr: Any array-like object.

    Returns:
        The ``cupy`` module if *arr* is a CuPy array and cupy is available,
        otherwise the ``numpy`` module.
    """
    try:
        import cupy as cp

        return cp.get_array_module(arr)
    except ImportError:
        return np


def _to_numpy(arr) -> np.ndarray:
    """Ensure *arr* is a NumPy array, copying from GPU memory if needed.

    Args:
        arr: A NumPy or CuPy array.

    Returns:
        A NumPy ndarray.
    """
    xp = _get_xp(arr)
    if xp is np:
        return np.asarray(arr)
    return xp.asnumpy(arr)


def _gabor_filter(image, frequency: float, theta: float):
    """Apply a Gabor filter using cucim on GPU arrays or skimage on CPU.

    Args:
        image: 2-D array (H, W).
        frequency: Spatial frequency of the harmonic function.
        theta: Orientation in radians.

    Returns:
        Tuple ``(real, imag)`` of NumPy arrays with the same shape as *image*.
    """
    xp = _get_xp(image)
    if xp is not np:
        try:
            from cucim.skimage.filters import gabor as _cucim_gabor

            real, imag = _cucim_gabor(image, frequency=frequency, theta=theta)
            return _to_numpy(real), _to_numpy(imag)
        except ImportError:
            image = _to_numpy(image)
    from skimage.filters import gabor as _sk_gabor

    real, imag = _sk_gabor(image, frequency=frequency, theta=theta)
    return np.asarray(real), np.asarray(imag)


def _gaussian_filter(image, sigma: float) -> np.ndarray:
    """Apply a Gaussian filter using cucim on GPU arrays or skimage on CPU.

    Args:
        image: 2-D array (H, W).
        sigma: Standard deviation for the Gaussian kernel.

    Returns:
        Filtered NumPy array with the same shape as *image*.
    """
    xp = _get_xp(image)
    if xp is not np:
        try:
            from cucim.skimage.filters import gaussian as _cucim_gaussian

            return _to_numpy(_cucim_gaussian(image, sigma=sigma))
        except ImportError:
            image = _to_numpy(image)
    from skimage.filters import gaussian as _sk_gaussian

    return np.asarray(_sk_gaussian(image, sigma=sigma))


def _sobel_filter(image) -> np.ndarray:
    """Apply Sobel edge detection using cucim on GPU arrays or skimage on CPU.

    Args:
        image: 2-D array (H, W).

    Returns:
        Edge magnitude NumPy array with the same shape as *image*.
    """
    xp = _get_xp(image)
    if xp is not np:
        try:
            from cucim.skimage.filters import sobel as _cucim_sobel

            return _to_numpy(_cucim_sobel(image))
        except ImportError:
            image = _to_numpy(image)
    return np.asarray(sk_filters.sobel(image))


def _sobel_h_filter(image) -> np.ndarray:
    """Apply horizontal Sobel filter using cucim or skimage.

    Args:
        image: 2-D array (H, W).

    Returns:
        Horizontal gradient NumPy array.
    """
    xp = _get_xp(image)
    if xp is not np:
        try:
            from cucim.skimage.filters import sobel_h as _cucim_sobel_h

            return _to_numpy(_cucim_sobel_h(image))
        except ImportError:
            image = _to_numpy(image)
    return np.asarray(sk_filters.sobel_h(image))


def _sobel_v_filter(image) -> np.ndarray:
    """Apply vertical Sobel filter using cucim or skimage.

    Args:
        image: 2-D array (H, W).

    Returns:
        Vertical gradient NumPy array.
    """
    xp = _get_xp(image)
    if xp is not np:
        try:
            from cucim.skimage.filters import sobel_v as _cucim_sobel_v

            return _to_numpy(_cucim_sobel_v(image))
        except ImportError:
            image = _to_numpy(image)
    return np.asarray(sk_filters.sobel_v(image))


# ---------------------------------------------------------------------------
# Base extractor
# ---------------------------------------------------------------------------


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors.

    Subclasses implement :meth:`extract`, which accepts a ``(H, W)`` or
    ``(H, W, C)`` array and returns a ``(H, W, n_features)`` NumPy array.
    """

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract features from *image*.

        Args:
            image: Input image of shape ``(H, W)`` or ``(H, W, C)``.

        Returns:
            Feature map of shape ``(H, W, n_features)``.
        """


# ---------------------------------------------------------------------------
# Concrete extractors
# ---------------------------------------------------------------------------


class GaborFilterExtractor(BaseFeatureExtractor):
    """Extract Gabor filter responses at multiple frequencies and orientations.

    For multi-channel images each channel is filtered independently and the
    responses are stacked.

    Args:
        frequencies: Sequence of spatial frequencies for the Gabor kernel.
        num_orientations: Number of equally-spaced orientations in [0, π).
        **kwargs: Ignored extra Hydra arguments.

    Returns:
        Feature map of shape ``(H, W, n_channels * len(frequencies) * num_orientations)``.

    Example YAML:
        feature_engineering:
          extractors:
            - _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.GaborFilterExtractor
              frequencies: [0.1, 0.25, 0.4]
              num_orientations: 4
    """

    def __init__(
        self,
        frequencies: Sequence[float] = (0.1, 0.25, 0.4),
        num_orientations: int = 4,
        **kwargs,
    ):
        self.frequencies = list(frequencies)
        self.thetas = [i * np.pi / num_orientations for i in range(num_orientations)]

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract Gabor responses.

        Args:
            image: Input image of shape ``(H, W)`` or ``(H, W, C)``.

        Returns:
            Feature map of shape ``(H, W, n_features)``.
        """
        channels = _split_channels(image)
        features: List[np.ndarray] = []
        for channel in channels:
            for freq in self.frequencies:
                for theta in self.thetas:
                    real, imag = _gabor_filter(channel, frequency=freq, theta=theta)
                    magnitude = np.sqrt(real**2 + imag**2)
                    features.append(magnitude)
        return np.stack(features, axis=-1)


class GradientExtractor(BaseFeatureExtractor):
    """Extract gradient features using Sobel operators.

    Computes horizontal gradient, vertical gradient, and magnitude for each
    channel.

    Args:
        **kwargs: Ignored extra Hydra arguments.

    Returns:
        Feature map of shape ``(H, W, n_channels * 3)``.

    Example YAML:
        feature_engineering:
          extractors:
            - _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.GradientExtractor
    """

    def __init__(self, **kwargs):
        pass

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract gradient features.

        Args:
            image: Input image of shape ``(H, W)`` or ``(H, W, C)``.

        Returns:
            Feature map of shape ``(H, W, n_channels * 3)``.
        """
        channels = _split_channels(image)
        features: List[np.ndarray] = []
        for channel in channels:
            grad_h = _sobel_h_filter(channel)
            grad_v = _sobel_v_filter(channel)
            magnitude = _sobel_filter(channel)
            features.extend([grad_h, grad_v, magnitude])
        return np.stack(features, axis=-1)


class MultiscaleExtractor(BaseFeatureExtractor):
    """Extract multi-scale Gaussian features.

    Applies Gaussian smoothing at each sigma and stacks the blurred images as
    features.  This captures local context at different spatial scales.

    Args:
        sigmas: Sequence of Gaussian standard deviations.
        **kwargs: Ignored extra Hydra arguments.

    Returns:
        Feature map of shape ``(H, W, n_channels * len(sigmas))``.

    Example YAML:
        feature_engineering:
          extractors:
            - _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.MultiscaleExtractor
              sigmas: [1.0, 2.0, 4.0, 8.0]
    """

    def __init__(self, sigmas: Sequence[float] = (1.0, 2.0, 4.0), **kwargs):
        self.sigmas = list(sigmas)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract multi-scale Gaussian features.

        Args:
            image: Input image of shape ``(H, W)`` or ``(H, W, C)``.

        Returns:
            Feature map of shape ``(H, W, n_features)``.
        """
        channels = _split_channels(image)
        features: List[np.ndarray] = []
        for channel in channels:
            for sigma in self.sigmas:
                smoothed = _gaussian_filter(channel, sigma=sigma)
                features.append(smoothed)
        return np.stack(features, axis=-1)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class FeatureEngineeringPipeline:
    """Compose multiple feature extractors into a single transform.

    Applies each extractor to the input image, concatenates the resulting
    feature maps, and flattens to a 2-D matrix suitable for sklearn
    classifiers.

    Args:
        extractors: List of :class:`BaseFeatureExtractor` instances.
        **kwargs: Ignored extra Hydra arguments.

    Example YAML:
        feature_engineering:
          _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.FeatureEngineeringPipeline
          extractors:
            - _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.GaborFilterExtractor
              frequencies: [0.1, 0.25, 0.4]
              num_orientations: 4
            - _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.GradientExtractor
            - _target_: pytorch_segmentation_models_trainer.classic_ml.feature_engineering.MultiscaleExtractor
              sigmas: [1.0, 2.0, 4.0]
    """

    def __init__(self, extractors: List[BaseFeatureExtractor], **kwargs):
        self.extractors = list(extractors)

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Transform an image into a flat feature matrix.

        Args:
            image: Input image of shape ``(H, W)`` or ``(H, W, C)``.

        Returns:
            Feature matrix of shape ``(H*W, total_features)``.
        """
        H, W = image.shape[:2]
        parts: List[np.ndarray] = [
            extractor.extract(image) for extractor in self.extractors
        ]
        combined = np.concatenate(parts, axis=-1)
        return combined.reshape(H * W, -1)

    def transform_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Transform a list of images into a concatenated feature matrix.

        Args:
            images: List of images, each of shape ``(H, W)`` or ``(H, W, C)``.

        Returns:
            Feature matrix of shape ``(sum(H_i * W_i), total_features)``.
        """
        return np.concatenate([self.transform(img) for img in images], axis=0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_channels(image: np.ndarray) -> List[np.ndarray]:
    """Return a list of 2-D single-channel arrays from *image*.

    For a ``(H, W)`` input returns ``[image]``.
    For a ``(H, W, C)`` input returns a list of ``C`` slices.

    Args:
        image: Image of shape ``(H, W)`` or ``(H, W, C)``.

    Returns:
        List of 2-D arrays.
    """
    if image.ndim == 2:
        return [image]
    return [image[:, :, c] for c in range(image.shape[2])]
