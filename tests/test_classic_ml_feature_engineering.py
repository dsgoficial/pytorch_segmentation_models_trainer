# -*- coding: utf-8 -*-
"""Tests for pytorch_segmentation_models_trainer.classic_ml.feature_engineering."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np

from pytorch_segmentation_models_trainer.classic_ml.feature_engineering import (
    FeatureEngineeringPipeline,
    GaborFilterExtractor,
    GradientExtractor,
    MultiscaleExtractor,
    _gabor_filter,
    _gaussian_filter,
    _get_xp,
    _sobel_filter,
    _sobel_h_filter,
    _sobel_v_filter,
    _split_channels,
    _to_numpy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 8, 8


def _grey_image():
    return np.random.rand(H, W).astype(np.float32)


def _rgb_image():
    return np.random.rand(H, W, 3).astype(np.float32)


def _make_mock_cupy(array):
    """Build a mock cupy module that treats *array* as a cupy array."""

    class FakeCupyArray:
        pass

    mock_cp = MagicMock()
    mock_cp.ndarray = FakeCupyArray
    # get_array_module returns the mock cp for FakeCupyArray instances
    mock_cp.get_array_module.side_effect = lambda arr: (
        mock_cp if isinstance(arr, FakeCupyArray) else __import__("numpy")
    )
    mock_cp.asnumpy.return_value = array
    return mock_cp, FakeCupyArray


# ---------------------------------------------------------------------------
# _get_xp
# ---------------------------------------------------------------------------


class TestGetXp:
    def test_returns_numpy_when_cupy_missing(self):
        with patch.dict(sys.modules, {"cupy": None}):
            xp = _get_xp(np.zeros(3))
        assert xp is np

    def test_returns_numpy_for_numpy_array(self):
        mock_cp = MagicMock()
        mock_cp.get_array_module.return_value = np
        with patch.dict(sys.modules, {"cupy": mock_cp}):
            xp = _get_xp(np.zeros(3))
        assert xp is np

    def test_returns_cupy_for_cupy_array(self):
        mock_cp = MagicMock()
        mock_cp.get_array_module.return_value = mock_cp
        with patch.dict(sys.modules, {"cupy": mock_cp}):
            xp = _get_xp(MagicMock())
        assert xp is mock_cp


# ---------------------------------------------------------------------------
# _to_numpy
# ---------------------------------------------------------------------------


class TestToNumpy:
    def test_numpy_passthrough(self):
        arr = np.array([1.0, 2.0])
        result = _to_numpy(arr)
        assert isinstance(result, np.ndarray)

    def test_cupy_array_converted(self):
        image = _grey_image()
        mock_cp, FakeCupyArray = _make_mock_cupy(image)
        fake_arr = FakeCupyArray()
        with patch.dict(sys.modules, {"cupy": mock_cp}):
            result = _to_numpy(fake_arr)
        mock_cp.asnumpy.assert_called_once_with(fake_arr)
        np.testing.assert_array_equal(result, image)


# ---------------------------------------------------------------------------
# _gabor_filter
# ---------------------------------------------------------------------------


class TestGaborFilter:
    def test_cpu_path_shape(self):
        image = _grey_image()
        real, imag = _gabor_filter(image, frequency=0.1, theta=0.0)
        assert real.shape == (H, W)
        assert imag.shape == (H, W)

    def test_cpu_path_returns_numpy(self):
        image = _grey_image()
        real, imag = _gabor_filter(image, frequency=0.1, theta=0.0)
        assert isinstance(real, np.ndarray)
        assert isinstance(imag, np.ndarray)

    def test_gpu_path_uses_cucim(self):
        image = _grey_image()
        mock_cp, FakeCupyArray = _make_mock_cupy(image)
        fake_arr = FakeCupyArray()
        mock_cp.asnumpy.return_value = image

        real_out = np.zeros((H, W), dtype=np.float32)
        imag_out = np.zeros((H, W), dtype=np.float32)

        mock_cucim_filters = MagicMock()
        mock_cucim_filters.gabor.return_value = (real_out, imag_out)

        with patch.dict(
            sys.modules,
            {
                "cupy": mock_cp,
                "cucim": MagicMock(),
                "cucim.skimage": MagicMock(),
                "cucim.skimage.filters": mock_cucim_filters,
            },
        ):
            real, imag = _gabor_filter(fake_arr, frequency=0.1, theta=0.0)

        mock_cucim_filters.gabor.assert_called_once()
        assert isinstance(real, np.ndarray)

    def test_gpu_path_falls_back_when_cucim_missing(self):
        """GPU array + no cucim → falls back to skimage on CPU."""
        image = _grey_image()
        mock_cp, FakeCupyArray = _make_mock_cupy(image)
        fake_arr = FakeCupyArray()
        mock_cp.asnumpy.return_value = image

        with patch.dict(
            sys.modules,
            {
                "cupy": mock_cp,
                "cucim": None,
                "cucim.skimage": None,
                "cucim.skimage.filters": None,
            },
        ):
            real, imag = _gabor_filter(fake_arr, frequency=0.1, theta=0.0)

        assert isinstance(real, np.ndarray)
        assert real.shape == (H, W)


# ---------------------------------------------------------------------------
# _gaussian_filter
# ---------------------------------------------------------------------------


class TestGaussianFilter:
    def test_cpu_path_shape(self):
        image = _grey_image()
        result = _gaussian_filter(image, sigma=1.0)
        assert result.shape == (H, W)
        assert isinstance(result, np.ndarray)

    def test_gpu_path_uses_cucim(self):
        image = _grey_image()
        mock_cp, FakeCupyArray = _make_mock_cupy(image)
        fake_arr = FakeCupyArray()
        smoothed = np.zeros((H, W), dtype=np.float32)
        mock_cp.asnumpy.return_value = smoothed

        mock_cucim_filters = MagicMock()
        mock_cucim_filters.gaussian.return_value = smoothed

        with patch.dict(
            sys.modules,
            {
                "cupy": mock_cp,
                "cucim": MagicMock(),
                "cucim.skimage": MagicMock(),
                "cucim.skimage.filters": mock_cucim_filters,
            },
        ):
            result = _gaussian_filter(fake_arr, sigma=1.0)

        mock_cucim_filters.gaussian.assert_called_once()
        assert isinstance(result, np.ndarray)

    def test_gpu_fallback_when_cucim_missing(self):
        image = _grey_image()
        mock_cp, FakeCupyArray = _make_mock_cupy(image)
        fake_arr = FakeCupyArray()
        mock_cp.asnumpy.return_value = image

        with patch.dict(
            sys.modules,
            {
                "cupy": mock_cp,
                "cucim": None,
                "cucim.skimage": None,
                "cucim.skimage.filters": None,
            },
        ):
            result = _gaussian_filter(fake_arr, sigma=1.0)

        assert isinstance(result, np.ndarray)
        assert result.shape == (H, W)


# ---------------------------------------------------------------------------
# _sobel_filter / _sobel_h_filter / _sobel_v_filter
# ---------------------------------------------------------------------------


class TestSobelFilters:
    def test_sobel_cpu_shape(self):
        image = _grey_image()
        assert _sobel_filter(image).shape == (H, W)

    def test_sobel_h_cpu_shape(self):
        image = _grey_image()
        assert _sobel_h_filter(image).shape == (H, W)

    def test_sobel_v_cpu_shape(self):
        image = _grey_image()
        assert _sobel_v_filter(image).shape == (H, W)

    def _gpu_test(self, filter_fn, cucim_fn_name):
        image = _grey_image()
        mock_cp, FakeCupyArray = _make_mock_cupy(image)
        fake_arr = FakeCupyArray()
        out = np.zeros((H, W), dtype=np.float32)
        mock_cp.asnumpy.return_value = out

        mock_cucim_filters = MagicMock()
        getattr(mock_cucim_filters, cucim_fn_name).return_value = out

        with patch.dict(
            sys.modules,
            {
                "cupy": mock_cp,
                "cucim": MagicMock(),
                "cucim.skimage": MagicMock(),
                "cucim.skimage.filters": mock_cucim_filters,
            },
        ):
            result = filter_fn(fake_arr)

        getattr(mock_cucim_filters, cucim_fn_name).assert_called_once()
        assert isinstance(result, np.ndarray)

    def _gpu_fallback_test(self, filter_fn):
        image = _grey_image()
        mock_cp, FakeCupyArray = _make_mock_cupy(image)
        fake_arr = FakeCupyArray()
        mock_cp.asnumpy.return_value = image

        with patch.dict(
            sys.modules,
            {
                "cupy": mock_cp,
                "cucim": None,
                "cucim.skimage": None,
                "cucim.skimage.filters": None,
            },
        ):
            result = filter_fn(fake_arr)

        assert isinstance(result, np.ndarray)
        assert result.shape == (H, W)

    def test_sobel_gpu_path(self):
        self._gpu_test(_sobel_filter, "sobel")

    def test_sobel_h_gpu_path(self):
        self._gpu_test(_sobel_h_filter, "sobel_h")

    def test_sobel_v_gpu_path(self):
        self._gpu_test(_sobel_v_filter, "sobel_v")

    def test_sobel_gpu_fallback(self):
        self._gpu_fallback_test(_sobel_filter)

    def test_sobel_h_gpu_fallback(self):
        self._gpu_fallback_test(_sobel_h_filter)

    def test_sobel_v_gpu_fallback(self):
        self._gpu_fallback_test(_sobel_v_filter)


# ---------------------------------------------------------------------------
# _split_channels
# ---------------------------------------------------------------------------


class TestSplitChannels:
    def test_2d_image_returns_single_channel(self):
        image = _grey_image()
        channels = _split_channels(image)
        assert len(channels) == 1
        assert channels[0].shape == (H, W)

    def test_rgb_image_returns_three_channels(self):
        image = _rgb_image()
        channels = _split_channels(image)
        assert len(channels) == 3
        for c in channels:
            assert c.shape == (H, W)


# ---------------------------------------------------------------------------
# GaborFilterExtractor
# ---------------------------------------------------------------------------


class TestGaborFilterExtractor:
    def test_output_shape_grayscale(self):
        extractor = GaborFilterExtractor(frequencies=[0.1, 0.25], num_orientations=2)
        # 1 channel × 2 frequencies × 2 orientations = 4 features
        result = extractor.extract(_grey_image())
        assert result.shape == (H, W, 4)

    def test_output_shape_rgb(self):
        extractor = GaborFilterExtractor(frequencies=[0.1], num_orientations=2)
        # 3 channels × 1 frequency × 2 orientations = 6 features
        result = extractor.extract(_rgb_image())
        assert result.shape == (H, W, 6)

    def test_output_dtype_is_float(self):
        extractor = GaborFilterExtractor(frequencies=[0.1], num_orientations=1)
        result = extractor.extract(_grey_image())
        assert np.issubdtype(result.dtype, np.floating)

    def test_extra_kwargs_ignored(self):
        extractor = GaborFilterExtractor(
            frequencies=[0.1], num_orientations=1, foo="bar"
        )
        assert extractor.extract(_grey_image()).shape[0] == H


# ---------------------------------------------------------------------------
# GradientExtractor
# ---------------------------------------------------------------------------


class TestGradientExtractor:
    def test_output_shape_grayscale(self):
        extractor = GradientExtractor()
        # 1 channel × 3 gradient features = 3
        result = extractor.extract(_grey_image())
        assert result.shape == (H, W, 3)

    def test_output_shape_rgb(self):
        extractor = GradientExtractor()
        # 3 channels × 3 features = 9
        result = extractor.extract(_rgb_image())
        assert result.shape == (H, W, 9)

    def test_extra_kwargs_ignored(self):
        extractor = GradientExtractor(unused_param=True)
        result = extractor.extract(_grey_image())
        assert result.ndim == 3


# ---------------------------------------------------------------------------
# MultiscaleExtractor
# ---------------------------------------------------------------------------


class TestMultiscaleExtractor:
    def test_output_shape_grayscale(self):
        extractor = MultiscaleExtractor(sigmas=[1.0, 2.0])
        # 1 channel × 2 sigmas = 2 features
        result = extractor.extract(_grey_image())
        assert result.shape == (H, W, 2)

    def test_output_shape_rgb(self):
        extractor = MultiscaleExtractor(sigmas=[1.0, 2.0, 4.0])
        # 3 channels × 3 sigmas = 9
        result = extractor.extract(_rgb_image())
        assert result.shape == (H, W, 9)

    def test_extra_kwargs_ignored(self):
        extractor = MultiscaleExtractor(sigmas=[1.0], extra=True)
        result = extractor.extract(_grey_image())
        assert result.shape == (H, W, 1)


# ---------------------------------------------------------------------------
# FeatureEngineeringPipeline
# ---------------------------------------------------------------------------


class TestFeatureEngineeringPipeline:
    def _pipeline(self):
        return FeatureEngineeringPipeline(
            extractors=[
                GaborFilterExtractor(frequencies=[0.1], num_orientations=1),
                GradientExtractor(),
                MultiscaleExtractor(sigmas=[1.0]),
            ]
        )

    def test_transform_shape_grayscale(self):
        pipeline = self._pipeline()
        result = pipeline.transform(_grey_image())
        # pixels × (1*1*1 + 3 + 1) = H*W × 5
        assert result.shape == (H * W, 5)

    def test_transform_shape_rgb(self):
        pipeline = self._pipeline()
        result = pipeline.transform(_rgb_image())
        # 3 channels: (3*1*1 + 9 + 3) = H*W × 15
        assert result.shape == (H * W, 15)

    def test_transform_batch(self):
        pipeline = self._pipeline()
        images = [_grey_image(), _grey_image()]
        result = pipeline.transform_batch(images)
        assert result.shape == (2 * H * W, 5)

    def test_transform_batch_single(self):
        pipeline = self._pipeline()
        result = pipeline.transform_batch([_grey_image()])
        assert result.shape == (H * W, 5)

    def test_extra_kwargs_ignored(self):
        pipeline = FeatureEngineeringPipeline(
            extractors=[GradientExtractor()], unused=True
        )
        result = pipeline.transform(_grey_image())
        assert result.ndim == 2
