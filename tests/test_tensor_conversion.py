# -*- coding: utf-8 -*-
"""Tests for pytorch_segmentation_models_trainer.utils.tensor_conversion."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from pytorch_segmentation_models_trainer.utils.tensor_conversion import (
    cupy_to_tensor,
    ensure_numpy,
    numpy_to_tensor,
    tensor_to_cupy,
    tensor_to_numpy,
)

# ---------------------------------------------------------------------------
# tensor_to_numpy
# ---------------------------------------------------------------------------


class TestTensorToNumpy:
    def test_cpu_float_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = tensor_to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_preserves_shape(self):
        t = torch.zeros(3, 4, 5)
        result = tensor_to_numpy(t)
        assert result.shape == (3, 4, 5)

    def test_preserves_dtype_float32(self):
        t = torch.ones(4, dtype=torch.float32)
        result = tensor_to_numpy(t)
        assert result.dtype == np.float32

    def test_preserves_dtype_int64(self):
        t = torch.tensor([0, 1, 2], dtype=torch.int64)
        result = tensor_to_numpy(t)
        assert result.dtype == np.int64

    def test_grad_tensor_detached(self):
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        result = tensor_to_numpy(t)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# numpy_to_tensor
# ---------------------------------------------------------------------------


class TestNumpyToTensor:
    def test_basic_conversion(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = numpy_to_tensor(arr)
        assert isinstance(t, torch.Tensor)
        assert t.shape == (3,)

    def test_device_none_stays_cpu(self):
        arr = np.zeros((2, 3), dtype=np.float32)
        t = numpy_to_tensor(arr)
        assert t.device.type == "cpu"

    def test_device_string(self):
        arr = np.zeros(4, dtype=np.float32)
        t = numpy_to_tensor(arr, device="cpu")
        assert t.device.type == "cpu"

    def test_device_object(self):
        arr = np.zeros(4, dtype=np.float32)
        t = numpy_to_tensor(arr, device=torch.device("cpu"))
        assert t.device.type == "cpu"

    def test_non_contiguous_array_coerced(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        sliced = arr[::2]  # non-contiguous
        t = numpy_to_tensor(sliced)
        assert isinstance(t, torch.Tensor)

    def test_roundtrip_with_tensor_to_numpy(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = numpy_to_tensor(arr)
        result = tensor_to_numpy(t)
        np.testing.assert_array_equal(arr, result)


# ---------------------------------------------------------------------------
# tensor_to_cupy — ImportError path (cupy not installed)
# ---------------------------------------------------------------------------


class TestTensorToCupyNoCupy:
    def test_raises_import_error_when_cupy_missing(self):
        with patch.dict(sys.modules, {"cupy": None}):
            with pytest.raises(ImportError, match="cupy"):
                tensor_to_cupy(torch.zeros(3))

    def test_raises_value_error_for_cpu_tensor(self):
        mock_cupy = MagicMock()
        mock_cupy.asarray = MagicMock(return_value=MagicMock())
        with patch.dict(sys.modules, {"cupy": mock_cupy}):
            with pytest.raises(ValueError, match="CUDA"):
                tensor_to_cupy(torch.zeros(3))


class TestTensorToCupyWithCupy:
    def test_calls_cp_asarray(self):
        mock_cupy = MagicMock()
        fake_array = MagicMock()
        mock_cupy.asarray.return_value = fake_array

        # Simulate a CUDA tensor
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.is_cuda = True

        with patch.dict(sys.modules, {"cupy": mock_cupy}):
            result = tensor_to_cupy(mock_tensor)

        mock_cupy.asarray.assert_called_once_with(mock_tensor)
        assert result is fake_array


# ---------------------------------------------------------------------------
# cupy_to_tensor — ImportError path (cupy not installed)
# ---------------------------------------------------------------------------


class TestCupyToTensorNoCupy:
    def test_raises_import_error_when_cupy_missing(self):
        with patch.dict(sys.modules, {"cupy": None}):
            with pytest.raises(ImportError, match="cupy"):
                cupy_to_tensor(MagicMock())


class TestCupyToTensorWithCupy:
    def test_calls_torch_as_tensor(self):
        mock_cupy = MagicMock()
        fake_arr = MagicMock()
        fake_arr.device.id = 0

        with patch.dict(sys.modules, {"cupy": mock_cupy}):
            with patch("torch.as_tensor") as mock_as_tensor:
                fake_tensor = MagicMock()
                mock_as_tensor.return_value = fake_tensor
                result = cupy_to_tensor(fake_arr)

        mock_as_tensor.assert_called_once_with(fake_arr, device="cuda:0")
        assert result is fake_tensor


# ---------------------------------------------------------------------------
# ensure_numpy
# ---------------------------------------------------------------------------


class TestEnsureNumpy:
    def test_numpy_passthrough(self):
        arr = np.array([1, 2, 3])
        result = ensure_numpy(arr)
        assert result is arr

    def test_pytorch_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = ensure_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_cupy_array_with_cupy_available(self):
        mock_cupy = MagicMock()
        fake_arr = MagicMock()
        expected = np.array([1.0, 2.0])
        mock_cupy.asnumpy.return_value = expected

        # isinstance check needs to match the cupy.ndarray type
        mock_cupy.ndarray = type(fake_arr)

        with patch.dict(sys.modules, {"cupy": mock_cupy}):
            with patch(
                "pytorch_segmentation_models_trainer.utils.tensor_conversion.isinstance",
                side_effect=lambda obj, cls: (
                    cls is mock_cupy.ndarray or isinstance.__wrapped__(obj, cls)
                    if hasattr(isinstance, "__wrapped__")
                    else (cls is mock_cupy.ndarray)
                ),
            ):
                # Use a simpler approach: patch the cupy import path directly
                pass

        # Simpler: mock the cupy module so isinstance(arr, cp.ndarray) is True
        class FakeCupyArray:
            pass

        mock_cupy2 = MagicMock()
        mock_cupy2.ndarray = FakeCupyArray
        mock_cupy2.asnumpy = MagicMock(return_value=expected)

        fake_cupy_arr = FakeCupyArray()
        with patch.dict(sys.modules, {"cupy": mock_cupy2}):
            result = ensure_numpy(fake_cupy_arr)
        mock_cupy2.asnumpy.assert_called_once_with(fake_cupy_arr)
        np.testing.assert_array_equal(result, expected)

    def test_cupy_import_error_skipped(self):
        # When cupy raises ImportError, only numpy and tensor paths are tried
        t = torch.tensor([5.0])
        with patch.dict(sys.modules, {"cupy": None}):
            result = ensure_numpy(t)
        assert isinstance(result, np.ndarray)

    def test_unsupported_type_raises_type_error(self):
        with patch.dict(sys.modules, {"cupy": None}):
            with pytest.raises(TypeError, match="Cannot convert"):
                ensure_numpy({"not": "an array"})
