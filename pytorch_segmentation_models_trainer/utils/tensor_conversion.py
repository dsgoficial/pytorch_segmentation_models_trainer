# -*- coding: utf-8 -*-
"""Utilities for converting arrays between PyTorch, NumPy, and CuPy.

CuPy operations use zero-copy GPU memory sharing where possible.
"""

from typing import Union

import numpy as np
import torch


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy array on CPU.

    Args:
        tensor: Any PyTorch tensor (CPU or GPU).

    Returns:
        A NumPy ndarray on CPU with the same dtype.

    Example:
        >>> import torch
        >>> t = torch.tensor([1.0, 2.0])
        >>> tensor_to_numpy(t)
        array([1., 2.], dtype=float32)
    """
    return tensor.detach().cpu().numpy()


def numpy_to_tensor(
    arr: np.ndarray,
    device: Union[str, torch.device, None] = None,
) -> torch.Tensor:
    """Convert a NumPy array to a PyTorch tensor.

    Args:
        arr: A NumPy ndarray.
        device: Target device string or ``torch.device``.  When ``None``,
            the tensor is created on CPU.

    Returns:
        A PyTorch tensor.

    Example:
        >>> import numpy as np
        >>> numpy_to_tensor(np.zeros((3, 4), dtype=np.float32)).shape
        torch.Size([3, 4])
    """
    tensor = torch.from_numpy(np.asarray(arr))
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def tensor_to_cupy(tensor: torch.Tensor):
    """Convert a CUDA PyTorch tensor to a CuPy array (zero-copy).

    The returned CuPy array shares the same GPU memory as the input tensor.
    No data is copied.

    Args:
        tensor: A PyTorch tensor that lives on a CUDA device.

    Returns:
        A CuPy ndarray pointing to the same GPU memory.

    Raises:
        ImportError: If ``cupy`` is not installed.
        ValueError: If the tensor is not on a CUDA device.

    Example YAML:
        # No YAML config — this is a pure utility function.
    """
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError(
            "cupy is required for tensor_to_cupy. "
            "Install the GPU extras: pip install pytorch-segmentation-models-trainer[gpu-ml]"
        ) from exc
    if not tensor.is_cuda:
        raise ValueError(
            "tensor_to_cupy requires a CUDA tensor. "
            f"Got tensor on device '{tensor.device}'."
        )
    return cp.asarray(tensor)


def cupy_to_tensor(arr) -> torch.Tensor:
    """Convert a CuPy array to a PyTorch CUDA tensor (zero-copy on same device).

    The returned tensor shares the same GPU memory as the input CuPy array.

    Args:
        arr: A CuPy ndarray.

    Returns:
        A PyTorch CUDA tensor on the same device as ``arr``.

    Raises:
        ImportError: If ``cupy`` is not installed.

    Example YAML:
        # No YAML config — this is a pure utility function.
    """
    try:
        import cupy as cp  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "cupy is required for cupy_to_tensor. "
            "Install the GPU extras: pip install pytorch-segmentation-models-trainer[gpu-ml]"
        ) from exc
    return torch.as_tensor(arr, device=f"cuda:{arr.device.id}")


def ensure_numpy(arr) -> np.ndarray:
    """Convert any supported array type to a NumPy array on CPU.

    Handles ``np.ndarray``, ``torch.Tensor``, and ``cupy.ndarray``.

    Args:
        arr: A NumPy ndarray, PyTorch tensor, or CuPy array.

    Returns:
        A NumPy ndarray on CPU.

    Raises:
        TypeError: If ``arr`` is not a supported array type.

    Example:
        >>> import numpy as np, torch
        >>> ensure_numpy(torch.ones(3)).tolist()
        [1.0, 1.0, 1.0]
    """
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, torch.Tensor):
        return tensor_to_numpy(arr)
    try:
        import cupy as cp

        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except ImportError:
        pass
    raise TypeError(
        f"Cannot convert {type(arr).__name__} to numpy. "
        "Supported types: np.ndarray, torch.Tensor, cupy.ndarray."
    )
