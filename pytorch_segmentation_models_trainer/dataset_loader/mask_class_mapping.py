# -*- coding: utf-8 -*-
"""On-the-fly remapping of mask class values for windowed datasets.

Lets a dataset merge, rename or ignore classes when the mask window is read,
without rewriting the mask rasters on disk.

Example YAML::

    train_dataset:
      _target_: pytorch_segmentation_models_trainer.dataset_loader.dataset.CSVWindowedSegmentationDataset
      input_csv_path: /data/train_windows.csv
      n_classes: 5
      mask_class_mapping:
        5: 3     # merge class 5 into class 3
        4: 255   # send class 4 to the ignore index
"""

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

_UINT8_MAX = 255


def _to_class_value(value: Any, role: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"mask_class_mapping {role} must be an integer, got {value!r}")
    if isinstance(value, str):
        if not value.strip().lstrip("-").isdigit():
            raise ValueError(
                f"mask_class_mapping {role} must be an integer, got {value!r}"
            )
        value = int(value)
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"mask_class_mapping {role} must be an integer, got {value!r}")
    value = int(value)
    if not 0 <= value <= _UINT8_MAX:
        raise ValueError(
            f"mask_class_mapping {role} must be in [0, {_UINT8_MAX}], got {value}"
        )
    return value


def build_mask_class_lut(mapping: Optional[Mapping]) -> Optional[np.ndarray]:
    """Build a uint8 lookup table from a ``{source: target}`` class mapping.

    Values not present in ``mapping`` are kept unchanged (identity). The
    mapping is applied in a single pass, so ``{1: 2, 2: 3}`` sends 1 to 2 and
    2 to 3 (it is not chained).

    Args:
        mapping: Dict-like ``{source_class: target_class}`` (``dict`` or
            OmegaConf ``DictConfig``). Keys and values must be integers (or
            numeric strings) in ``[0, 255]``. ``None`` or empty disables the
            remapping.

    Returns:
        ``np.ndarray`` of shape ``(256,)`` and dtype ``uint8``, or ``None``
        when there is nothing to remap.

    Raises:
        ValueError: If ``mapping`` is not dict-like or has invalid entries.

    Example YAML::

        mask_class_mapping:
          5: 3
    """
    if mapping is None:
        return None
    if not isinstance(mapping, Mapping):
        raise ValueError(
            f"mask_class_mapping must be a dict {{source: target}}, got {type(mapping).__name__}"
        )
    if len(mapping) == 0:
        return None
    lut = np.arange(_UINT8_MAX + 1, dtype=np.uint8)
    for source, target in mapping.items():
        lut[_to_class_value(source, "key")] = _to_class_value(target, "value")
    return lut


def apply_mask_class_lut(mask: np.ndarray, lut: Optional[np.ndarray]) -> np.ndarray:
    """Remap a mask array with a lookup table built by :func:`build_mask_class_lut`.

    Args:
        mask: Integer mask array of any shape with values in ``[0, 255]``.
        lut: Lookup table, or ``None`` to return ``mask`` unchanged.

    Returns:
        New ``uint8`` array with the same shape as ``mask`` (or ``mask``
        itself when ``lut`` is ``None``).

    Raises:
        ValueError: If ``mask`` has values outside ``[0, 255]``.

    Example YAML::

        mask_class_mapping:
          5: 3
    """
    if lut is None:
        return mask
    if mask.dtype != np.uint8:
        if mask.size and (mask.min() < 0 or mask.max() > _UINT8_MAX):
            raise ValueError(
                f"mask values must be in [0, {_UINT8_MAX}] to apply mask_class_mapping"
            )
        mask = mask.astype(np.uint8)
    return lut[mask]
