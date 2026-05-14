# -*- coding: utf-8 -*-
"""
Computes per-channel mean and std from a training dataset defined in a YAML
config and writes the statistics back into the same YAML file.

The dataset is instantiated without augmentations so that statistics are
computed on raw (scaled-to-[0,1]) pixel values.  The results are written as an
``albumentations.Normalize`` entry inside each dataset's ``augmentation_list``
and, optionally, as ``norm_params`` in image-visualization callbacks.
"""

import importlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from ruamel.yaml import YAML

_DTYPE_MAX_PIXEL_VALUE: Dict[str, float] = {
    "uint8": 255.0,
    "uint16": 65535.0,
    "float32": 1.0,
    "native": 1.0,
}

_DATASET_KEYS = ("train_dataset", "val_dataset", "test_dataset")

_IMAGE_CALLBACK_MODULE_PATTERNS = (
    "custom_callbacks.image_callbacks",
    "custom_callbacks.edl_callbacks",
)

_HYDRA_REF_RE = re.compile(r"\$\{[^}]+\}")


def _strip_hydra_refs(d: dict) -> dict:
    """Return a copy of *d* with keys whose values contain ``${...}`` removed."""
    result = {}
    for k, v in d.items():
        if isinstance(v, str) and _HYDRA_REF_RE.search(v):
            continue
        elif isinstance(v, dict):
            result[k] = _strip_hydra_refs(v)
        elif isinstance(v, list):
            cleaned = []
            for item in v:
                if isinstance(item, str) and _HYDRA_REF_RE.search(item):
                    continue
                cleaned.append(
                    _strip_hydra_refs(item) if isinstance(item, dict) else item
                )
            result[k] = cleaned
        else:
            result[k] = v
    return result


def _import_class(target: str):
    """Import and return a class given its fully-qualified dotted name."""
    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def instantiate_dataset_for_stats(dataset_cfg: dict):
    """Instantiate the dataset class without augmentations or data_loader config.

    Args:
        dataset_cfg: Raw dict from the YAML ``train_dataset`` (or similar) key.

    Returns:
        An instantiated dataset object.
    """
    cfg = dict(dataset_cfg)
    target = cfg.pop("_target_")
    cfg.pop("data_loader", None)
    cfg.pop("augmentation_list", None)
    cfg.pop("corruption_augmentation_list", None)
    cfg = _strip_hydra_refs(cfg)
    cls = _import_class(target)
    return cls(**cfg)


def compute_stats(
    dataset,
    batch_size: int = 64,
    num_workers: int = 4,
    progress: bool = True,
) -> Tuple[List[float], List[float]]:
    """Compute per-channel mean and std by streaming through *dataset*.

    Args:
        dataset: A torch Dataset whose ``__getitem__`` returns a dict with an
            ``"image"`` key holding a ``(C, H, W)`` float tensor in [0, 1].
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.
        progress: Show tqdm progress bar.

    Returns:
        Tuple of (mean, std) as lists of floats rounded to 6 decimal places.

    Raises:
        RuntimeError: If the dataset yields no samples.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    iterable = loader
    if progress:
        try:
            from tqdm import tqdm

            iterable = tqdm(loader, desc="Computing stats", unit="batch")
        except ImportError:
            pass

    channel_sum: Optional[torch.Tensor] = None
    channel_sq_sum: Optional[torch.Tensor] = None
    n_pixels = 0

    for batch in iterable:
        images = batch["image"].float()

        if images.ndim == 3:
            images = images.unsqueeze(0)

        B, C, H, W = images.shape
        if channel_sum is None:
            channel_sum = torch.zeros(C)
            channel_sq_sum = torch.zeros(C)

        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sq_sum += (images**2).sum(dim=[0, 2, 3])
        n_pixels += B * H * W

    if n_pixels == 0 or channel_sum is None:
        raise RuntimeError("Dataset is empty — cannot compute statistics.")

    mean = channel_sum / n_pixels
    std = (channel_sq_sum / n_pixels - mean**2).clamp(min=0) ** 0.5
    return [round(v, 6) for v in mean.tolist()], [round(v, 6) for v in std.tolist()]


def _is_image_callback(target: str) -> bool:
    """Return True if *target* belongs to a known image-visualization callback module."""
    return any(pattern in target for pattern in _IMAGE_CALLBACK_MODULE_PATTERNS)


def _make_normalize_entry(
    mean: List[float], std: List[float], max_pixel_value: float
) -> dict:
    return {
        "_target_": "albumentations.Normalize",
        "mean": mean,
        "std": std,
        "max_pixel_value": max_pixel_value,
    }


def _make_to_tensor_entry() -> dict:
    return {"_target_": "albumentations.pytorch.ToTensorV2"}


def update_dataset_augmentation_list(
    dataset_cfg,
    mean: List[float],
    std: List[float],
    max_pixel_value: float,
) -> None:
    """Upsert ``albumentations.Normalize`` and ``ToTensorV2`` in *dataset_cfg* in-place.

    Args:
        dataset_cfg: Mutable mapping (e.g. ruamel CommentedMap) for one dataset key.
        mean: Per-channel mean values in [0, 1].
        std: Per-channel std values in [0, 1].
        max_pixel_value: Passed to ``albumentations.Normalize``
            (255.0 for uint8, 65535.0 for uint16, 1.0 for float32/native).
    """
    normalize_entry = _make_normalize_entry(mean, std, max_pixel_value)
    to_tensor_entry = _make_to_tensor_entry()

    aug_list = dataset_cfg.get("augmentation_list")
    if aug_list is None:
        dataset_cfg["augmentation_list"] = [normalize_entry, to_tensor_entry]
        return

    normalize_idx: Optional[int] = None
    has_to_tensor = False
    for i, entry in enumerate(aug_list):
        entry_target = entry.get("_target_", "") if isinstance(entry, dict) else ""
        if "Normalize" in entry_target:
            normalize_idx = i
        if "ToTensorV2" in entry_target:
            has_to_tensor = True

    if normalize_idx is not None:
        aug_list[normalize_idx] = normalize_entry
    else:
        aug_list.insert(0, normalize_entry)

    if not has_to_tensor:
        aug_list.append(to_tensor_entry)


def update_normalization_parameters(
    yaml_data: dict, mean: List[float], std: List[float]
) -> None:
    """Write top-level ``normalization_parameters`` key with mean and std arrays.

    Args:
        yaml_data: Mutable mapping of the full parsed YAML.
        mean: Per-channel mean values.
        std: Per-channel std values.
    """
    yaml_data["normalization_parameters"] = {"mean": mean, "std": std}


def update_callbacks(yaml_data, mean, std) -> int:
    """Write ``norm_params`` into every image callback in *yaml_data*.

    Args:
        yaml_data: Mutable mapping of the full parsed YAML.
        mean: Per-channel mean values or Hydra interpolation string.
        std: Per-channel std values or Hydra interpolation string.

    Returns:
        Number of callbacks updated.
    """
    callbacks = yaml_data.get("callbacks") or []
    updated = 0
    for cb in callbacks:
        if not isinstance(cb, dict):
            continue
        if _is_image_callback(cb.get("_target_", "")):
            cb["normalized_input"] = True
            cb["norm_params"] = {"mean": mean, "std": std}
            updated += 1
    return updated


def process_yaml(
    yaml_path: Path,
    dataset_key: str = "train_dataset",
    batch_size: int = 64,
    num_workers: int = 4,
    dry_run: bool = False,
    skip_callbacks: bool = False,
    progress: bool = True,
) -> Tuple[List[float], List[float]]:
    """Full pipeline: load YAML → instantiate dataset → compute stats → update YAML → save.

    Args:
        yaml_path: Path to the YAML config file to read and (optionally) update.
        dataset_key: Top-level YAML key for the dataset used to compute statistics.
        batch_size: DataLoader batch size during stats computation.
        num_workers: DataLoader worker count.
        dry_run: If True, print changes but do not write to disk.
        skip_callbacks: If True, skip updating image callbacks with norm_params.
        progress: Print progress information.

    Returns:
        Tuple of (mean, std) as lists of floats.

    Raises:
        KeyError: If *dataset_key* is not present in the YAML.
    """
    ryaml = YAML()
    ryaml.preserve_quotes = True
    with open(yaml_path) as fh:
        yaml_data = ryaml.load(fh)

    if dataset_key not in yaml_data:
        raise KeyError(f"Key '{dataset_key}' not found in {yaml_path}")

    dataset_cfg = yaml_data[dataset_key]
    image_dtype = dataset_cfg.get("image_dtype", "uint8")
    max_pixel_value = _DTYPE_MAX_PIXEL_VALUE.get(image_dtype, 1.0)

    dataset = instantiate_dataset_for_stats(dict(dataset_cfg))
    n_total = len(dataset)

    if progress:
        print(f"Computing stats from {dataset_key}...")
        print(f"  → {n_total} samples found, dtype={image_dtype}")

    mean, std = compute_stats(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        progress=progress,
    )

    if progress:
        print(f"\n  mean: {[round(v, 4) for v in mean]}")
        print(f"  std:  {[round(v, 4) for v in std]}")
        print()

    update_normalization_parameters(yaml_data, mean, std)
    if progress:
        print("  ✓ normalization_parameters → written")

    mean_ref = "${normalization_parameters.mean}"
    std_ref = "${normalization_parameters.std}"

    keys_to_update = [k for k in _DATASET_KEYS if k in yaml_data]
    for key in keys_to_update:
        update_dataset_augmentation_list(
            yaml_data[key], mean_ref, std_ref, max_pixel_value
        )
        if progress:
            print(f"  ✓ {key}.augmentation_list → updated")

    if not skip_callbacks:
        n_updated = update_callbacks(yaml_data, mean_ref, std_ref)
        if progress and n_updated:
            print(f"  ✓ {n_updated} image callback(s) → norm_params updated")

    if dry_run:
        if progress:
            print("\n[dry-run] File not modified.")
    else:
        with open(yaml_path, "w") as fh:
            ryaml.dump(yaml_data, fh)
        if progress:
            print(f"\nSaved: {yaml_path}")

    return mean, std
