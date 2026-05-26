# -*- coding: utf-8 -*-
"""Generate train/val/test CSV splits from a soft-label manifest.

Reads the manifest produced by ``build_soft_labels.py`` (or any CSV with
``tile_id``, ``image_path``, ``p_soft_path``, optionally ``w_conf_path``)
and writes three split CSVs that ``SoftLabelDataset`` can consume directly.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Randomly split a DataFrame into train / val / test subsets.

    Args:
        df: Input DataFrame.
        train_ratio: Fraction of rows for training.
        val_ratio: Fraction of rows for validation.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to DataFrames.

    Raises:
        ValueError: if train_ratio + val_ratio >= 1.0.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1.0, "
            f"got {train_ratio} + {val_ratio} = {train_ratio + val_ratio}."
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    n_train = int(len(df) * train_ratio)
    n_val = int(len(df) * val_ratio)

    return {
        "train": df.iloc[indices[:n_train]],
        "val": df.iloc[indices[n_train : n_train + n_val]],
        "test": df.iloc[indices[n_train + n_val :]],
    }


def run(
    manifest_path: str,
    output_dir: Path,
    image_dir: Optional[str] = None,
    image_extension: str = ".tif",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Path]:
    """Generate train/val/test CSV splits from a soft-label manifest.

    Args:
        manifest_path: CSV produced by build_soft_labels or any CSV with
                       tile_id, p_soft_path, optionally w_conf_path.
        output_dir: Directory where train.csv, val.csv, test.csv are written.
        image_dir: Directory containing ``<tile_id><image_extension>`` images.
                   Ignored when the manifest already has an ``image_path`` column.
        image_extension: File extension for images in image_dir (default: .tif).
        train_ratio: Fraction of rows for training (default: 0.70).
        val_ratio: Fraction of rows for validation (default: 0.15).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Dict mapping split names (``"train"``, ``"val"``, ``"test"``) to output paths.

    Raises:
        ValueError: if the manifest has no ``image_path`` column and ``image_dir``
                    is not provided, or if train_ratio + val_ratio >= 1.0.
    """
    df = pd.read_csv(manifest_path)

    if "image_path" not in df.columns:
        if image_dir is None:
            raise ValueError(
                "Manifest has no 'image_path' column and image_dir was not provided."
            )
        img_dir = Path(image_dir)
        df["image_path"] = df["tile_id"].apply(
            lambda tid: str(img_dir / f"{tid}{image_extension}")
        )

    splits = split_dataframe(
        df, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: Dict[str, Path] = {}
    for name, split_df in splits.items():
        out_path = output_dir / f"{name}.csv"
        split_df.to_csv(out_path, index=False)
        output_paths[name] = out_path
        logger.info("Wrote %s (%d rows) → %s", name, len(split_df), out_path)

    logger.info(
        "Split complete: train=%d val=%d test=%d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return output_paths
