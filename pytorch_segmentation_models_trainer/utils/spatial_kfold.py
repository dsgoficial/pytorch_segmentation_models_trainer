# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-05-20
        copyright            : (C) 2026 by Philipe Borba
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


class SpatialKFoldSplitter:
    """Generates spatially-correct k-fold splits for sliding-window patch datasets.

    Prevents data leakage caused by pixel overlap between patches (when
    ``stride < patch_size``) and reduces spatial autocorrelation at fold
    boundaries via an optional buffer exclusion zone.

    Two split strategies are available:

    ``"by_image"``
        Uses :class:`sklearn.model_selection.GroupKFold` grouping all patches
        by their source image (``cfg.group_col``).  Every patch from a given
        image lands in the same fold, eliminating leakage from overlapping
        patches across folds.

    ``"by_spatial_region"``
        Divides the image height into ``n_splits`` horizontal bands.  For each
        fold, one band becomes the validation set and the remaining bands form
        the training set.  Patches whose footprint intersects a
        ``buffer_px``-wide zone around each band boundary are excluded from
        both train and val, reducing spatial autocorrelation at borders.

    Args:
        cfg: A :class:`~pytorch_segmentation_models_trainer.config_definitions.kfold_config.KFoldConfig`
            instance (or any object with equivalent attributes):
            ``n_splits``, ``seed``, ``split_strategy``, ``group_col``,
            ``buffer_px``, ``row_col``, ``col_col``, ``input_csv_path``,
            ``save_fold_csvs_dir``.

    Example YAML:

    .. code-block:: yaml

        kfold:
          n_splits: 5
          seed: 42
          split_strategy: by_image
          group_col: image
          buffer_px: 0
          input_csv_path: /data/all_patches.csv
          save_fold_csvs_dir: fold_splits
    """

    VALID_STRATEGIES = {"by_image", "by_spatial_region"}

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def split_by_image(
        self, df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Group K-Fold by source image — no patch-overlap leakage across folds.

        Args:
            df: DataFrame containing at least the column referenced by
                ``cfg.group_col`` (default ``"image"``).

        Returns:
            List of ``(df_train, df_val)`` tuples, one per fold.

        Raises:
            ValueError: If ``df`` is empty or contains fewer unique groups
                than ``cfg.n_splits``.
        """
        if len(df) == 0:
            raise ValueError("DataFrame is empty.")

        group_col = self.cfg.group_col
        n_splits = self.cfg.n_splits
        groups = df[group_col].values
        n_unique = len(np.unique(groups))

        if n_unique < n_splits:
            raise ValueError(
                f"Number of unique groups ({n_unique}) is less than "
                f"n_splits ({n_splits}). Reduce n_splits or add more images."
            )

        gkf = GroupKFold(n_splits=n_splits)
        folds = []
        for train_idx, val_idx in gkf.split(df, groups=groups):
            folds.append((df.iloc[train_idx], df.iloc[val_idx]))

        return folds

    def split_by_spatial_region(
        self,
        df: pd.DataFrame,
        img_height: int,
        img_width: int,
        patch_size: int,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Divide patches into horizontal bands with optional boundary buffer.

        The image height is split into ``cfg.n_splits`` equal bands.  For each
        fold, one band is used as the validation set; the remainder form the
        training set.  Patches whose row footprint intersects any
        ``[boundary - buffer_px, boundary + buffer_px)`` zone are excluded
        from both sides of every fold.

        Args:
            df: DataFrame with at least a row-offset column (``cfg.row_col``).
            img_height: Total image height in pixels.
            img_width: Total image width in pixels (currently unused but kept
                for API completeness).
            patch_size: Patch height/width in pixels (square patches assumed).

        Returns:
            List of ``(df_train, df_val)`` tuples, one per fold.

        Raises:
            ValueError: If ``df`` is empty.
        """
        if len(df) == 0:
            raise ValueError("DataFrame is empty.")

        n_splits = self.cfg.n_splits
        buffer_px = self.cfg.buffer_px
        row_col = self.cfg.row_col

        # Compute band boundaries (n_splits - 1 internal boundaries)
        boundaries = [int(round(k * img_height / n_splits)) for k in range(1, n_splits)]

        # Pre-compute global buffer exclusion mask across all boundaries
        exclude_mask = pd.Series(False, index=df.index)
        if buffer_px > 0:
            for boundary in boundaries:
                lo = boundary - buffer_px
                hi = boundary + buffer_px
                in_buf = (df[row_col] + patch_size > lo) & (df[row_col] < hi)
                exclude_mask |= in_buf

        folds = []
        for k in range(n_splits):
            band_start = int(round(k * img_height / n_splits))
            band_end = int(round((k + 1) * img_height / n_splits))

            in_band = (df[row_col] >= band_start) & (df[row_col] < band_end)

            df_val = df[in_band & ~exclude_mask]
            df_train = df[~in_band & ~exclude_mask]
            folds.append((df_train, df_val))

        return folds

    def generate_and_save_folds(
        self,
        input_csv_path: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> List[Tuple[Path, Path]]:
        """Read a CSV, generate k-fold splits, and write per-fold CSVs to disk.

        Produces ``fold_0_train.csv``, ``fold_0_val.csv``, …,
        ``fold_{K-1}_train.csv``, ``fold_{K-1}_val.csv`` in ``output_dir``.

        For ``"by_spatial_region"``, ``patch_size`` is inferred from the
        ``patch_size`` column when present (default 256), and ``img_height`` /
        ``img_width`` are inferred from the maximum offsets in the DataFrame.

        Args:
            input_csv_path: Path to the master CSV with all patches.
            output_dir: Directory where fold CSVs are written (created if absent).

        Returns:
            List of ``(train_path, val_path)`` :class:`pathlib.Path` tuples,
            one per fold.

        Raises:
            ValueError: If ``cfg.split_strategy`` is not a recognised strategy.
        """
        strategy = self.cfg.split_strategy
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown split_strategy: {strategy!r}. "
                f"Must be one of {sorted(self.VALID_STRATEGIES)}."
            )

        input_csv_path = Path(input_csv_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(input_csv_path)

        if strategy == "by_image":
            folds = self.split_by_image(df)
        else:  # by_spatial_region
            patch_size = (
                int(df["patch_size"].iloc[0]) if "patch_size" in df.columns else 256
            )
            img_height = int(df[self.cfg.row_col].max()) + patch_size
            img_width = int(df[self.cfg.col_col].max()) + patch_size
            folds = self.split_by_spatial_region(
                df, img_height=img_height, img_width=img_width, patch_size=patch_size
            )

        paths: List[Tuple[Path, Path]] = []
        for k, (df_train, df_val) in enumerate(folds):
            train_path = output_dir / f"fold_{k}_train.csv"
            val_path = output_dir / f"fold_{k}_val.csv"
            df_train.to_csv(train_path, index=False)
            df_val.to_csv(val_path, index=False)
            paths.append((train_path, val_path))

        return paths

    def _apply_buffer(
        self,
        df: pd.DataFrame,
        boundary_row: int,
        buffer_px: int,
        patch_size: int,
    ) -> pd.DataFrame:
        """Return ``df`` without patches whose footprint intersects the buffer zone.

        The buffer zone around ``boundary_row`` spans
        ``[boundary_row - buffer_px, boundary_row + buffer_px)``.  A patch at
        ``row_off`` intersects this zone when
        ``row_off + patch_size > boundary_row - buffer_px`` and
        ``row_off < boundary_row + buffer_px``.

        Args:
            df: Input DataFrame with a row-offset column (``cfg.row_col``).
            boundary_row: Row coordinate of the fold boundary.
            buffer_px: Half-width of the exclusion zone in pixels.
            patch_size: Patch height in pixels.

        Returns:
            Filtered DataFrame with buffer-zone patches removed.
        """
        row_col = self.cfg.row_col
        lo = boundary_row - buffer_px
        hi = boundary_row + buffer_px
        in_buf = (df[row_col] + patch_size > lo) & (df[row_col] < hi)
        return df[~in_buf]
