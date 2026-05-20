# -*- coding: utf-8 -*-
"""
Hydra ConfigStore dataclass for spatial k-fold cross-validation.
"""

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class KFoldConfig:
    """Configuration for spatially-correct k-fold cross-validation.

    Defines how to split patch data into K folds without data leakage from
    overlapping patches (via Group K-Fold by source image) and with an optional
    exclusion buffer to reduce spatial autocorrelation at fold boundaries.

    Args:
        n_splits: Number of folds. Default: 5.
        seed: Reproducibility seed for the split. Default: 42.
        split_strategy: Split strategy. ``"by_image"`` uses Group K-Fold
            grouping all patches by their source image (``group_col``),
            eliminating data leakage from patch overlap. ``"by_spatial_region"``
            divides by geographic region within images, applying an exclusion
            buffer at boundaries. Default: ``"by_image"``.
        group_col: Column used to group patches by source image. Used only
            when ``split_strategy="by_image"``. Default: ``"image"``.
        buffer_px: Width in pixels of the exclusion zone around fold boundaries.
            Patches whose footprint intersects this zone are removed from both
            train and val. Recommended ``>= patch_size`` for zero pixel overlap,
            ``>= 2 * patch_size`` to further reduce residual autocorrelation.
            ``0`` disables. Used only with ``split_strategy="by_spatial_region"``.
            Default: 0.
        row_col: Column containing the patch row offset. Used by
            ``split_strategy="by_spatial_region"``. Default: ``"row_off"``.
        col_col: Column containing the patch column offset. Used by
            ``split_strategy="by_spatial_region"``. Default: ``"col_off"``.
        input_csv_path: Path to the master CSV with all patches to split.
            Read by ``ExperimentsRunner`` to generate splits before the training loop.
        save_fold_csvs_dir: Subdirectory (relative to ``output_base_dir``) where
            per-fold CSVs are saved (``fold_0_train.csv``, ``fold_0_val.csv``,
            …). Default: ``"fold_splits"``.

    Example YAML:

    .. code-block:: yaml

        experiments_runner:
          seeds: [42, 101]
          output_base_dir: outputs/kfold_study
          kfold:
            n_splits: 5
            seed: 42
            split_strategy: by_image
            group_col: image
            buffer_px: 0
            input_csv_path: /data/all_patches.csv
            save_fold_csvs_dir: fold_splits
    """

    n_splits: int = 5
    seed: int = 42
    split_strategy: str = "by_image"
    group_col: str = "image"
    buffer_px: int = 0
    row_col: str = "row_off"
    col_col: str = "col_off"
    input_csv_path: str = ""
    save_fold_csvs_dir: str = "fold_splits"


def _register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="kfold_config",
        node=KFoldConfig,
        group="kfold",
    )


_register_configs()
