# -*- coding: utf-8 -*-
"""
Tests for SpatialKFoldSplitter (TDD — written before implementation).

Run with:
    python -m pytest tests/test_spatial_kfold.py -v --tb=short
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds

from tests.utils import BasicTestCase


def _write_tif(path: Path, width: int = 256, height: int = 256, bands: int = 3):
    """Create a minimal valid GeoTIFF for tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=0)
    data = rng.integers(0, 255, (bands, height, width), dtype=np.uint8)
    transform = from_bounds(0, 0, 1, 1, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


def _make_image_df(n_images: int, patches_per_image: int = 4, tmp_dir: Path = None):
    """Build a DataFrame with n_images × patches_per_image rows, simulating
    a CSVWindowedSegmentationDataset CSV (image, mask, row_off, col_off, patch_size)."""
    records = []
    for i in range(n_images):
        img_path = str(tmp_dir / f"img_{i}.tif") if tmp_dir else f"/data/img_{i}.tif"
        mask_path = str(tmp_dir / f"mask_{i}.tif") if tmp_dir else f"/data/mask_{i}.tif"
        for p in range(patches_per_image):
            records.append(
                {
                    "image": img_path,
                    "mask": mask_path,
                    "row_off": (p % 2) * 256,
                    "col_off": (p // 2) * 256,
                    "patch_size": 256,
                }
            )
    return pd.DataFrame(records)


def _make_spatial_df(
    n_images: int = 1,
    img_height: int = 1000,
    img_width: int = 1000,
    patch_size: int = 256,
    stride: int = 256,
):
    """Build a DataFrame with patch positions covering an image grid."""
    records = []
    img_path = "/data/img_0.tif"
    mask_path = "/data/mask_0.tif"
    row = 0
    while row + patch_size <= img_height:
        col = 0
        while col + patch_size <= img_width:
            records.append(
                {
                    "image": img_path,
                    "mask": mask_path,
                    "row_off": row,
                    "col_off": col,
                    "patch_size": patch_size,
                }
            )
            col += stride
        row += stride
    return pd.DataFrame(records)


def _make_kfold_cfg(
    n_splits=5,
    seed=42,
    split_strategy="by_image",
    group_col="image",
    buffer_px=0,
    row_col="row_off",
    col_col="col_off",
    input_csv_path="",
    save_fold_csvs_dir="fold_splits",
):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "n_splits": n_splits,
            "seed": seed,
            "split_strategy": split_strategy,
            "group_col": group_col,
            "buffer_px": buffer_px,
            "row_col": row_col,
            "col_col": col_col,
            "input_csv_path": input_csv_path,
            "save_fold_csvs_dir": save_fold_csvs_dir,
        }
    )


class TestSpatialKFoldSplitterByImage(BasicTestCase):
    def _import(self):
        from pytorch_segmentation_models_trainer.utils.spatial_kfold import (
            SpatialKFoldSplitter,
        )

        return SpatialKFoldSplitter

    def _make_splitter(self, **kwargs):
        SpatialKFoldSplitter = self._import()
        return SpatialKFoldSplitter(_make_kfold_cfg(**kwargs))

    def test_split_by_image_no_leak(self):
        """No image should appear in both train and val of the same fold."""
        splitter = self._make_splitter(n_splits=5)
        df = _make_image_df(n_images=10, patches_per_image=4)
        folds = splitter.split_by_image(df)
        for df_train, df_val in folds:
            train_images = set(df_train["image"].unique())
            val_images = set(df_val["image"].unique())
            self.assertEqual(
                len(train_images & val_images),
                0,
                msg="Data leak: image(s) appear in both train and val of the same fold.",
            )

    def test_split_by_image_all_rows_covered(self):
        """The union of all val sets must cover every row index in the DataFrame."""
        splitter = self._make_splitter(n_splits=5)
        df = _make_image_df(n_images=10, patches_per_image=4)
        folds = splitter.split_by_image(df)
        covered = set()
        for _, df_val in folds:
            covered.update(df_val.index.tolist())
        self.assertEqual(covered, set(df.index.tolist()))

    def test_split_by_image_n_folds_correct(self):
        """Returns exactly n_splits folds."""
        splitter = self._make_splitter(n_splits=5)
        df = _make_image_df(n_images=10, patches_per_image=4)
        folds = splitter.split_by_image(df)
        self.assertEqual(len(folds), 5)

    def test_split_by_image_patches_same_image_same_fold(self):
        """All patches from a single image must land in the same fold."""
        splitter = self._make_splitter(n_splits=5)
        df = _make_image_df(n_images=10, patches_per_image=4)
        folds = splitter.split_by_image(df)
        for fold_idx, (df_train, df_val) in enumerate(folds):
            for image_path in df["image"].unique():
                train_count = (df_train["image"] == image_path).sum()
                val_count = (df_val["image"] == image_path).sum()
                # Patches from an image are either all in train or all in val, never both
                self.assertTrue(
                    train_count == 0 or val_count == 0,
                    msg=f"Fold {fold_idx}: patches from {image_path} split between train and val.",
                )

    def test_split_by_image_custom_group_col(self):
        """A group_col other than 'image' is respected."""
        splitter = self._make_splitter(n_splits=3, group_col="mask")
        df = _make_image_df(n_images=6, patches_per_image=2)
        folds = splitter.split_by_image(df)
        self.assertEqual(len(folds), 3)
        for df_train, df_val in folds:
            train_masks = set(df_train["mask"].unique())
            val_masks = set(df_val["mask"].unique())
            self.assertEqual(len(train_masks & val_masks), 0)

    def test_empty_dataframe_raises(self):
        """An empty DataFrame raises ValueError."""
        splitter = self._make_splitter()
        df = pd.DataFrame(columns=["image", "mask", "row_off", "col_off", "patch_size"])
        with self.assertRaises(ValueError):
            splitter.split_by_image(df)

    def test_fewer_images_than_splits_raises(self):
        """Fewer unique images than n_splits raises ValueError."""
        splitter = self._make_splitter(n_splits=5)
        df = _make_image_df(n_images=3, patches_per_image=4)
        with self.assertRaises(ValueError):
            splitter.split_by_image(df)

    def test_invalid_strategy_raises(self):
        """An invalid split_strategy raises ValueError."""
        splitter = self._make_splitter(split_strategy="invalid_strategy")
        df = _make_image_df(n_images=10, patches_per_image=4)
        with self.assertRaises(ValueError, msg="should raise for invalid strategy"):
            splitter.generate_and_save_folds(
                "/does/not/matter.csv", Path(self.make_temp_dir())
            )

    def test_buffer_zero_no_exclusion_by_image(self):
        """With strategy='by_image' and buffer_px=0, every patch appears in some val fold."""
        splitter = self._make_splitter(n_splits=5, buffer_px=0)
        df = _make_image_df(n_images=10, patches_per_image=4)
        folds = splitter.split_by_image(df)
        total_val = sum(len(df_val) for _, df_val in folds)
        self.assertEqual(total_val, len(df))


class TestSpatialKFoldSplitterByRegion(BasicTestCase):
    def _import(self):
        from pytorch_segmentation_models_trainer.utils.spatial_kfold import (
            SpatialKFoldSplitter,
        )

        return SpatialKFoldSplitter

    def _make_splitter(self, **kwargs):
        SpatialKFoldSplitter = self._import()
        return SpatialKFoldSplitter(
            _make_kfold_cfg(split_strategy="by_spatial_region", **kwargs)
        )

    def test_buffer_excludes_boundary_patches(self):
        """Patches in the buffer zone are excluded from both train and val."""
        patch_size = 256
        img_height = 1000
        buffer_px = 256
        boundary = img_height // 2  # 500

        splitter = self._make_splitter(n_splits=2, buffer_px=buffer_px)
        df = _make_spatial_df(
            img_height=img_height,
            img_width=1000,
            patch_size=patch_size,
            stride=patch_size,
        )

        folds = splitter.split_by_spatial_region(
            df, img_height=img_height, img_width=1000, patch_size=patch_size
        )
        self.assertEqual(len(folds), 2)

        for df_train, df_val in folds:
            all_rows = pd.concat([df_train, df_val])
            # No patch should start inside the buffer: [boundary-buffer_px, boundary+buffer_px)
            lo = boundary - buffer_px
            hi = boundary + buffer_px
            in_buffer = all_rows[
                (all_rows["row_off"] + patch_size > lo) & (all_rows["row_off"] < hi)
            ]
            self.assertEqual(
                len(in_buffer),
                0,
                msg=f"Patches found in buffer zone [{lo},{hi}): {in_buffer['row_off'].tolist()}",
            )

    def test_no_geometric_overlap_with_buffer(self):
        """With buffer >= patch_size, no train patch overlaps with any val patch."""
        patch_size = 256
        img_height = 1024
        buffer_px = patch_size  # buffer equal to patch_size eliminates overlap

        splitter = self._make_splitter(n_splits=2, buffer_px=buffer_px)
        df = _make_spatial_df(
            img_height=img_height,
            img_width=512,
            patch_size=patch_size,
            stride=patch_size,
        )

        folds = splitter.split_by_spatial_region(
            df, img_height=img_height, img_width=512, patch_size=patch_size
        )
        df_train, df_val = folds[0]

        for _, train_row in df_train.iterrows():
            t_top = train_row["row_off"]
            t_bot = t_top + patch_size
            for _, val_row in df_val.iterrows():
                v_top = val_row["row_off"]
                v_bot = v_top + patch_size
                overlap = min(t_bot, v_bot) - max(t_top, v_top)
                self.assertLessEqual(
                    overlap,
                    0,
                    msg=f"Overlap between train patch row={t_top} and val patch row={v_top}",
                )

    def test_n_folds_correct_by_region(self):
        """split_by_spatial_region returns exactly n_splits folds."""
        splitter = self._make_splitter(n_splits=3, buffer_px=0)
        df = _make_spatial_df(img_height=768, img_width=256, patch_size=256, stride=256)
        folds = splitter.split_by_spatial_region(
            df, img_height=768, img_width=256, patch_size=256
        )
        self.assertEqual(len(folds), 3)

    def test_empty_dataframe_raises_by_region(self):
        """An empty DataFrame raises ValueError."""
        splitter = self._make_splitter(n_splits=2, buffer_px=0)
        df = pd.DataFrame(columns=["image", "mask", "row_off", "col_off", "patch_size"])
        with self.assertRaises(ValueError):
            splitter.split_by_spatial_region(
                df, img_height=1000, img_width=1000, patch_size=256
            )


class TestGenerateAndSaveFolds(BasicTestCase):
    def _import(self):
        from pytorch_segmentation_models_trainer.utils.spatial_kfold import (
            SpatialKFoldSplitter,
        )

        return SpatialKFoldSplitter

    def _make_splitter(self, **kwargs):
        SpatialKFoldSplitter = self._import()
        return SpatialKFoldSplitter(_make_kfold_cfg(**kwargs))

    def test_creates_csv_files_on_disk(self):
        """generate_and_save_folds creates fold_K_train.csv and fold_K_val.csv."""
        tmp = Path(self.make_temp_dir())
        csv_path = tmp / "all_patches.csv"
        df = _make_image_df(n_images=10, patches_per_image=4)
        df.to_csv(csv_path, index=False)

        splitter = self._make_splitter(n_splits=3)
        fold_paths = splitter.generate_and_save_folds(csv_path, tmp / "splits")

        self.assertEqual(len(fold_paths), 3)
        for k, (train_path, val_path) in enumerate(fold_paths):
            self.assertTrue(Path(train_path).exists(), f"fold_{k}_train.csv not found")
            self.assertTrue(Path(val_path).exists(), f"fold_{k}_val.csv not found")
            self.assertIn(f"fold_{k}_train", str(train_path))
            self.assertIn(f"fold_{k}_val", str(val_path))

    def test_csv_columns_preserved(self):
        """Generated CSVs have the same columns as the input CSV."""
        tmp = Path(self.make_temp_dir())
        csv_path = tmp / "all_patches.csv"
        df = _make_image_df(n_images=10, patches_per_image=4)
        df.to_csv(csv_path, index=False)

        splitter = self._make_splitter(n_splits=3)
        fold_paths = splitter.generate_and_save_folds(csv_path, tmp / "splits")

        for train_path, val_path in fold_paths:
            df_train = pd.read_csv(train_path)
            df_val = pd.read_csv(val_path)
            self.assertEqual(set(df_train.columns), set(df.columns))
            self.assertEqual(set(df_val.columns), set(df.columns))

    def test_train_plus_val_equals_all_data(self):
        """For each fold, len(train) + len(val) equals the total number of rows."""
        tmp = Path(self.make_temp_dir())
        csv_path = tmp / "all_patches.csv"
        df = _make_image_df(n_images=10, patches_per_image=4)
        df.to_csv(csv_path, index=False)

        splitter = self._make_splitter(n_splits=5)
        fold_paths = splitter.generate_and_save_folds(csv_path, tmp / "splits")

        for train_path, val_path in fold_paths:
            df_train = pd.read_csv(train_path)
            df_val = pd.read_csv(val_path)
            self.assertEqual(len(df_train) + len(df_val), len(df))

    def test_returns_path_tuples(self):
        """generate_and_save_folds returns a list of (Path, Path) tuples."""
        tmp = Path(self.make_temp_dir())
        csv_path = tmp / "all_patches.csv"
        df = _make_image_df(n_images=6, patches_per_image=2)
        df.to_csv(csv_path, index=False)

        splitter = self._make_splitter(n_splits=3)
        fold_paths = splitter.generate_and_save_folds(csv_path, tmp / "splits")

        self.assertIsInstance(fold_paths, list)
        for item in fold_paths:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)

    def test_invalid_strategy_in_generate_raises(self):
        """An invalid strategy in generate_and_save_folds raises ValueError."""
        tmp = Path(self.make_temp_dir())
        csv_path = tmp / "all_patches.csv"
        df = _make_image_df(n_images=6, patches_per_image=2)
        df.to_csv(csv_path, index=False)

        SpatialKFoldSplitter = self._import()
        splitter = SpatialKFoldSplitter(
            _make_kfold_cfg(n_splits=3, split_strategy="invalid")
        )
        with self.assertRaises(ValueError):
            splitter.generate_and_save_folds(csv_path, tmp / "splits")

    def test_idempotent_second_call(self):
        """Calling generate_and_save_folds twice with the same output dir overwrites CSVs."""
        tmp = Path(self.make_temp_dir())
        csv_path = tmp / "all_patches.csv"
        df = _make_image_df(n_images=6, patches_per_image=2)
        df.to_csv(csv_path, index=False)

        splitter = self._make_splitter(n_splits=3, seed=42)
        folds1 = splitter.generate_and_save_folds(csv_path, tmp / "splits")
        folds2 = splitter.generate_and_save_folds(csv_path, tmp / "splits")

        # Same seed → same split
        for (t1, v1), (t2, v2) in zip(folds1, folds2):
            df_t1 = pd.read_csv(t1)
            df_t2 = pd.read_csv(t2)
            pd.testing.assert_frame_equal(
                df_t1.reset_index(drop=True), df_t2.reset_index(drop=True)
            )

    def test_generate_by_spatial_region(self):
        """generate_and_save_folds with split_strategy='by_spatial_region' creates correct CSVs."""
        tmp = Path(self.make_temp_dir())
        csv_path = tmp / "spatial_patches.csv"
        # 3 patches at distinct vertical positions → img_height inferred = 512 + 256 = 768
        df = _make_spatial_df(img_height=768, img_width=256, patch_size=256, stride=256)
        df.to_csv(csv_path, index=False)

        SpatialKFoldSplitter = self._import()
        splitter = SpatialKFoldSplitter(
            _make_kfold_cfg(n_splits=3, split_strategy="by_spatial_region", buffer_px=0)
        )
        fold_paths = splitter.generate_and_save_folds(csv_path, tmp / "splits")

        self.assertEqual(len(fold_paths), 3)
        for train_path, val_path in fold_paths:
            self.assertTrue(Path(train_path).exists())
            self.assertTrue(Path(val_path).exists())
            df_t = pd.read_csv(train_path)
            df_v = pd.read_csv(val_path)
            self.assertEqual(set(df_t.columns), set(df.columns))
            self.assertEqual(len(df_t) + len(df_v), len(df))


class TestApplyBuffer(BasicTestCase):
    def _make_splitter(self, **kwargs):
        from pytorch_segmentation_models_trainer.utils.spatial_kfold import (
            SpatialKFoldSplitter,
        )

        return SpatialKFoldSplitter(_make_kfold_cfg(**kwargs))

    def test_apply_buffer_removes_boundary_patches(self):
        """_apply_buffer removes patches whose footprint intersects the buffer zone."""
        splitter = self._make_splitter(row_col="row_off")
        df = _make_spatial_df(img_height=768, img_width=256, patch_size=256, stride=256)
        # boundary=256, buffer_px=128 → removes patches with row_off+256>128 and row_off<384
        # row_off=0: 0+256=256>128 and 0<384 → removed
        # row_off=256: 256+256=512>128 and 256<384 → removed
        # row_off=512: 512+256=768>128 and 512<384? → False → not removed
        filtered = splitter._apply_buffer(
            df, boundary_row=256, buffer_px=128, patch_size=256
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["row_off"], 512)

    def test_apply_buffer_zero_no_exclusion(self):
        """_apply_buffer with buffer_px=0 removes nothing."""
        splitter = self._make_splitter(row_col="row_off")
        df = _make_spatial_df(img_height=768, img_width=256, patch_size=256, stride=256)
        filtered = splitter._apply_buffer(
            df, boundary_row=256, buffer_px=0, patch_size=256
        )
        # boundary_row=256, buffer_px=0 → lo=256, hi=256
        # condition: row_off + patch_size > 256 AND row_off < 256
        # row_off=0: 256>256? False → not excluded
        # Nothing is excluded when lo == hi
        self.assertEqual(len(filtered), len(df))
