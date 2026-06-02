# -*- coding: utf-8 -*-
"""Tests for tools/soft_labels/generate_training_csv.py."""

import pytest
import pandas as pd
from pathlib import Path

from pytorch_segmentation_models_trainer.tools.soft_labels.generate_training_csv import (
    split_dataframe,
    run,
)

N = 20  # enough rows for meaningful splits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def manifest_csv(tmp_path):
    """Write a minimal manifest CSV with tile_id, image_path, p_soft_path."""
    rows = [
        {
            "tile_id": f"tile_{i:03d}",
            "image_path": str(tmp_path / f"img_{i}.tif"),
            "p_soft_path": str(tmp_path / f"psoft_{i}.tif"),
            "w_conf_path": str(tmp_path / f"wconf_{i}.tif"),
        }
        for i in range(N)
    ]
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "manifest.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def manifest_csv_no_image(tmp_path):
    """Manifest CSV without image_path column."""
    rows = [
        {
            "tile_id": f"tile_{i:03d}",
            "p_soft_path": str(tmp_path / f"psoft_{i}.tif"),
        }
        for i in range(N)
    ]
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "manifest_no_img.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# split_dataframe unit tests
# ---------------------------------------------------------------------------


class TestSplitDataframe:
    def _df(self):
        return pd.DataFrame({"x": range(N)})

    def test_keys(self):
        splits = split_dataframe(self._df())
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_total_rows_preserved(self):
        splits = split_dataframe(self._df())
        total = sum(len(v) for v in splits.values())
        assert total == N

    def test_no_overlap_between_splits(self):
        splits = split_dataframe(self._df())
        train_idx = set(splits["train"].index)
        val_idx = set(splits["val"].index)
        test_idx = set(splits["test"].index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_train_size_approx(self):
        splits = split_dataframe(self._df(), train_ratio=0.70)
        assert abs(len(splits["train"]) - int(N * 0.70)) <= 1

    def test_val_size_approx(self):
        splits = split_dataframe(self._df(), val_ratio=0.15)
        assert abs(len(splits["val"]) - int(N * 0.15)) <= 1

    def test_reproducible_with_same_seed(self):
        s1 = split_dataframe(self._df(), seed=0)
        s2 = split_dataframe(self._df(), seed=0)
        assert list(s1["train"].index) == list(s2["train"].index)

    def test_different_seeds_give_different_splits(self):
        s1 = split_dataframe(self._df(), seed=0)
        s2 = split_dataframe(self._df(), seed=999)
        assert list(s1["train"].index) != list(s2["train"].index)

    def test_invalid_ratios_raise(self):
        with pytest.raises(ValueError):
            split_dataframe(self._df(), train_ratio=0.8, val_ratio=0.3)


# ---------------------------------------------------------------------------
# run() integration tests
# ---------------------------------------------------------------------------


class TestRunGenerateTrainingCSV:
    def test_creates_three_csv_files(self, manifest_csv, tmp_path):
        out = tmp_path / "splits"
        run(manifest_path=str(manifest_csv), output_dir=out)
        assert (out / "train.csv").exists()
        assert (out / "val.csv").exists()
        assert (out / "test.csv").exists()

    def test_returns_paths_dict(self, manifest_csv, tmp_path):
        out = tmp_path / "splits"
        paths = run(manifest_path=str(manifest_csv), output_dir=out)
        assert set(paths.keys()) == {"train", "val", "test"}
        for p in paths.values():
            assert isinstance(p, Path)

    def test_total_rows_preserved(self, manifest_csv, tmp_path):
        out = tmp_path / "splits"
        paths = run(manifest_path=str(manifest_csv), output_dir=out)
        total = sum(len(pd.read_csv(p)) for p in paths.values())
        assert total == N

    def test_output_dir_created(self, manifest_csv, tmp_path):
        out = tmp_path / "deep" / "nested" / "splits"
        run(manifest_path=str(manifest_csv), output_dir=out)
        assert out.is_dir()

    def test_columns_preserved(self, manifest_csv, tmp_path):
        out = tmp_path / "splits"
        paths = run(manifest_path=str(manifest_csv), output_dir=out)
        df = pd.read_csv(paths["train"])
        assert "tile_id" in df.columns
        assert "image_path" in df.columns
        assert "p_soft_path" in df.columns

    def test_image_dir_injected_when_missing(self, manifest_csv_no_image, tmp_path):
        out = tmp_path / "splits"
        paths = run(
            manifest_path=str(manifest_csv_no_image),
            output_dir=out,
            image_dir=str(tmp_path / "images"),
            image_extension=".tif",
        )
        df = pd.read_csv(paths["train"])
        assert "image_path" in df.columns
        assert df["image_path"].str.endswith(".tif").all()

    def test_no_image_dir_raises_when_column_missing(
        self, manifest_csv_no_image, tmp_path
    ):
        out = tmp_path / "splits"
        with pytest.raises(ValueError, match="image_path"):
            run(manifest_path=str(manifest_csv_no_image), output_dir=out)

    def test_custom_ratios(self, manifest_csv, tmp_path):
        out = tmp_path / "splits"
        paths = run(
            manifest_path=str(manifest_csv),
            output_dir=out,
            train_ratio=0.6,
            val_ratio=0.2,
        )
        assert abs(len(pd.read_csv(paths["train"])) - int(N * 0.6)) <= 1
        assert abs(len(pd.read_csv(paths["val"])) - int(N * 0.2)) <= 1

    def test_invalid_ratios_raise(self, manifest_csv, tmp_path):
        out = tmp_path / "splits"
        with pytest.raises(ValueError):
            run(
                manifest_path=str(manifest_csv),
                output_dir=out,
                train_ratio=0.8,
                val_ratio=0.3,
            )
