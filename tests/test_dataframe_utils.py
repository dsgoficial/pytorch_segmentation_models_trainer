# -*- coding: utf-8 -*-
import os
import pandas as pd
from pytorch_segmentation_models_trainer.utils.dataframe_utils import read_dataframe


def test_read_dataframe_csv_cache(tmp_path):
    # 1. Create a dummy CSV
    csv_path = tmp_path / "test.csv"
    df_orig = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df_orig.to_csv(csv_path, index=False)

    # 2. Read it for the first time (should create cache)
    df_read = read_dataframe(csv_path, use_cache=True)
    pd.testing.assert_frame_equal(df_orig, df_read)

    cache_path = csv_path.with_suffix(".cache.parquet")
    assert cache_path.exists()

    # 3. Read it again (should use cache)
    # We can verify it by checking if it's actually a parquet file being read
    # or by modifying the cache and seeing if it reflects the change
    df_cached = pd.read_parquet(cache_path)
    df_cached.iloc[0, 0] = 99
    df_cached.to_parquet(cache_path, index=False)

    df_read_back = read_dataframe(csv_path, use_cache=True)
    assert df_read_back.iloc[0, 0] == 99

    # 4. Modify CSV (cache should be invalidated/updated)
    df_orig.iloc[0, 0] = 100
    df_orig.to_csv(csv_path, index=False)
    # Ensure mtime of CSV is greater than cache
    os.utime(csv_path, (os.path.getatime(csv_path), os.path.getmtime(cache_path) + 10))

    df_updated = read_dataframe(csv_path, use_cache=True)
    assert df_updated.iloc[0, 0] == 100


def test_read_dataframe_parquet(tmp_path):
    # 1. Create a dummy Parquet
    parquet_path = tmp_path / "test.parquet"
    df_orig = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df_orig.to_parquet(parquet_path, index=False)

    # 2. Read it
    df_read = read_dataframe(parquet_path)
    pd.testing.assert_frame_equal(df_orig, df_read)


def test_read_dataframe_parquet_nrows(tmp_path):
    parquet_path = tmp_path / "test_rows.parquet"
    df_orig = pd.DataFrame({"a": [1, 2, 3]})
    df_orig.to_parquet(parquet_path, index=False)

    df_read = read_dataframe(parquet_path, nrows=2)

    assert list(df_read["a"]) == [1, 2]


def test_read_dataframe_nrows(tmp_path):
    csv_path = tmp_path / "test_rows.csv"
    df_orig = pd.DataFrame({"a": range(10)})
    df_orig.to_csv(csv_path, index=False)

    df_read = read_dataframe(csv_path, nrows=5, use_cache=False)
    assert len(df_read) == 5
    assert list(df_read["a"]) == [0, 1, 2, 3, 4]


def test_read_dataframe_cache_read_failure_falls_back_to_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "test.csv"
    df_orig = pd.DataFrame({"a": [1, 2, 3]})
    df_orig.to_csv(csv_path, index=False)
    cache_path = csv_path.with_suffix(".cache.parquet")
    pd.DataFrame({"a": [99]}).to_parquet(cache_path, index=False)
    os.utime(
        cache_path, (os.path.getatime(cache_path), os.path.getmtime(csv_path) + 10)
    )

    def raise_cache_error(*args, **kwargs):
        raise RuntimeError("bad cache")

    monkeypatch.setattr(pd, "read_parquet", raise_cache_error)

    df_read = read_dataframe(csv_path, use_cache=True)

    pd.testing.assert_frame_equal(df_orig, df_read)


def test_read_dataframe_fresh_cache_respects_nrows(tmp_path):
    csv_path = tmp_path / "test.csv"
    df_orig = pd.DataFrame({"a": [1, 2, 3]})
    df_orig.to_csv(csv_path, index=False)
    cache_path = csv_path.with_suffix(".cache.parquet")
    df_orig.to_parquet(cache_path, index=False)
    os.utime(
        cache_path, (os.path.getatime(cache_path), os.path.getmtime(csv_path) + 10)
    )

    df_read = read_dataframe(csv_path, use_cache=True, nrows=2)

    assert list(df_read["a"]) == [1, 2]


def test_read_dataframe_cache_save_failure_still_returns_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "test.csv"
    df_orig = pd.DataFrame({"a": [1, 2, 3]})
    df_orig.to_csv(csv_path, index=False)

    def raise_save_error(self, *args, **kwargs):
        raise RuntimeError("cannot save")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", raise_save_error)

    df_read = read_dataframe(csv_path, use_cache=True)

    pd.testing.assert_frame_equal(df_orig, df_read)


def test_read_dataframe_falls_back_to_csv_for_unknown_extension(tmp_path):
    data_path = tmp_path / "test.data"
    df_orig = pd.DataFrame({"a": [1, 2, 3]})
    df_orig.to_csv(data_path, index=False)

    df_read = read_dataframe(data_path)

    pd.testing.assert_frame_equal(df_orig, df_read)
