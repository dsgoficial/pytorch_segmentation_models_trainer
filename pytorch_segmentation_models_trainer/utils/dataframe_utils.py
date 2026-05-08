# -*- coding: utf-8 -*-
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Union


def read_dataframe(
    path: Union[str, Path],
    nrows: Optional[int] = None,
    use_cache: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Reads a dataframe from a CSV or Parquet file.

    If the file is a CSV and use_cache is True, it tries to read from a
    corresponding .cache.parquet file. If the cache doesn't exist or is
    older than the CSV, it reads the CSV and updates the cache.

    Args:
        path (Union[str, Path]): Path to the file.
        nrows (Optional[int], optional): Number of rows to read. Defaults to None.
        use_cache (bool, optional): Whether to use parquet cache for CSV files.
            Defaults to True.
        **kwargs: Additional arguments passed to pd.read_csv or pd.read_parquet.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """
    path = Path(path)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, **kwargs)
        if nrows is not None:
            df = df.head(nrows)
        return df

    if path.suffix.lower() == ".csv":
        cache_path = path.with_suffix(".cache.parquet")

        if use_cache:
            if (
                cache_path.exists()
                and cache_path.stat().st_mtime > path.stat().st_mtime
            ):
                try:
                    df = pd.read_parquet(cache_path, **kwargs)
                    if nrows is not None:
                        df = df.head(nrows)
                    return df
                except Exception as e:
                    print(
                        f"Failed to read cache {cache_path}: {e}. Falling back to CSV."
                    )

        # Read CSV
        df = pd.read_csv(path, nrows=nrows, **kwargs)

        # Update cache if requested and we read the whole file
        if use_cache and nrows is None:
            try:
                df.to_parquet(cache_path, index=False)
            except Exception as e:
                print(f"Failed to save cache {cache_path}: {e}")

        return df

    # Fallback for other extensions or if suffix is missing
    return pd.read_csv(path, nrows=nrows, **kwargs)
