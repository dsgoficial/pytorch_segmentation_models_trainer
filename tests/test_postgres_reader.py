# -*- coding: utf-8 -*-
"""Tests for tools/sampling/postgres_reader.py."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pytorch_segmentation_models_trainer.tools.sampling.postgres_reader import (
    PostgresReader,
)
from pytorch_segmentation_models_trainer.tools.sampling.sampling_config import (
    PostgresConfig,
)


def _default_config(**kwargs) -> PostgresConfig:
    return PostgresConfig(
        host="localhost",
        port=5432,
        database="test_db",
        user="user",
        password="pass",
        table="tile",
        **kwargs,
    )


def _raw_df() -> pd.DataFrame:
    """Return a DataFrame as psycopg2/pd.read_sql would return it (pre-rename)."""
    return pd.DataFrame(
        {
            "file_path": ["/img_0.tif", "/img_1.tif"],
            "mask_path": ["/msk_0.tif", "/msk_1.tif"],
            "row_off": [0, 128],
            "col_off": [0, 0],
            "patch_size": [256, 256],
            "class_dist": ['{"1": 0.5, "2": 0.5}', '{"3": 1.0}'],
            "uniqueness_score": [0.4, 0.7],
            "class_entropy": [0.693, 0.0],
            "nodata_ratio": [0.0, 0.1],
        }
    )


def _make_psycopg2_mock() -> MagicMock:
    mock = MagicMock()
    mock.connect.return_value = MagicMock()
    return mock


def test_read_tiles_returns_dataframe():
    cfg = _default_config()
    reader = PostgresReader(cfg)
    mock_psycopg2 = _make_psycopg2_mock()

    with (
        patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        patch("pandas.read_sql", return_value=_raw_df()),
    ):
        df = reader.read_tiles()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_read_tiles_renames_columns():
    cfg = _default_config()
    reader = PostgresReader(cfg)
    mock_psycopg2 = _make_psycopg2_mock()

    with (
        patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        patch("pandas.read_sql", return_value=_raw_df()),
    ):
        df = reader.read_tiles()

    assert "image_path" in df.columns
    assert "mask_path" in df.columns
    assert "uniqueness_score" in df.columns
    assert "file_path" not in df.columns


def test_read_tiles_parses_class_dist_json():
    cfg = _default_config()
    reader = PostgresReader(cfg)
    mock_psycopg2 = _make_psycopg2_mock()

    with (
        patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        patch("pandas.read_sql", return_value=_raw_df()),
    ):
        df = reader.read_tiles()

    for val in df["class_dist"]:
        assert isinstance(val, dict)


def test_read_tiles_with_where_clause():
    cfg = _default_config(where_clause="nodata_ratio < 0.1")
    reader = PostgresReader(cfg)
    mock_psycopg2 = _make_psycopg2_mock()

    with (
        patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        patch("pandas.read_sql", return_value=_raw_df()) as mock_sql,
    ):
        reader.read_tiles()

    call_args = mock_sql.call_args[0][0]
    assert "nodata_ratio < 0.1" in call_args


def test_read_tiles_no_where_clause():
    cfg = _default_config(where_clause=None)
    reader = PostgresReader(cfg)
    mock_psycopg2 = _make_psycopg2_mock()

    with (
        patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        patch("pandas.read_sql", return_value=_raw_df()) as mock_sql,
    ):
        reader.read_tiles()

    call_args = mock_sql.call_args[0][0]
    assert "WHERE" not in call_args


def test_read_tiles_closes_connection():
    cfg = _default_config()
    reader = PostgresReader(cfg)
    mock_psycopg2 = _make_psycopg2_mock()

    with (
        patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        patch("pandas.read_sql", return_value=_raw_df()),
    ):
        reader.read_tiles()

    mock_psycopg2.connect.return_value.close.assert_called_once()


def test_read_tiles_import_error_without_psycopg2():
    cfg = _default_config()
    reader = PostgresReader(cfg)

    with patch.dict("sys.modules", {"psycopg2": None}):
        with pytest.raises(ImportError, match="psycopg2"):
            reader.read_tiles()


def test_read_tiles_class_dist_already_dict():
    cfg = _default_config()
    reader = PostgresReader(cfg)
    mock_psycopg2 = _make_psycopg2_mock()

    df_with_dict = _raw_df()
    df_with_dict["class_dist"] = [{"1": 0.5, "2": 0.5}, {"3": 1.0}]

    with (
        patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        patch("pandas.read_sql", return_value=df_with_dict),
    ):
        df = reader.read_tiles()

    for val in df["class_dist"]:
        assert isinstance(val, dict)


def _raw_df_with_embedding() -> pd.DataFrame:
    """Raw DataFrame with embedding column as list (psycopg2 + pgvector adapter)."""
    import numpy as np

    df = _raw_df()
    df["embedding"] = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ]
    return df


def _raw_df_with_embedding_str() -> pd.DataFrame:
    """Raw DataFrame with embedding column as string (pgvector default repr)."""
    df = _raw_df()
    df["embedding"] = ["[0.1,0.2,0.3,0.4]", "[0.5,0.6,0.7,0.8]"]
    return df


def test_fetch_embeddings_includes_column_in_query():
    cfg = _default_config(fetch_embeddings=True)
    reader = PostgresReader(cfg)
    mock_psycopg2 = _make_psycopg2_mock()

    with (
        patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        patch("pandas.read_sql", return_value=_raw_df_with_embedding()) as mock_sql,
    ):
        reader.read_tiles()

    query = mock_sql.call_args[0][0]
    assert "embedding" in query


def test_fetch_embeddings_false_excludes_column_from_query():
    cfg = _default_config(fetch_embeddings=False)
    reader = PostgresReader(cfg)
    mock_psycopg2 = _make_psycopg2_mock()

    with (
        patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        patch("pandas.read_sql", return_value=_raw_df()) as mock_sql,
    ):
        reader.read_tiles()

    query = mock_sql.call_args[0][0]
    assert "embedding" not in query


def test_fetch_embeddings_parses_list_to_numpy():
    import numpy as np

    cfg = _default_config(fetch_embeddings=True)
    reader = PostgresReader(cfg)
    mock_psycopg2 = _make_psycopg2_mock()

    with (
        patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        patch("pandas.read_sql", return_value=_raw_df_with_embedding()),
    ):
        df = reader.read_tiles()

    assert "embedding" in df.columns
    assert isinstance(df["embedding"].iloc[0], np.ndarray)
    assert df["embedding"].iloc[0].dtype == np.float32


def test_fetch_embeddings_parses_string_to_numpy():
    import numpy as np

    cfg = _default_config(fetch_embeddings=True)
    reader = PostgresReader(cfg)
    mock_psycopg2 = _make_psycopg2_mock()

    with (
        patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        patch("pandas.read_sql", return_value=_raw_df_with_embedding_str()),
    ):
        df = reader.read_tiles()

    assert isinstance(df["embedding"].iloc[0], np.ndarray)
    assert df["embedding"].iloc[0].shape == (4,)


def test_fetch_embeddings_result_not_in_df_when_false():
    cfg = _default_config(fetch_embeddings=False)
    reader = PostgresReader(cfg)
    mock_psycopg2 = _make_psycopg2_mock()

    with (
        patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        patch("pandas.read_sql", return_value=_raw_df()),
    ):
        df = reader.read_tiles()

    assert "embedding" not in df.columns


def test_fetch_embeddings_handles_none_value():
    import numpy as np

    df_with_none = _raw_df()
    df_with_none["embedding"] = [None, [0.5, 0.6, 0.7, 0.8]]
    cfg = _default_config(fetch_embeddings=True)
    reader = PostgresReader(cfg)
    mock_psycopg2 = _make_psycopg2_mock()

    with (
        patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        patch("pandas.read_sql", return_value=df_with_none),
    ):
        df = reader.read_tiles()

    assert df["embedding"].iloc[0] is None
    assert isinstance(df["embedding"].iloc[1], np.ndarray)


def test_fetch_embeddings_parses_ndarray_to_float32():
    import numpy as np

    df_with_ndarray = _raw_df()
    df_with_ndarray["embedding"] = [
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64),
    ]
    cfg = _default_config(fetch_embeddings=True)
    reader = PostgresReader(cfg)
    mock_psycopg2 = _make_psycopg2_mock()

    with (
        patch.dict("sys.modules", {"psycopg2": mock_psycopg2}),
        patch("pandas.read_sql", return_value=df_with_ndarray),
    ):
        df = reader.read_tiles()

    assert df["embedding"].iloc[0].dtype == np.float32
