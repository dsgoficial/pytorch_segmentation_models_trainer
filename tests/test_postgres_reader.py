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
