# -*- coding: utf-8 -*-
"""Read patch metadata from dataset_explorer's PostgreSQL tile table."""

import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_embedding(v) -> np.ndarray:
    """Convert a pgvector value to a float32 numpy array.

    Handles three representations returned by psycopg2:
    - Python list / numpy ndarray (registered pgvector adapter)
    - JSON string ``"[0.1, 0.2, ...]"`` (default pgvector text repr)
    - None → returns None (caller must handle missing embeddings)
    """
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v.astype(np.float32)
    if isinstance(v, list):
        return np.array(v, dtype=np.float32)
    # String form: "[0.1,0.2,0.3]"
    return np.array(json.loads(v), dtype=np.float32)


class PostgresReader:
    """Read tile records from a dataset_explorer PostgreSQL database.

    Args:
        config: PostgresConfig dataclass with connection parameters.
    """

    def __init__(self, config):
        self.config = config

    def read_tiles(self) -> pd.DataFrame:
        """Query the tile table and return a DataFrame of patch records.

        Returns:
            DataFrame with at minimum: image_path, mask_path, row_off, col_off,
            patch_size, class_dist (dict), uniqueness_score, class_entropy,
            nodata_ratio.

        Raises:
            ImportError: if psycopg2 is not installed.
        """
        try:
            import psycopg2
        except ImportError as exc:
            raise ImportError(
                "psycopg2-binary is required for the postgres backend. "
                "Install with: pip install psycopg2-binary"
            ) from exc

        cfg = self.config
        conn = psycopg2.connect(
            host=cfg.host,
            port=cfg.port,
            database=cfg.database,
            user=cfg.user,
            password=cfg.password,
        )

        col_list = [
            cfg.image_path_column,
            cfg.mask_path_column,
            cfg.row_off_column,
            cfg.col_off_column,
            cfg.patch_size_column,
            cfg.class_dist_column,
            cfg.uniqueness_score_column,
            "class_entropy",
            "nodata_ratio",
        ]
        if cfg.fetch_embeddings:
            col_list.append(cfg.embedding_column)

        cols = ", ".join(col_list)
        query = f"SELECT {cols} FROM {cfg.table}"
        if cfg.where_clause:
            query += f" WHERE {cfg.where_clause}"

        logger.info("PostgresReader: %s", query)
        df = pd.read_sql(query, conn)
        conn.close()

        # Rename columns to canonical names
        rename = {
            cfg.image_path_column: "image_path",
            cfg.mask_path_column: "mask_path",
            cfg.row_off_column: "row_off",
            cfg.col_off_column: "col_off",
            cfg.patch_size_column: "patch_size",
            cfg.class_dist_column: "class_dist",
            cfg.uniqueness_score_column: "uniqueness_score",
        }
        df = df.rename(columns=rename)

        # Parse class_dist JSON strings if needed (handles both object and str dtypes)
        df["class_dist"] = df["class_dist"].apply(
            lambda v: json.loads(v) if isinstance(v, str) else v
        )

        if cfg.fetch_embeddings and cfg.embedding_column in df.columns:
            df[cfg.embedding_column] = df[cfg.embedding_column].apply(_parse_embedding)

        return df
