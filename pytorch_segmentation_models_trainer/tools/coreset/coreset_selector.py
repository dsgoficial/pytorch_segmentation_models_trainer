# -*- coding: utf-8 -*-
"""CoreSetSelector: orchestrates core-set scoring and budget selection."""

import numpy as np
import pandas as pd

from pytorch_segmentation_models_trainer.tools.coreset.coreset_config import (
    CoreSetConfig,
)
from pytorch_segmentation_models_trainer.tools.coreset.coreset_scorer import get_scorer


class CoreSetSelector:
    """Select a core-set from a balanced dataset DataFrame.

    Takes the output of ``BalancedDatasetBuilder`` (``balanced_dataset.csv``)
    and adds two columns: ``coreset_score`` (float in [0, 1]) and
    ``coreset_selected`` (bool).  The output CSV is written to
    ``config.output_csv_path``.

    Args:
        config: ``CoreSetConfig`` controlling method, budget, and parameters.

    Example:

        .. code-block:: python

            import pandas as pd
            from pytorch_segmentation_models_trainer.tools.coreset import (
                CoreSetConfig, CoreSetSelector
            )

            df = pd.read_csv("balanced_dataset.csv")
            cfg = CoreSetConfig(method="lc", budget=0.5)
            result = CoreSetSelector(cfg).select(df)
            # result has coreset_score and coreset_selected columns
    """

    def __init__(self, config: CoreSetConfig) -> None:
        self.config = config

    def _resolve_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge embeddings from Parquet if embedding_column is absent from df.

        Only runs when all three conditions hold:
          1. ``config.embedding_column`` is set.
          2. The column is NOT already in df.
          3. ``config.parquet_path`` is set.

        Merges on ``(image_path, row_off, col_off)`` — the join keys present in
        both ``balanced_dataset.csv`` and the Parquet written by
        ``embedding_extractor.save_embeddings_to_parquet``.

        Args:
            df: input DataFrame.

        Returns:
            df with the embedding column added (or unchanged if not needed).
        """
        cfg = self.config
        if cfg.embedding_column is None:
            return df
        if cfg.embedding_column in df.columns:
            return df
        if cfg.parquet_path is None:
            return df

        from pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor import (
            load_embeddings_from_parquet,
        )

        emb_df = load_embeddings_from_parquet(cfg.parquet_path)
        join_cols = ["image_path", "row_off", "col_off"]
        emb_cols = join_cols + ["embedding"]
        emb_subset = emb_df[[c for c in emb_cols if c in emb_df.columns]].rename(
            columns={"embedding": cfg.embedding_column}
        )
        return df.merge(emb_subset, on=join_cols, how="left")

    def _apply_exclusion_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply hard exclusion filters before scoring.

        Always excludes patches flagged by ``has_mask_border_nodata``,
        ``has_image_black_border``, or ``has_high_nodata`` when those columns
        are present.  Excludes ``has_low_entropy`` patches only when
        ``config.exclude_low_entropy`` is True.

        Args:
            df: full pool DataFrame.

        Returns:
            Filtered DataFrame (same columns, subset of rows).
        """
        keep = pd.Series(True, index=df.index)
        for col in (
            "has_mask_border_nodata",
            "has_image_black_border",
            "has_high_nodata",
        ):
            if col in df.columns:
                keep &= ~df[col].astype(bool)
        if self.config.exclude_low_entropy and "has_low_entropy" in df.columns:
            keep &= ~df["has_low_entropy"].astype(bool)
        return df[keep].copy()

    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all patches and select the top-budget subset.

        Applies hard exclusion filters (nodata, black border) and optionally
        the entropy filter (``config.exclude_low_entropy``) before scoring.
        If ``config.embedding_column`` is set but not present in df, and
        ``config.parquet_path`` points to a Parquet file produced by
        ``embedding_extractor``, the embeddings are loaded and merged
        automatically before scoring.

        Args:
            df: input DataFrame (from ``balanced_dataset.csv`` or equivalent).

        Returns:
            Copy of df with ``coreset_score`` and ``coreset_selected`` columns
            added.  The file at ``config.output_csv_path`` is written.
        """
        df = self._apply_exclusion_filters(df.copy())
        df = self._resolve_embeddings(df)
        N = len(df)

        scorer = get_scorer(self.config.method)
        scores = scorer(df, self.config)

        if self.config.budget_mode == "fraction":
            budget_count = max(1, int(self.config.budget * N))
        else:
            budget_count = max(1, int(self.config.budget))
        budget_count = min(budget_count, N)

        selected_idx = np.argsort(-scores)[:budget_count]
        selected_mask = np.zeros(N, dtype=bool)
        selected_mask[selected_idx] = True

        df[self.config.score_column] = scores
        df["coreset_selected"] = selected_mask

        df.to_csv(self.config.output_csv_path, index=False)
        return df
