# -*- coding: utf-8 -*-
"""Dataclass for CoreSetSelector configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CoreSetConfig:
    """Configuration for CoreSetSelector (Nogueira et al. 2026).

    Args:
        method: scoring method — ``'lc'`` | ``'cb'`` | ``'fa'`` | ``'fd'``
            | ``'lc_fd'`` | ``'fa_cb'``.
        budget: fraction [0, 1] of patches to select when
            ``budget_mode='fraction'``; integer count when
            ``budget_mode='count'``.
        budget_mode: ``'fraction'`` or ``'count'``.
        embedding_column: DataFrame column holding pre-computed embeddings
            (required for FA and FD methods; raise ValueError if None).
        fd_k_min: minimum K for elbow / Vendi K-selection in FD.
        fd_k_max: maximum K for elbow / Vendi K-selection in FD.
        fd_vendi_delta: convergence threshold for Vendi K-search.
        fd_use_vendi: use Vendi score for K-selection instead of elbow.
        lc_fd_cutoff_m: number of top-FD patches to retain before LC fill;
            defaults to ``int(0.1 * N)`` when ``None``.
        fa_cb_lambda: mixing coefficient for FA/CB hybrid
            (``λ × FA + (1-λ) × CB``).
        score_column: name of the output score column.
        output_csv_path: path where the output CSV will be written.
        input_csv_path: path to the input balanced_dataset.csv (used by CLI).

    Example YAML (``select-coreset`` command):

        .. code-block:: yaml

            coreset:
              method: lc
              budget: 0.5
              budget_mode: fraction
              input_csv_path: /path/to/balanced_dataset.csv
              output_csv_path: /path/to/coreset.csv
    """

    method: str = "lc"
    budget: float = 0.5
    budget_mode: str = "fraction"

    embedding_column: Optional[str] = None

    fd_k_min: int = 2
    fd_k_max: int = 20
    fd_vendi_delta: float = 0.005
    fd_use_vendi: bool = False

    lc_fd_cutoff_m: Optional[int] = None

    fa_cb_lambda: float = 0.5

    score_column: str = "coreset_score"
    output_csv_path: str = "coreset.csv"
    input_csv_path: str = ""

    parquet_path: Optional[str] = None
    # Path to a Parquet file produced by embedding_extractor.py.
    # When set and embedding_column is not present in the input DataFrame,
    # CoreSetSelector merges the embeddings automatically before scoring.
    # Required for FA, FD, LC/FD, and FA/CB methods when embeddings are not
    # already in the input CSV.
