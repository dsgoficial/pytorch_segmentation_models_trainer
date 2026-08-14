# -*- coding: utf-8 -*-
"""Hybrid coreset selection combining spatial vector intersection and embedding diversity.

Orchestrates multi-step coreset selection: vector-intersection steps for rare
classes, FD / LC-FD embedding steps for common classes, and an entropy sweep
to fill any remaining budget.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Set

import pandas as pd

from pytorch_segmentation_models_trainer.tools.coreset.vector_selector import (
    entropy_sweep_select,
    fd_embedding_select,
    lc_fd_select,
    select_by_vector_intersection,
)


@dataclass
class VectorSelectionStep:
    """One spatial vector-intersection selection step.

    Args:
        gpkg_layer: Layer name inside the GeoPackage to load.
        class_indices: Column names used to identify the target class set
            (for documentation purposes; not used in spatial filter itself).
        min_intersect_area_m2: Minimum intersection area for a patch to be
            selected (in the CRS unit squared, typically m²).

    Example YAML::

        vector_steps:
          - gpkg_layer: grassland_polygons
            class_indices: [c3]
            min_intersect_area_m2: 1000.0
    """

    gpkg_layer: str
    class_indices: list
    min_intersect_area_m2: float = 1000.0


@dataclass
class EmbeddingSelectionStep:
    """One embedding-based FD or LC/FD selection step.

    Args:
        class_filter: Dict mapping column name to ``(operator, threshold)`` tuple.
            E.g. ``{"c4": (">", 0.50)}`` selects rows where ``c4 > 0.50``.
        budget: Maximum number of patches to select in this step.
        k: KMeans cluster count for FD / LC-FD.
        method: ``"fd"`` (Facility-Location diversity) or ``"lc_fd"``
            (top-entropy + FD).
        lc_fraction: Fraction of budget for the LC (high-entropy) phase
            when ``method="lc_fd"``.

    Example YAML::

        embedding_steps:
          - class_filter: {c4: [">", 0.50]}
            budget: 500
            k: 10
            method: lc_fd
            lc_fraction: 0.40
    """

    class_filter: dict
    budget: int
    k: int
    method: str = "fd"
    lc_fraction: float = 0.40


@dataclass
class HybridVectorCoresetConfig:
    """Configuration for :class:`HybridVectorCoresetSelector`.

    Args:
        input_csv_path: Path to the full pool CSV.
        embeddings_parquet: Path to Parquet file with embedding vectors
            (joined to pool by index).
        vectors_gpkg: Path to GeoPackage file with reference vector layers.
        output_csv_path: Where to write the selected coreset CSV.
        pool_fraction: Fraction of the pool to select as the total budget.
        vector_steps: List of :class:`VectorSelectionStep` dicts.
        embedding_steps: List of :class:`EmbeddingSelectionStep` dicts.
        entropy_sweep: If True, fill remaining budget with top-entropy patches.
        crs: CRS string for building pool geometries from bbox columns.
        bbox_columns: Column names for ``[minx, miny, maxx, maxy]`` bboxes.
        embedding_col: Column name for embedding vectors.
        entropy_col: Column name for entropy values.
        random_state: Random state for reproducibility.

    Example YAML::

        input_csv_path: /data/pool.csv
        embeddings_parquet: /data/embeddings.parquet
        vectors_gpkg: /data/reference.gpkg
        output_csv_path: /data/coreset.csv
        pool_fraction: 0.30
        vector_steps:
          - gpkg_layer: grassland
            class_indices: [c3]
            min_intersect_area_m2: 1000.0
        embedding_steps:
          - class_filter: {c4: [">", 0.50]}
            budget: 500
            k: 10
            method: fd
        entropy_sweep: true
    """

    input_csv_path: str
    embeddings_parquet: str
    vectors_gpkg: str
    output_csv_path: str
    pool_fraction: float = 0.30
    vector_steps: list = field(default_factory=list)
    embedding_steps: list = field(default_factory=list)
    entropy_sweep: bool = True
    crs: str = "EPSG:3857"
    bbox_columns: list = field(
        default_factory=lambda: ["tile_minx", "tile_miny", "tile_maxx", "tile_maxy"]
    )
    embedding_col: str = "embedding"
    entropy_col: str = "class_entropy"
    random_state: int = 42


class HybridVectorCoresetSelector:
    """Orchestrates multi-step hybrid coreset selection.

    Selection proceeds in three phases:

    1. **Vector steps** — spatial intersection with GeoPackage layers.
    2. **Embedding steps** — FD or LC-FD on filtered subsets of the pool.
    3. **Entropy sweep** — top-entropy patches fill any remaining budget.

    Args:
        config: :class:`HybridVectorCoresetConfig` instance.

    Example::

        cfg = HybridVectorCoresetConfig(
            input_csv_path="/data/pool.csv",
            embeddings_parquet="/data/emb.parquet",
            vectors_gpkg="/data/ref.gpkg",
            output_csv_path="/data/coreset.csv",
        )
        selector = HybridVectorCoresetSelector(cfg)
        pool_df = pd.read_csv(cfg.input_csv_path)
        result = selector.select(pool_df)
    """

    def __init__(self, config: HybridVectorCoresetConfig) -> None:
        self._cfg = config

    def select(self, pool_df: pd.DataFrame) -> pd.DataFrame:
        """Run the full selection pipeline on ``pool_df``.

        Args:
            pool_df: Pool DataFrame. Must contain bbox columns, embedding column,
                and entropy column as specified in the config.

        Returns:
            Copy of ``pool_df`` with two new columns:

            - ``coreset_selected`` (int 0/1)
            - ``selection_step`` (str label, NaN for unselected)

            Also writes the result to ``config.output_csv_path``.
        """
        import geopandas as gpd
        from shapely.geometry import box

        result = pool_df.copy()
        result["coreset_selected"] = 0
        result["selection_step"] = pd.Series(pd.NA, index=result.index, dtype=object)

        budget = int(self._cfg.pool_fraction * len(pool_df))
        selected: Set[int] = set()

        # Build pool GeoDataFrame for spatial steps
        bx = self._cfg.bbox_columns
        pool_gdf = gpd.GeoDataFrame(
            pool_df,
            geometry=[
                box(row[bx[0]], row[bx[1]], row[bx[2]], row[bx[3]])
                for _, row in pool_df.iterrows()
            ],
            crs=self._cfg.crs,
        )

        # Phase 1: vector steps
        for step_dict in self._cfg.vector_steps:
            if len(selected) >= budget:
                break
            step = (
                VectorSelectionStep(**step_dict)
                if isinstance(step_dict, dict)
                else step_dict
            )
            vec_gdf = self._load_gpkg_layer(step.gpkg_layer)
            idx = select_by_vector_intersection(
                pool_gdf,
                vec_gdf,
                min_area_m2=step.min_intersect_area_m2,
                exclude_idx=selected,
            )
            new_idx = set(idx.tolist()) - selected
            for i in new_idx:
                result.at[i, "coreset_selected"] = 1
                result.at[i, "selection_step"] = f"vector:{step.gpkg_layer}"
            selected |= new_idx

        # Phase 2: embedding steps
        for step_num, step_dict in enumerate(self._cfg.embedding_steps):
            if len(selected) >= budget:
                break
            step = (
                EmbeddingSelectionStep(**step_dict)
                if isinstance(step_dict, dict)
                else step_dict
            )

            # Apply class filter to get the sub-pool for this step
            sub = pool_df.copy()
            for col, (op, thresh) in step.class_filter.items():
                if op == ">":
                    sub = sub[sub[col] > thresh]
                elif op == ">=":
                    sub = sub[sub[col] >= thresh]
                elif op == "<":
                    sub = sub[sub[col] < thresh]
                elif op == "<=":
                    sub = sub[sub[col] <= thresh]

            remaining_budget = min(step.budget, budget - len(selected))
            if step.method == "fd":
                idx = fd_embedding_select(
                    sub,
                    k=step.k,
                    budget=remaining_budget,
                    exclude_idx=selected,
                    embedding_col=self._cfg.embedding_col,
                    random_state=self._cfg.random_state,
                )
            else:
                idx = lc_fd_select(
                    sub,
                    k=step.k,
                    budget=remaining_budget,
                    exclude_idx=selected,
                    lc_fraction=step.lc_fraction,
                    embedding_col=self._cfg.embedding_col,
                    entropy_col=self._cfg.entropy_col,
                    random_state=self._cfg.random_state,
                )

            label = f"embedding:{step.method}:{step_num}"
            new_idx = set(idx.tolist()) - selected
            for i in new_idx:
                result.at[i, "coreset_selected"] = 1
                result.at[i, "selection_step"] = label
            selected |= new_idx

        # Phase 3: entropy sweep
        if self._cfg.entropy_sweep and len(selected) < budget:
            remaining = budget - len(selected)
            idx = entropy_sweep_select(
                pool_df,
                budget=remaining,
                exclude_idx=selected,
                entropy_col=self._cfg.entropy_col,
            )
            new_idx = set(idx.tolist()) - selected
            for i in new_idx:
                result.at[i, "coreset_selected"] = 1
                result.at[i, "selection_step"] = "entropy_sweep"
            selected |= new_idx

        Path(self._cfg.output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        result.drop(columns=["geometry"], errors="ignore").to_csv(
            self._cfg.output_csv_path, index=False
        )
        return result

    def _load_gpkg_layer(self, layer_name: str):
        """Load a GeoPackage layer. Separated for easy mocking in tests."""
        import geopandas as gpd

        return gpd.read_file(self._cfg.vectors_gpkg, layer=layer_name)
