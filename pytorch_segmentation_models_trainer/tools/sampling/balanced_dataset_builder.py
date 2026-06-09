# -*- coding: utf-8 -*-
"""BalancedDatasetBuilder: orchestrates mask stats, clustering, uniqueness, weighting."""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from pytorch_segmentation_models_trainer.tools.sampling.class_freq_clustering import (
    fit_kmeans_on_clr,
)
from pytorch_segmentation_models_trainer.tools.sampling.mask_stats import (
    compute_mask_stats,
    has_black_border_image,
    has_border_nodata_mask,
)
from pytorch_segmentation_models_trainer.tools.sampling.postgres_reader import (
    PostgresReader,
)
from pytorch_segmentation_models_trainer.tools.sampling.sampling_config import (
    BalancedDatasetConfig,
)
from pytorch_segmentation_models_trainer.tools.sampling.weight_calculator import (
    combine_weights,
)

logger = logging.getLogger(__name__)

_UNIQUENESS_METHODS = {"rank_max", "rank_add", "rank_multiply", "uniqueness_only"}


class BalancedDatasetBuilder:
    """Build a balanced sampling CSV from patch metadata.

    Supports two backends (local / postgres) and three modes
    (masks_only / embeddings_only / both).

    Args:
        config: BalancedDatasetConfig dataclass.
    """

    def __init__(self, config: BalancedDatasetConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """Run the full pipeline and return the output DataFrame.

        Also writes the CSV to config.output.csv_path.

        Returns:
            DataFrame with sampler_weight and all intermediate columns.

        Raises:
            ValueError: if mode/method combination is unsupported.
        """
        self._validate()
        df = self._load_records()
        df = self._compute_mask_stats(df)
        df = self._apply_exclusions(df)
        df = self._compute_clustering(df)

        if self.config.mode in ("both", "embeddings_only"):
            df = self._attach_uniqueness(df)

        df = self._compute_sampler_weights(df)
        self._write_csv(df)
        return df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        cfg = self.config
        needs_uniqueness = cfg.sampling_method in _UNIQUENESS_METHODS
        if cfg.mode == "masks_only" and needs_uniqueness:
            raise ValueError(
                f"sampling_method={cfg.sampling_method!r} requires uniqueness scores, "
                "but mode='masks_only' does not compute them. "
                "Use mode='both' or mode='embeddings_only', or choose "
                "'composition_only' as sampling_method."
            )

    # ------------------------------------------------------------------
    # Record loading
    # ------------------------------------------------------------------

    def _load_records(self) -> pd.DataFrame:
        if self.config.backend == "postgres":
            return self._load_postgres()
        return self._load_local()

    def _load_local(self) -> pd.DataFrame:
        cfg = self.config.input
        if cfg.patch_records is not None:
            return cfg.patch_records.copy()
        raise NotImplementedError(
            "Directory scanning (image_dir/mask_dir) is not yet implemented. "
            "Provide patch_records directly."
        )

    def _load_postgres(self) -> pd.DataFrame:
        reader = PostgresReader(self.config.postgres)
        return reader.read_tiles()

    # ------------------------------------------------------------------
    # Mask stats
    # ------------------------------------------------------------------

    def _compute_mask_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        if "class_dist" in df.columns:
            # Postgres backend already has pre-computed stats
            return df

        exc_cfg = self.config.exclusion

        def _compute_row(row) -> Dict:
            stats = compute_mask_stats(
                row["mask_path"],
                int(row["row_off"]),
                int(row["col_off"]),
                int(row["patch_size"]),
                nodata_class=exc_cfg.nodata_class,
            )
            return stats

        results = [_compute_row(row) for _, row in df.iterrows()]

        df = df.copy()
        df["class_dist"] = [r["class_dist"] for r in results]
        df["class_entropy"] = [r["class_entropy"] for r in results]
        df["nodata_ratio"] = [r["nodata_ratio"] for r in results]
        return df

    # ------------------------------------------------------------------
    # Exclusions
    # ------------------------------------------------------------------

    def _apply_exclusions(self, df: pd.DataFrame) -> pd.DataFrame:
        exc_cfg = self.config.exclusion
        df = df.copy()

        if exc_cfg.exclude_mask_border_nodata:
            import rasterio
            import rasterio.windows

            flags = []
            for _, row in df.iterrows():
                try:
                    window = rasterio.windows.Window(
                        int(row["col_off"]),
                        int(row["row_off"]),
                        int(row["patch_size"]),
                        int(row["patch_size"]),
                    )
                    with rasterio.open(row["mask_path"]) as src:
                        mask_data = src.read(1, window=window)
                    flags.append(
                        has_border_nodata_mask(mask_data, exc_cfg.nodata_class)
                    )
                except Exception:
                    flags.append(False)
            df["has_mask_border_nodata"] = flags
        else:
            df["has_mask_border_nodata"] = False

        if exc_cfg.exclude_image_black_border:
            import rasterio
            import rasterio.windows

            flags = []
            for _, row in df.iterrows():
                try:
                    window = rasterio.windows.Window(
                        int(row["col_off"]),
                        int(row["row_off"]),
                        int(row["patch_size"]),
                        int(row["patch_size"]),
                    )
                    with rasterio.open(row["image_path"]) as src:
                        img_data = src.read(window=window)
                    img_hwc = np.transpose(img_data, (1, 2, 0))
                    flags.append(
                        has_black_border_image(
                            img_hwc, tolerance=exc_cfg.black_border_tolerance
                        )
                    )
                except Exception:
                    flags.append(False)
            df["has_image_black_border"] = flags
        else:
            df["has_image_black_border"] = False

        df["excluded"] = df["has_mask_border_nodata"] | df["has_image_black_border"]
        return df

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _compute_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.mode == "embeddings_only":
            df["freq_cluster_id"] = -1
            df["freq_cluster_size"] = len(df)
            df["freq_cluster_weight"] = 1.0
            return df

        clust_cfg = self.config.clustering

        # Build 6D (or C-D) frequency matrix from class_dist dicts
        all_classes = sorted(
            {
                cls
                for dist in df["class_dist"]
                for cls in (dist.keys() if isinstance(dist, dict) else [])
            }
        )
        if not all_classes:
            df["freq_cluster_id"] = 0
            df["freq_cluster_size"] = len(df)
            df["freq_cluster_weight"] = 1.0 / len(df)
            return df

        n = len(df)
        c = len(all_classes)
        freq_matrix = np.zeros((n, c), dtype=np.float64)
        for i, dist in enumerate(df["class_dist"]):
            if isinstance(dist, dict):
                for j, cls in enumerate(all_classes):
                    freq_matrix[i, j] = dist.get(str(cls), dist.get(cls, 0.0))

        result = fit_kmeans_on_clr(
            freq_matrix,
            k_min=clust_cfg.k_min,
            k_max=clust_cfg.k_max,
            k_selection=clust_cfg.k_selection,
            k=clust_cfg.k,
            random_state=clust_cfg.random_state,
        )

        df = df.copy()
        df["freq_cluster_id"] = result["cluster_ids"]
        df["freq_cluster_size"] = result["cluster_sizes"]
        df["freq_cluster_weight"] = result["cluster_weights"]
        return df

    # ------------------------------------------------------------------
    # Uniqueness
    # ------------------------------------------------------------------

    def _attach_uniqueness(self, df: pd.DataFrame) -> pd.DataFrame:
        if "uniqueness_score" in df.columns:
            return df

        from pytorch_segmentation_models_trainer.tools.sampling.uniqueness import (
            compute_uniqueness,
        )

        emb_cfg = self.config.embeddings
        parquet_path = emb_cfg.parquet_path

        from pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor import (
            extract_embeddings,
            load_embeddings_from_parquet,
            save_embeddings_to_parquet,
        )

        if parquet_path and Path(parquet_path).exists():
            logger.info("Loading embeddings from %s", parquet_path)
            emb_df = load_embeddings_from_parquet(parquet_path)
        else:
            logger.info("Extracting DINOv2 embeddings (%s)…", emb_cfg.dino_model)
            emb_df = extract_embeddings(
                image_paths=df["image_path"].tolist(),
                row_offs=df["row_off"].tolist(),
                col_offs=df["col_off"].tolist(),
                patch_sizes=df["patch_size"].tolist(),
                mask_paths=df["mask_path"].tolist(),
                dino_model=emb_cfg.dino_model,
                batch_size=emb_cfg.batch_size,
                use_gpu=emb_cfg.use_gpu,
                num_workers=emb_cfg.num_workers,
            )
            if parquet_path:
                save_embeddings_to_parquet(emb_df, parquet_path)

        embeddings = np.stack(emb_df["embedding"].values)
        uniq_cfg = self.config.uniqueness
        scores = compute_uniqueness(
            embeddings.astype(np.float32),
            k=uniq_cfg.k_neighbors,
            backend=uniq_cfg.backend,
        )

        df = df.copy()
        df["uniqueness_score"] = scores
        return df

    # ------------------------------------------------------------------
    # Weight combination
    # ------------------------------------------------------------------

    def _compute_sampler_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        method = self.config.sampling_method

        comp_score = df["freq_cluster_weight"].values.astype(np.float64)
        uniq_score = (
            df["uniqueness_score"].values.astype(np.float64)
            if "uniqueness_score" in df.columns
            else np.ones(len(df), dtype=np.float64)
        )

        weights = combine_weights(comp_score, uniq_score, method=method)

        # Ensure every non-excluded patch has a positive weight so
        # WeightedRandomSampler can sample any patch.
        min_weight = 1.0 / max(len(df), 1)
        weights = np.maximum(weights, min_weight)

        # Excluded patches get weight 0; ensure WeightedRandomSampler can still run
        if "excluded" in df.columns:
            weights = np.where(df["excluded"].values, 0.0, weights)
        # Guard: if all weights are 0, set uniform
        if np.all(weights == 0):
            weights = np.ones(len(df), dtype=np.float64)

        df["sampler_weight"] = weights
        return df

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------

    def _write_csv(self, df: pd.DataFrame) -> None:
        out = self.config.output
        Path(out.csv_path).parent.mkdir(parents=True, exist_ok=True)

        out_df = df
        if not out.include_excluded and "excluded" in df.columns:
            out_df = df[~df["excluded"]].reset_index(drop=True)

        # Serialise class_dist dict to JSON string for CSV
        if "class_dist" in out_df.columns:
            out_df = out_df.copy()
            out_df["class_dist_json"] = out_df["class_dist"].apply(
                lambda v: json.dumps(v) if isinstance(v, dict) else str(v)
            )
            out_df = out_df.drop(columns=["class_dist"])

        out_df.to_csv(out.csv_path, index=False)
        logger.info("Wrote %d rows to %s", len(out_df), out.csv_path)

        if out.geojson_path:
            self._write_geojson(df, out)

    def _write_geojson(self, df: pd.DataFrame, out) -> None:
        """Write patch footprints as a GeoJSON for QGIS inspection.

        Each patch is represented as a polygon covering its pixel extent in the
        mask raster's CRS. The sampler_weight and metadata columns are stored as
        feature properties.
        """
        try:
            import rasterio
            import rasterio.windows
            from shapely.geometry import box, mapping
        except ImportError:
            logger.warning("shapely/rasterio not available; skipping GeoJSON output.")
            return

        features = []
        for _, row in df.iterrows():
            try:
                window = rasterio.windows.Window(
                    int(row["col_off"]),
                    int(row["row_off"]),
                    int(row["patch_size"]),
                    int(row["patch_size"]),
                )
                with rasterio.open(row["mask_path"]) as src:
                    bounds = rasterio.windows.bounds(window, src.transform)
            except Exception:
                continue

            props = {
                "image_path": str(row.get("image_path", "")),
                "mask_path": str(row.get("mask_path", "")),
                "row_off": int(row.get("row_off", 0)),
                "col_off": int(row.get("col_off", 0)),
                "patch_size": int(row.get("patch_size", 0)),
                "sampler_weight": float(row.get("sampler_weight", 0.0)),
                "freq_cluster_id": int(row.get("freq_cluster_id", -1)),
                "excluded": bool(row.get("excluded", False)),
            }
            if "uniqueness_score" in row:
                props["uniqueness_score"] = float(row["uniqueness_score"])
            if "nodata_ratio" in row:
                props["nodata_ratio"] = float(row["nodata_ratio"])

            geom = box(bounds[0], bounds[1], bounds[2], bounds[3])
            features.append(
                {"type": "Feature", "geometry": mapping(geom), "properties": props}
            )

        import json as _json

        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": out.geojson_crs},
            },
            "features": features,
        }
        Path(out.geojson_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out.geojson_path).write_text(_json.dumps(geojson, indent=2))
        logger.info("Wrote %d features to %s", len(features), out.geojson_path)
