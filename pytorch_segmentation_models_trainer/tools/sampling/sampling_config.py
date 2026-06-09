# -*- coding: utf-8 -*-
"""Dataclasses for BalancedDatasetBuilder configuration."""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class LocalInputConfig:
    """Configuration for the local backend input.

    Args:
        patch_records: pre-built DataFrame with image_path, mask_path, row_off,
            col_off, patch_size columns.  Used directly when provided.
        image_dir: directory to scan for images (used when patch_records is None).
        mask_dir: directory to scan for masks (used when patch_records is None).
        extension: file extension to scan.
        patch_size: default patch size when scanning directories.
        stride: stride when generating sliding windows.
    """

    patch_records: Optional[Any] = None  # pd.DataFrame
    image_dir: Optional[str] = None
    mask_dir: Optional[str] = None
    extension: str = ".tif"
    patch_size: int = 256
    stride: int = 256


@dataclass
class PostgresConfig:
    """Connection and schema settings for the postgres backend.

    Args:
        host: database host.
        port: database port.
        database: database name.
        user: database user.
        password: database password.
        table: tile table name.
        image_path_column: column containing image file paths.
        mask_path_column: column containing mask file paths.
        row_off_column: column for row offset.
        col_off_column: column for column offset.
        patch_size_column: column for patch size.
        class_dist_column: column for class distribution JSON.
        uniqueness_score_column: column for pre-computed uniqueness score.
        embedding_column: column for pgvector embeddings.
        tile_id_column: primary key column.
        where_clause: optional SQL WHERE clause to filter rows.
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "dataset_explorer"
    user: str = "explorer"
    password: str = "changeme"
    table: str = "tile"
    image_path_column: str = "file_path"
    mask_path_column: str = "mask_path"
    row_off_column: str = "row_off"
    col_off_column: str = "col_off"
    patch_size_column: str = "patch_size"
    class_dist_column: str = "class_dist"
    uniqueness_score_column: str = "uniqueness_score"
    embedding_column: str = "embedding"
    tile_id_column: str = "id"
    where_clause: Optional[str] = None
    fetch_embeddings: bool = False


@dataclass
class ClusteringConfig:
    """K-Means clustering configuration for class frequency vectors.

    Args:
        k_min: minimum K for elbow search.
        k_max: maximum K for elbow search.
        k_selection: 'elbow' for auto-selection; 'fixed' to use k directly.
        k: fixed K value (used when k_selection='fixed').
        random_state: random seed for reproducibility.
    """

    k_min: int = 8
    k_max: int = 20
    k_selection: str = "elbow"
    k: Optional[int] = None
    random_state: int = 42


@dataclass
class UniquenessConfig:
    """Uniqueness score computation configuration.

    Args:
        k_neighbors: number of nearest neighbors.
        backend: 'sklearn' (default, uses all CPU cores via n_jobs=-1),
                 'faiss' (exact search, requires faiss-cpu/faiss-gpu, faster for large N),
                 'hnsw' (approximate, O(N log N), recommended for N > 50k, requires hnswlib).
    """

    k_neighbors: int = 5
    backend: str = "sklearn"


@dataclass
class EmbeddingsConfig:
    """DINOv2 embedding extraction configuration (local backend only).

    Args:
        dino_model: model identifier: 'dinov2-small', 'dinov2-base', 'dinov2-large'.
        batch_size: inference batch size.
        use_gpu: use CUDA when available.
        num_workers: DataLoader worker count for parallel patch loading.
        parquet_path: path to save/load embeddings Parquet file.
    """

    dino_model: str = "dinov2-base"
    batch_size: int = 64
    use_gpu: bool = True
    num_workers: int = 4
    parquet_path: Optional[str] = None


@dataclass
class ExclusionConfig:
    """Border-nodata exclusion settings.

    Args:
        exclude_mask_border_nodata: exclude patches with nodata on mask border.
        exclude_image_black_border: exclude patches with black border on image.
        black_border_tolerance: pixel value at or below this is 'black'.
        nodata_class: mask pixel value treated as no-data.
        min_entropy_threshold: exclude patches whose class entropy is below this
            value (0.0 disables the filter).
        max_nodata_ratio: exclude patches whose nodata pixel ratio exceeds this
            value (1.0 disables the filter).
    """

    exclude_mask_border_nodata: bool = True
    exclude_image_black_border: bool = True
    black_border_tolerance: int = 0
    nodata_class: int = 0
    min_entropy_threshold: float = 0.0
    max_nodata_ratio: float = 1.0


@dataclass
class WeightBoostConfig:
    """Configuration for boosting weights of patches containing rare classes.

    Args:
        rare_class_boost: enable rare-class weight boosting.
        rare_class_ids: explicit list of class ids to treat as rare.  When
            ``None``, classes whose global frequency is below the mean are
            automatically selected.
        rare_class_multiplier: maximum weight multiplier applied to a patch
            that consists entirely of rare-class pixels (default 3.0).
    """

    rare_class_boost: bool = False
    rare_class_ids: Optional[List[int]] = None
    rare_class_multiplier: float = 3.0


@dataclass
class OutputConfig:
    """Output CSV settings.

    Args:
        csv_path: path where the output CSV will be written.
        include_excluded: include excluded patches in the CSV (weight=0).
        geojson_path: optional path for a GeoJSON with patch footprints and
            sampler_weight as a property, for inspection in QGIS or other GIS tools.
        geojson_crs: EPSG code of the mask rasters (used to project footprints).
    """

    csv_path: str = "balanced_dataset.csv"
    include_excluded: bool = True
    geojson_path: Optional[str] = None
    geojson_crs: str = "EPSG:4326"


@dataclass
class BalancedDatasetConfig:
    """Top-level configuration for BalancedDatasetBuilder.

    Args:
        backend: 'local' or 'postgres'.
        mode: 'masks_only', 'embeddings_only', or 'both'.
        sampling_method: weight combination method.
        num_workers: number of threads for mask stats and exclusion loops.
        input: local input config.
        postgres: postgres connection config.
        clustering: K-Means clustering config.
        uniqueness: uniqueness scoring config.
        embeddings: DINOv2 extraction config.
        exclusion: exclusion filter config.
        boost: rare-class weight boosting config.
        output: output CSV config.
    """

    backend: str = "local"
    mode: str = "masks_only"
    sampling_method: str = "rank_max"
    num_workers: int = 8
    input: LocalInputConfig = field(default_factory=LocalInputConfig)
    postgres: Optional[PostgresConfig] = None
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    uniqueness: UniquenessConfig = field(default_factory=UniquenessConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    exclusion: ExclusionConfig = field(default_factory=ExclusionConfig)
    boost: WeightBoostConfig = field(default_factory=WeightBoostConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
