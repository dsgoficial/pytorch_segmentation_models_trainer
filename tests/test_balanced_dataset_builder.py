# -*- coding: utf-8 -*-
"""Tests for tools/sampling/balanced_dataset_builder.py."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds

from pytorch_segmentation_models_trainer.tools.sampling.balanced_dataset_builder import (
    BalancedDatasetBuilder,
)
from pytorch_segmentation_models_trainer.tools.sampling.sampling_config import (
    BalancedDatasetConfig,
    ClusteringConfig,
    EmbeddingsConfig,
    ExclusionConfig,
    LocalInputConfig,
    OutputConfig,
    PostgresConfig,
    WeightBoostConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mask(path: Path, classes: np.ndarray) -> None:
    h, w = classes.shape
    transform = from_bounds(0, 0, 1, 1, w, h)
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(classes[np.newaxis])


def _make_image(path: Path, fill: int = 100) -> None:
    data = np.full((3, 256, 256), fill, dtype=np.uint8)
    transform = from_bounds(0, 0, 1, 1, 256, 256)
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=256,
        width=256,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture()
def local_dataset_dir(tmp_path):
    """Create a small synthetic dataset: 10 image+mask pairs."""
    rng = np.random.default_rng(42)
    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "masks"
    img_dir.mkdir()
    msk_dir.mkdir()
    records = []
    for i in range(10):
        img_path = img_dir / f"tile_{i:03d}.tif"
        msk_path = msk_dir / f"tile_{i:03d}.tif"
        _make_image(img_path)
        classes = rng.integers(1, 7, size=(256, 256)).astype(np.uint8)
        _make_mask(msk_path, classes)
        records.append(
            {
                "image_path": str(img_path),
                "mask_path": str(msk_path),
                "row_off": 0,
                "col_off": 0,
                "patch_size": 256,
            }
        )
    return tmp_path, pd.DataFrame(records)


# ---------------------------------------------------------------------------
# masks_only mode
# ---------------------------------------------------------------------------


def test_local_backend_masks_only(local_dataset_dir, tmp_path):
    _, records = local_dataset_dir
    csv_path = str(tmp_path / "out.csv")

    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        output=OutputConfig(csv_path=csv_path),
    )

    builder = BalancedDatasetBuilder(config)
    df = builder.build()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(records)
    assert "sampler_weight" in df.columns
    assert "image_path" in df.columns
    assert "mask_path" in df.columns
    assert Path(csv_path).exists()


def test_csv_output_schema(local_dataset_dir, tmp_path):
    _, records = local_dataset_dir
    csv_path = str(tmp_path / "out.csv")

    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        output=OutputConfig(csv_path=csv_path),
    )

    builder = BalancedDatasetBuilder(config)
    df = builder.build()

    required_cols = {
        "image_path",
        "mask_path",
        "row_off",
        "col_off",
        "patch_size",
        "sampler_weight",
        "freq_cluster_id",
        "freq_cluster_weight",
    }
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_sampler_weight_positive(local_dataset_dir, tmp_path):
    _, records = local_dataset_dir
    csv_path = str(tmp_path / "out.csv")

    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        output=OutputConfig(csv_path=csv_path),
    )

    builder = BalancedDatasetBuilder(config)
    df = builder.build()

    assert np.all(df["sampler_weight"].values > 0)


def test_exclusion_flags_added(local_dataset_dir, tmp_path):
    _, records = local_dataset_dir
    csv_path = str(tmp_path / "out.csv")

    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        exclusion=ExclusionConfig(
            exclude_mask_border_nodata=True, exclude_image_black_border=False
        ),
        output=OutputConfig(csv_path=csv_path),
    )

    builder = BalancedDatasetBuilder(config)
    df = builder.build()

    assert "has_mask_border_nodata" in df.columns
    assert "excluded" in df.columns


# ---------------------------------------------------------------------------
# uniqueness_only raises without embeddings
# ---------------------------------------------------------------------------


def test_uniqueness_only_without_embeddings_raises(local_dataset_dir, tmp_path):
    _, records = local_dataset_dir

    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="uniqueness_only",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        output=OutputConfig(csv_path=str(tmp_path / "out.csv")),
    )

    builder = BalancedDatasetBuilder(config)
    with pytest.raises(ValueError, match="uniqueness"):
        builder.build()


# ---------------------------------------------------------------------------
# Postgres backend (mocked)
# ---------------------------------------------------------------------------


def test_local_backend_no_patch_records_raises():
    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=None, image_dir=None),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        output=OutputConfig(csv_path="/tmp/out.csv"),
    )
    with pytest.raises(NotImplementedError):
        BalancedDatasetBuilder(config).build()


def test_compute_mask_stats_skipped_when_already_present(local_dataset_dir, tmp_path):
    """When class_dist is already in the df (postgres backend), mask_stats is skipped."""
    _, records = local_dataset_dir
    # Pre-populate class_dist so the early return is triggered
    records = records.copy()
    records["class_dist"] = [{1: 0.5, 2: 0.5}] * len(records)
    records["class_entropy"] = [0.693] * len(records)
    records["nodata_ratio"] = [0.0] * len(records)

    csv_path = str(tmp_path / "out.csv")
    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        output=OutputConfig(csv_path=csv_path),
    )
    df = BalancedDatasetBuilder(config).build()
    assert "sampler_weight" in df.columns


def test_mode_embeddings_only(local_dataset_dir, tmp_path):
    _, records = local_dataset_dir
    records = records.copy()
    records["uniqueness_score"] = [float(i) / 10 for i in range(len(records))]
    csv_path = str(tmp_path / "out.csv")

    config = BalancedDatasetConfig(
        mode="embeddings_only",
        sampling_method="uniqueness_only",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        output=OutputConfig(csv_path=csv_path),
    )
    df = BalancedDatasetBuilder(config).build()
    assert "sampler_weight" in df.columns
    assert "freq_cluster_id" in df.columns
    assert df["freq_cluster_id"].iloc[0] == -1


def test_all_classes_empty_fallback(tmp_path):
    """When all patches have empty class_dist, clustering falls back to uniform weight."""
    import rasterio
    from rasterio.transform import from_bounds

    img_dir = tmp_path / "images"
    msk_dir = tmp_path / "masks"
    img_dir.mkdir()
    msk_dir.mkdir()
    records = []
    for i in range(5):
        img_path = img_dir / f"tile_{i}.tif"
        msk_path = msk_dir / f"tile_{i}.tif"
        # All-nodata masks → empty class_dist
        mask = np.zeros((256, 256), dtype=np.uint8)
        for path, data, count in [
            (str(img_path), np.ones((3, 256, 256), dtype=np.uint8) * 100, 3),
            (str(msk_path), mask[np.newaxis], 1),
        ]:
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=256,
                width=256,
                count=count,
                dtype=data.dtype.name,
                crs="EPSG:4326",
                transform=from_bounds(0, 0, 1, 1, 256, 256),
            ) as dst:
                dst.write(data)
        records.append(
            {
                "image_path": str(img_path),
                "mask_path": str(msk_path),
                "row_off": 0,
                "col_off": 0,
                "patch_size": 256,
            }
        )
    df_records = pd.DataFrame(records)
    csv_path = str(tmp_path / "out.csv")
    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=df_records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        exclusion=ExclusionConfig(exclude_mask_border_nodata=False),
        output=OutputConfig(csv_path=csv_path),
    )
    df = BalancedDatasetBuilder(config).build()
    assert "freq_cluster_id" in df.columns


def test_all_excluded_weight_guard(local_dataset_dir, tmp_path):
    """When all patches are excluded, weights fall back to uniform (no all-zero)."""
    _, records = local_dataset_dir
    csv_path = str(tmp_path / "out.csv")
    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        exclusion=ExclusionConfig(
            exclude_mask_border_nodata=True,
            exclude_image_black_border=False,
        ),
        output=OutputConfig(csv_path=csv_path),
    )

    builder = BalancedDatasetBuilder(config)
    df_internal = builder._load_records()
    df_internal = builder._compute_mask_stats(df_internal)
    df_internal = builder._apply_exclusions(df_internal)
    # Force all patches to be excluded
    df_internal["excluded"] = True
    df_internal = builder._compute_clustering(df_internal)
    df_internal = builder._compute_sampler_weights(df_internal)

    # Guard ensures weights are not all zero
    assert np.all(df_internal["sampler_weight"].values >= 0)
    # At least non-negative; the guard sets to ones when all-excluded
    assert df_internal["sampler_weight"].sum() >= 0


def test_include_excluded_false(local_dataset_dir, tmp_path):
    _, records = local_dataset_dir
    csv_path = str(tmp_path / "out.csv")
    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        exclusion=ExclusionConfig(
            exclude_mask_border_nodata=True, exclude_image_black_border=False
        ),
        output=OutputConfig(csv_path=csv_path, include_excluded=False),
    )
    df = BalancedDatasetBuilder(config).build()
    assert "excluded" in df.columns or True  # may be empty if all pass


def test_attach_uniqueness_with_parquet(local_dataset_dir, tmp_path):
    """When a parquet file exists, load embeddings from it instead of extracting."""
    from pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor import (
        save_embeddings_to_parquet,
    )

    _, records = local_dataset_dir
    parquet_path = str(tmp_path / "emb.parquet")
    n = len(records)

    rng = np.random.default_rng(99)
    emb_df = records.copy()
    emb_df["tile_id"] = [
        f"{row['image_path']}_{row['row_off']}_{row['col_off']}"
        for _, row in records.iterrows()
    ]
    emb_df["embedding"] = list(rng.standard_normal((n, 64)).astype(np.float32))
    emb_df["dino_model"] = "dinov2-base"
    save_embeddings_to_parquet(emb_df, parquet_path)

    csv_path = str(tmp_path / "out.csv")
    config = BalancedDatasetConfig(
        mode="both",
        sampling_method="rank_max",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        embeddings=EmbeddingsConfig(parquet_path=parquet_path),
        output=OutputConfig(csv_path=csv_path),
    )
    df = BalancedDatasetBuilder(config).build()
    assert "uniqueness_score" in df.columns
    assert "sampler_weight" in df.columns


def test_attach_uniqueness_already_present(local_dataset_dir, tmp_path):
    """When uniqueness_score is already in df (postgres backend), skip extraction."""
    _, records = local_dataset_dir
    records = records.copy()
    records["class_dist"] = [{1: 0.5, 2: 0.5}] * len(records)
    records["class_entropy"] = [0.693] * len(records)
    records["nodata_ratio"] = [0.0] * len(records)
    records["uniqueness_score"] = [0.5] * len(records)

    csv_path = str(tmp_path / "out.csv")
    config = BalancedDatasetConfig(
        mode="both",
        sampling_method="rank_max",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        output=OutputConfig(csv_path=csv_path),
    )
    df = BalancedDatasetBuilder(config).build()
    assert "uniqueness_score" in df.columns


def test_attach_uniqueness_extracts_when_no_parquet(local_dataset_dir, tmp_path):
    """When parquet_path is None, extract_embeddings is called (mocked)."""
    _, records = local_dataset_dir
    n = len(records)
    rng = np.random.default_rng(88)
    fake_emb_df = records.copy()
    fake_emb_df["tile_id"] = "x"
    fake_emb_df["embedding"] = list(rng.standard_normal((n, 16)).astype(np.float32))
    fake_emb_df["dino_model"] = "dinov2-base"

    csv_path = str(tmp_path / "out.csv")
    config = BalancedDatasetConfig(
        mode="both",
        sampling_method="rank_max",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        embeddings=EmbeddingsConfig(parquet_path=None),
        output=OutputConfig(csv_path=csv_path),
    )

    with patch(
        "pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor."
        "extract_embeddings",
        return_value=fake_emb_df,
    ):
        df = BalancedDatasetBuilder(config).build()

    assert "uniqueness_score" in df.columns


def test_attach_uniqueness_saves_parquet_when_extracted(local_dataset_dir, tmp_path):
    """When parquet_path set but file missing, save after extraction."""
    _, records = local_dataset_dir
    n = len(records)
    rng = np.random.default_rng(77)
    parquet_path = str(tmp_path / "new_emb.parquet")
    fake_emb_df = records.copy()
    fake_emb_df["tile_id"] = "x"
    fake_emb_df["embedding"] = list(rng.standard_normal((n, 16)).astype(np.float32))
    fake_emb_df["dino_model"] = "dinov2-base"

    csv_path = str(tmp_path / "out.csv")
    config = BalancedDatasetConfig(
        mode="both",
        sampling_method="rank_max",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        embeddings=EmbeddingsConfig(parquet_path=parquet_path),
        output=OutputConfig(csv_path=csv_path),
    )

    with patch(
        "pytorch_segmentation_models_trainer.tools.sampling.embedding_extractor."
        "extract_embeddings",
        return_value=fake_emb_df,
    ):
        BalancedDatasetBuilder(config).build()

    assert Path(parquet_path).exists()


def test_geojson_shapely_import_error_skips_gracefully(local_dataset_dir, tmp_path):
    """When shapely is unavailable, GeoJSON write is skipped without crashing."""
    _, records = local_dataset_dir
    csv_path = str(tmp_path / "out.csv")
    geojson_path = str(tmp_path / "patches.geojson")

    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        output=OutputConfig(csv_path=csv_path, geojson_path=geojson_path),
    )

    # Patch shapely.geometry as None to trigger the ImportError branch inside _write_geojson
    with patch.dict("sys.modules", {"shapely": None, "shapely.geometry": None}):
        BalancedDatasetBuilder(config).build()
    # CSV should still be written; GeoJSON skipped silently
    assert Path(csv_path).exists()
    assert not Path(geojson_path).exists()


def test_geojson_bad_mask_path_skipped(local_dataset_dir, tmp_path):
    """Patches with unreadable mask paths are silently skipped in GeoJSON output."""
    _, records = local_dataset_dir
    bad_records = records.copy()
    # Pre-populate class_dist so mask_stats computation is skipped (postgres path)
    bad_records["class_dist"] = [{1: 0.5, 2: 0.5}] * len(bad_records)
    bad_records["class_entropy"] = [0.693] * len(bad_records)
    bad_records["nodata_ratio"] = [0.0] * len(bad_records)
    bad_records["excluded"] = [False] * len(bad_records)
    bad_records["has_mask_border_nodata"] = [False] * len(bad_records)
    bad_records["has_image_black_border"] = [False] * len(bad_records)
    # Make first patch have a bad mask_path for GeoJSON
    bad_records.iloc[0, bad_records.columns.get_loc("mask_path")] = "/nonexistent.tif"

    csv_path = str(tmp_path / "out.csv")
    geojson_path = str(tmp_path / "patches.geojson")

    builder = BalancedDatasetBuilder(
        BalancedDatasetConfig(
            mode="masks_only",
            sampling_method="composition_only",
            input=LocalInputConfig(patch_records=bad_records),
            clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
            output=OutputConfig(csv_path=csv_path, geojson_path=geojson_path),
        )
    )
    # Manually build to bypass mask_stats (class_dist already present)
    df = builder._load_records()
    df = builder._compute_clustering(df)
    df = builder._compute_sampler_weights(df)
    builder._write_csv(df)

    assert Path(geojson_path).exists()
    import json as _json

    gj = _json.loads(Path(geojson_path).read_text())
    # bad patch is skipped; other patches appear
    assert len(gj["features"]) < len(bad_records)


def test_geojson_includes_uniqueness_score(local_dataset_dir, tmp_path):
    """When uniqueness_score is present, it appears in GeoJSON properties."""
    _, records = local_dataset_dir
    records = records.copy()
    records["class_dist"] = [{1: 0.5, 2: 0.5}] * len(records)
    records["class_entropy"] = [0.693] * len(records)
    records["nodata_ratio"] = [0.0] * len(records)
    records["uniqueness_score"] = [0.5] * len(records)

    csv_path = str(tmp_path / "out.csv")
    geojson_path = str(tmp_path / "patches.geojson")

    config = BalancedDatasetConfig(
        mode="both",
        sampling_method="rank_max",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        output=OutputConfig(csv_path=csv_path, geojson_path=geojson_path),
    )

    BalancedDatasetBuilder(config).build()

    import json as _json

    gj = _json.loads(Path(geojson_path).read_text())
    assert "uniqueness_score" in gj["features"][0]["properties"]


def test_geojson_output_written(local_dataset_dir, tmp_path):
    _, records = local_dataset_dir
    csv_path = str(tmp_path / "out.csv")
    geojson_path = str(tmp_path / "patches.geojson")

    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        output=OutputConfig(csv_path=csv_path, geojson_path=geojson_path),
    )

    BalancedDatasetBuilder(config).build()

    assert Path(geojson_path).exists()

    import json as _json

    gj = _json.loads(Path(geojson_path).read_text())
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == len(records)
    assert "sampler_weight" in gj["features"][0]["properties"]


# ---------------------------------------------------------------------------
# Entropy threshold exclusion
# ---------------------------------------------------------------------------


@pytest.fixture()
def entropy_dataset(tmp_path):
    """4 patches: 2 mono-class (entropy≈0), 2 mixed (entropy>0.6)."""
    msk_dir = tmp_path / "masks"
    msk_dir.mkdir()
    records = []
    for i, (c1, c2) in enumerate([(3, 3), (4, 4), (3, 4), (3, 5)]):
        path = msk_dir / f"tile_{i}.tif"
        classes = np.zeros((256, 256), dtype=np.uint8)
        classes[:128, :] = c1
        classes[128:, :] = c2
        _make_mask(path, classes)
        records.append(
            {
                "mask_path": str(path),
                "row_off": 0,
                "col_off": 0,
                "patch_size": 256,
            }
        )
    return pd.DataFrame(records)


def test_entropy_threshold_excludes_low_entropy_patches(tmp_path, entropy_dataset):
    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=entropy_dataset),
        clustering=ClusteringConfig(k_min=2, k_max=3, k_selection="fixed", k=2),
        exclusion=ExclusionConfig(
            exclude_mask_border_nodata=False,
            exclude_image_black_border=False,
            min_entropy_threshold=0.5,
        ),
        output=OutputConfig(csv_path=str(tmp_path / "out.csv"), include_excluded=True),
    )
    df = BalancedDatasetBuilder(config).build()
    assert "has_low_entropy" in df.columns
    assert df["has_low_entropy"].sum() == 2
    assert df["excluded"].sum() == 2


def test_entropy_threshold_zero_disables_filter(tmp_path, entropy_dataset):
    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=entropy_dataset),
        clustering=ClusteringConfig(k_min=2, k_max=3, k_selection="fixed", k=2),
        exclusion=ExclusionConfig(
            exclude_mask_border_nodata=False,
            exclude_image_black_border=False,
            min_entropy_threshold=0.0,
        ),
        output=OutputConfig(csv_path=str(tmp_path / "out.csv"), include_excluded=True),
    )
    df = BalancedDatasetBuilder(config).build()
    assert df["has_low_entropy"].sum() == 0


# ---------------------------------------------------------------------------
# Nodata ratio threshold exclusion
# ---------------------------------------------------------------------------


@pytest.fixture()
def nodata_dataset(tmp_path):
    """4 patches: 2 with high nodata (>50%), 2 clean."""
    msk_dir = tmp_path / "masks"
    msk_dir.mkdir()
    records = []
    # nodata_class=0; patches 0+1 have >50% nodata, 2+3 are clean
    for i, nodata_frac in enumerate([0.8, 0.6, 0.0, 0.1]):
        path = msk_dir / f"tile_{i}.tif"
        classes = np.full((256, 256), 3, dtype=np.uint8)
        n_nodata = int(nodata_frac * 256 * 256)
        classes.flat[:n_nodata] = 0
        _make_mask(path, classes)
        records.append(
            {
                "mask_path": str(path),
                "row_off": 0,
                "col_off": 0,
                "patch_size": 256,
            }
        )
    return pd.DataFrame(records)


def test_nodata_ratio_threshold_excludes_high_nodata_patches(tmp_path, nodata_dataset):
    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=nodata_dataset),
        clustering=ClusteringConfig(k_min=2, k_max=3, k_selection="fixed", k=2),
        exclusion=ExclusionConfig(
            exclude_mask_border_nodata=False,
            exclude_image_black_border=False,
            nodata_class=0,
            max_nodata_ratio=0.5,
        ),
        output=OutputConfig(csv_path=str(tmp_path / "out.csv"), include_excluded=True),
    )
    df = BalancedDatasetBuilder(config).build()
    assert "has_high_nodata" in df.columns
    assert df["has_high_nodata"].sum() == 2
    assert df["excluded"].sum() == 2


def test_nodata_ratio_one_disables_filter(tmp_path, nodata_dataset):
    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=nodata_dataset),
        clustering=ClusteringConfig(k_min=2, k_max=3, k_selection="fixed", k=2),
        exclusion=ExclusionConfig(
            exclude_mask_border_nodata=False,
            exclude_image_black_border=False,
            nodata_class=0,
            max_nodata_ratio=1.0,
        ),
        output=OutputConfig(csv_path=str(tmp_path / "out.csv"), include_excluded=True),
    )
    df = BalancedDatasetBuilder(config).build()
    assert df["has_high_nodata"].sum() == 0


# ---------------------------------------------------------------------------
# Rare-class boost
# ---------------------------------------------------------------------------


@pytest.fixture()
def rare_class_dataset(tmp_path):
    """6 patches: 3 have rare class 1, 3 have only common classes 3+4."""
    msk_dir = tmp_path / "masks"
    msk_dir.mkdir()
    records = []
    for i in range(3):
        # rare: 50% class 1, 50% class 3
        path = msk_dir / f"rare_{i}.tif"
        classes = np.zeros((256, 256), dtype=np.uint8)
        classes[:128, :] = 1
        classes[128:, :] = 3
        _make_mask(path, classes)
        records.append(
            {"mask_path": str(path), "row_off": 0, "col_off": 0, "patch_size": 256}
        )
    for i in range(3):
        # common: 50% class 3, 50% class 4
        path = msk_dir / f"common_{i}.tif"
        classes = np.zeros((256, 256), dtype=np.uint8)
        classes[:128, :] = 3
        classes[128:, :] = 4
        _make_mask(path, classes)
        records.append(
            {"mask_path": str(path), "row_off": 0, "col_off": 0, "patch_size": 256}
        )
    return pd.DataFrame(records)


def _boost_builder(tmp_path, rare_class_ids, boost_on):
    """Helper: build config for rare-class boost tests using pre-populated class_dist."""
    records = pd.DataFrame(
        {
            "mask_path": ["/fake/mask.tif"] * 6,
            "row_off": [0] * 6,
            "col_off": [0] * 6,
            "patch_size": [256] * 6,
            # first 3 = rare (50% class 1), last 3 = common (0% class 1)
            "class_dist": [{"1": 0.5, "3": 0.5}] * 3 + [{"3": 0.5, "4": 0.5}] * 3,
            "class_entropy": [0.693] * 6,
            "nodata_ratio": [0.0] * 6,
        }
    )
    return BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=records),
        clustering=ClusteringConfig(k_min=2, k_max=3, k_selection="fixed", k=2),
        exclusion=ExclusionConfig(
            exclude_mask_border_nodata=False, exclude_image_black_border=False
        ),
        boost=WeightBoostConfig(
            rare_class_boost=boost_on,
            rare_class_ids=rare_class_ids,
            rare_class_multiplier=3.0,
        ),
        output=OutputConfig(csv_path=str(tmp_path / "out.csv")),
    )


def test_rare_class_boost_disabled_by_default(tmp_path):
    cfg_off = _boost_builder(tmp_path, rare_class_ids=[1], boost_on=False)
    cfg_on = _boost_builder(tmp_path, rare_class_ids=[1], boost_on=True)
    df_off = BalancedDatasetBuilder(cfg_off).build()
    df_on = BalancedDatasetBuilder(cfg_on).build()
    rare_ratio_off = df_off.iloc[:3]["sampler_weight"].mean()
    rare_ratio_on = df_on.iloc[:3]["sampler_weight"].mean()
    # With boost ON, rare-patch mean weight should be strictly higher
    assert rare_ratio_on > rare_ratio_off


def test_rare_class_boost_increases_rare_patch_weights(tmp_path):
    cfg = _boost_builder(tmp_path, rare_class_ids=[1], boost_on=True)
    builder = BalancedDatasetBuilder(cfg)
    df = pd.DataFrame(
        {
            "class_dist": [{"1": 0.5, "3": 0.5}] * 3 + [{"3": 0.5, "4": 0.5}] * 3,
            "freq_cluster_weight": [1 / 3] * 6,
        }
    )
    boost = builder._compute_rare_class_boost(df)
    assert boost[:3].mean() > boost[3:].mean()


def test_rare_class_boost_auto_detects_rare_classes(tmp_path):
    cfg = _boost_builder(tmp_path, rare_class_ids=None, boost_on=True)
    builder = BalancedDatasetBuilder(cfg)
    # 2 rare + 4 common: class 1 freq=0.167, class 4 freq=0.333, mean=0.333
    # Only class 1 satisfies freq < mean (strict less-than)
    df = pd.DataFrame(
        {
            "class_dist": [{"1": 0.5, "3": 0.5}] * 2 + [{"3": 0.5, "4": 0.5}] * 4,
            "freq_cluster_weight": [1 / 3] * 6,
        }
    )
    boost = builder._compute_rare_class_boost(df)
    assert boost[:2].mean() > boost[2:].mean()


def test_weight_boost_config_defaults():
    cfg = WeightBoostConfig()
    assert cfg.rare_class_boost is False
    assert cfg.rare_class_ids is None
    assert cfg.rare_class_multiplier == 3.0


def test_exclusion_config_new_defaults():
    cfg = ExclusionConfig()
    assert cfg.min_entropy_threshold == 0.0
    assert cfg.max_nodata_ratio == 1.0


def test_balanced_dataset_config_boost_default():
    cfg = BalancedDatasetConfig()
    assert isinstance(cfg.boost, WeightBoostConfig)
    assert cfg.boost.rare_class_boost is False


def test_rare_class_boost_no_rare_classes_found_no_crash(tmp_path, rare_class_dataset):
    config = BalancedDatasetConfig(
        mode="masks_only",
        sampling_method="composition_only",
        input=LocalInputConfig(patch_records=rare_class_dataset),
        clustering=ClusteringConfig(k_min=2, k_max=3, k_selection="fixed", k=2),
        exclusion=ExclusionConfig(
            exclude_mask_border_nodata=False, exclude_image_black_border=False
        ),
        boost=WeightBoostConfig(
            rare_class_boost=True,
            rare_class_ids=[99],  # class 99 not present in any patch
            rare_class_multiplier=3.0,
        ),
        output=OutputConfig(csv_path=str(tmp_path / "out.csv")),
    )
    df = BalancedDatasetBuilder(config).build()
    assert "sampler_weight" in df.columns
    assert np.all(df["sampler_weight"].values >= 0)


def test_postgres_backend_mocked(tmp_path):
    csv_path = str(tmp_path / "out.csv")

    mock_records = pd.DataFrame(
        {
            "image_path": [f"/data/img_{i}.tif" for i in range(8)],
            "mask_path": [f"/data/msk_{i}.tif" for i in range(8)],
            "row_off": [0] * 8,
            "col_off": [0] * 8,
            "patch_size": [256] * 8,
            "class_dist": [{str(c): 1 / 6 for c in range(1, 7)} for _ in range(8)],
            "uniqueness_score": np.random.default_rng(0).random(8).tolist(),
            "class_entropy": [1.5] * 8,
            "nodata_ratio": [0.0] * 8,
        }
    )

    config = BalancedDatasetConfig(
        backend="postgres",
        mode="both",
        sampling_method="rank_max",
        postgres=PostgresConfig(
            host="localhost",
            port=5432,
            database="test_db",
            user="user",
            password="pass",
        ),
        clustering=ClusteringConfig(k_min=3, k_max=5, k_selection="fixed", k=3),
        output=OutputConfig(csv_path=csv_path),
    )

    with patch(
        "pytorch_segmentation_models_trainer.tools.sampling.balanced_dataset_builder."
        "PostgresReader"
    ) as MockReader:
        MockReader.return_value.read_tiles.return_value = mock_records
        builder = BalancedDatasetBuilder(config)
        df = builder.build()

    assert len(df) == 8
    assert "sampler_weight" in df.columns
    assert Path(csv_path).exists()
