# -*- coding: utf-8 -*-
"""Tests for MBTilesMaskWindowedDataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import MissingMandatoryValue, OmegaConf

from pytorch_segmentation_models_trainer.config_definitions import (
    dataset_config,
)
from pytorch_segmentation_models_trainer.dataset_loader import (
    mbtiles_mask_dataset,
)
from pytorch_segmentation_models_trainer.model_loader.model import Model

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

pytestmark = pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")


def _write_rgb_raster(path: Path, width=32, height=32, dtype="uint8"):
    yy, xx = np.indices((height, width))
    data = np.stack(
        [
            xx.astype(dtype),
            yy.astype(dtype),
            ((xx + yy) % 255).astype(dtype),
        ]
    )
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": dtype,
        "crs": "EPSG:3857",
        "transform": from_origin(0, height, 1, 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _write_mask(path: Path, width=32, height=32, values=(0, 2)):
    mask = np.zeros((height, width), dtype=np.uint8)
    if values[1] is not None:
        mask[8:24, 8:24] = values[1]
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": from_origin(0, height, 1, 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask, 1)


def test_dataset_returns_expected_contract(tmp_path):
    source = tmp_path / "source.tif"
    mask = tmp_path / "masks" / "mask.tif"
    _write_rgb_raster(source)
    _write_mask(mask)

    ds = mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
        mbtiles_path=source,
        mask_paths=[mask],
        patch_size=16,
        stride=16,
    )

    item = ds[0]
    assert len(ds) == 4
    assert set(item) == {"image", "mask", "mask_path", "row_off", "col_off"}
    assert item["image"].shape == (3, 16, 16)
    assert item["mask"].shape == (16, 16)
    assert item["image"].dtype == torch.float32
    assert item["mask"].dtype == torch.int64
    assert item["image"].min() >= 0
    assert item["image"].max() <= 1


def test_dataset_batch_sizes_and_degenerate_empty_mask(tmp_path):
    source = tmp_path / "source.tif"
    mask = tmp_path / "masks" / "empty.tif"
    _write_rgb_raster(source)
    _write_mask(mask, values=(0, None))

    ds = mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
        mbtiles_path=source,
        mask_paths=[mask],
        patch_size=16,
        stride=16,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=3)
    batch = next(iter(loader))

    assert batch["image"].shape == (3, 3, 16, 16)
    assert batch["mask"].shape == (3, 16, 16)
    assert torch.count_nonzero(batch["mask"]) == 0


def test_multiclass_masks_are_preserved(tmp_path):
    source = tmp_path / "source.tif"
    mask = tmp_path / "masks" / "mask.tif"
    _write_rgb_raster(source)
    _write_mask(mask)

    ds = mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
        mbtiles_path=source,
        mask_paths=[mask],
        patch_size=32,
        stride=32,
        n_classes=3,
    )

    assert torch.max(ds[0]["mask"]) == 2


def test_gradient_flow_with_dataset_batch(tmp_path):
    source = tmp_path / "source.tif"
    mask = tmp_path / "masks" / "mask.tif"
    _write_rgb_raster(source)
    _write_mask(mask)
    ds = mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
        mbtiles_path=source,
        mask_paths=[mask],
        patch_size=16,
        stride=16,
    )
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))
    model = nn.Conv2d(3, 2, kernel_size=1)
    loss = nn.CrossEntropyLoss()(model(batch["image"]), batch["mask"])
    loss.backward()

    assert model.weight.grad is not None
    assert torch.count_nonzero(model.weight.grad) > 0


def test_hydra_config_and_model_integration(tmp_path):
    source = tmp_path / "source.tif"
    mask = tmp_path / "masks" / "mask.tif"
    _write_rgb_raster(source)
    _write_mask(mask)

    dataset_cfg = {
        "_target_": (
            "pytorch_segmentation_models_trainer.dataset_loader"
            ".mbtiles_mask_dataset.MBTilesMaskWindowedDataset"
        ),
        "mbtiles_path": str(source),
        "mask_paths": [str(mask)],
        "patch_size": 16,
        "stride": 16,
        "data_loader": {
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": False,
        },
    }
    ds = instantiate(OmegaConf.create(dataset_cfg), _recursive_=False)
    assert isinstance(ds, mbtiles_mask_dataset.MBTilesMaskWindowedDataset)

    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "torch.nn.Conv2d",
                "in_channels": 3,
                "out_channels": 2,
                "kernel_size": 1,
            },
            "loss": {"_target_": "torch.nn.CrossEntropyLoss"},
            "hyperparameters": {"batch_size": 2},
            "train_dataset": dataset_cfg,
            "val_dataset": dataset_cfg,
        }
    )
    lightning_model = Model(cfg, inference_mode=False)
    batch = next(iter(lightning_model.train_dataloader()))
    loss = lightning_model.training_step(batch, 0)

    assert loss.ndim == 0


def test_dataset_config_dataclass():
    cfg = OmegaConf.structured(dataset_config.MBTilesMaskWindowedDatasetConfig)
    assert "MBTilesMaskWindowedDataset" in cfg._target_
    assert cfg.patch_size == 512
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.mbtiles_path


def test_invalid_modes_raise(tmp_path):
    source = tmp_path / "source.tif"
    small_mask = tmp_path / "masks" / "small.tif"
    _write_rgb_raster(source)
    _write_mask(small_mask, width=8, height=8)

    with pytest.raises(ValueError, match="No mask files found"):
        mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
            mbtiles_path=source,
            mask_dir=tmp_path / "missing",
            patch_size=16,
        )

    with pytest.raises(ValueError, match="No valid mask windows"):
        mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
            mbtiles_path=source,
            mask_paths=[small_mask],
            patch_size=16,
        )


def test_window_index_cache_csv_is_written_and_reused(tmp_path):
    source = tmp_path / "source.tif"
    mask = tmp_path / "masks" / "mask.tif"
    cache = tmp_path / "window_index.csv"
    _write_rgb_raster(source)
    _write_mask(mask)

    ds = mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
        mbtiles_path=source,
        mask_paths=[mask],
        patch_size=16,
        stride=16,
        window_index_cache=cache,
    )
    assert len(ds) == 4
    assert cache.exists()

    mask.unlink()
    cached = mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
        mbtiles_path=source,
        mask_paths=[mask],
        patch_size=16,
        stride=16,
        window_index_cache=cache,
    )

    assert len(cached) == 4
    assert cached.windows[0]["mask_path"] == mask


def test_window_index_cache_accepts_patch_size_column(tmp_path):
    source = tmp_path / "source.tif"
    mask = tmp_path / "masks" / "mask.tif"
    cache = tmp_path / "coreset_windows.csv"
    _write_rgb_raster(source)
    _write_mask(mask)
    pd.DataFrame(
        [
            {
                "mask_path": str(mask),
                "row_off": 8,
                "col_off": 8,
                "patch_size": 16,
                "class_entropy": 0.5,
            }
        ]
    ).to_csv(cache, index=False)

    ds = mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
        mbtiles_path=source,
        mask_paths=[mask],
        window_index_cache=cache,
    )

    item = ds[0]
    assert len(ds) == 1
    assert item["image"].shape == (3, 16, 16)
    assert item["mask"].shape == (16, 16)
    assert ds.windows[0]["width"] == 16
    assert ds.windows[0]["height"] == 16


def test_window_index_cache_accepts_world_bounds(tmp_path):
    source = tmp_path / "source.tif"
    mask = tmp_path / "masks" / "mask.tif"
    cache = tmp_path / "bounds_windows.csv"
    _write_rgb_raster(source)
    _write_mask(mask)
    pd.DataFrame(
        [
            {
                "image_path": str(mask),
                "minx": 8,
                "miny": 8,
                "maxx": 24,
                "maxy": 24,
                "crs": "EPSG:3857",
            }
        ]
    ).to_csv(cache, index=False)

    ds = mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
        mbtiles_path=source,
        window_index_cache=cache,
        window_index_mask_path_key="image_path",
        window_index_coordinate_mode="bounds",
    )

    item = ds[0]
    assert len(ds) == 1
    assert item["image"].shape == (3, 16, 16)
    assert item["mask"].shape == (16, 16)
    assert ds.windows[0]["row_off"] == 8
    assert ds.windows[0]["col_off"] == 8
    assert ds.windows[0]["width"] == 16
    assert ds.windows[0]["height"] == 16


def test_window_index_cache_accepts_tile_world_bounds_alias(tmp_path):
    source = tmp_path / "source.tif"
    mask = tmp_path / "masks" / "mask.tif"
    cache = tmp_path / "tile_bounds_windows.csv"
    _write_rgb_raster(source)
    _write_mask(mask)
    pd.DataFrame(
        [
            {
                "mask_path": str(mask),
                "tile_minx": 8,
                "tile_miny": 8,
                "tile_maxx": 24,
                "tile_maxy": 24,
            }
        ]
    ).to_csv(cache, index=False)

    ds = mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
        mbtiles_path=source,
        window_index_cache=cache,
        window_index_coordinate_mode="bounds",
    )

    assert ds.windows[0]["row_off"] == 8
    assert ds.windows[0]["col_off"] == 8
    assert ds.windows[0]["width"] == 16
    assert ds.windows[0]["height"] == 16


def test_window_index_cache_parquet_is_written(tmp_path):
    source = tmp_path / "source.tif"
    mask = tmp_path / "masks" / "mask.tif"
    cache = tmp_path / "window_index.parquet"
    _write_rgb_raster(source)
    _write_mask(mask)

    ds = mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
        mbtiles_path=source,
        mask_paths=[mask],
        patch_size=16,
        stride=16,
        window_index_cache=cache,
    )

    assert len(ds) == 4
    assert cache.exists()
    assert len(pd.read_parquet(cache)) == 4


def test_mask_base_path_resolves_relative_paths(tmp_path):
    source = tmp_path / "source.tif"
    masks_dir = tmp_path / "masks"
    mask = masks_dir / "mask.tif"
    cache = tmp_path / "window_index.csv"
    _write_rgb_raster(source)
    _write_mask(mask)

    pd.DataFrame(
        [
            {
                "mask_path": "mask.tif",
                "row_off": 0,
                "col_off": 0,
                "width": 16,
                "height": 16,
            }
        ]
    ).to_csv(cache, index=False)

    ds = mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
        mbtiles_path=source,
        window_index_cache=str(cache),
        mask_base_path=str(masks_dir),
        patch_size=16,
        stride=16,
    )

    assert len(ds) == 1
    assert ds.windows[0]["mask_path"] == masks_dir / "mask.tif"
    item = ds[0]
    assert item["image"].shape == (3, 16, 16)
    assert item["mask"].shape == (16, 16)


def test_mask_base_path_does_not_affect_absolute_paths(tmp_path):
    source = tmp_path / "source.tif"
    masks_dir = tmp_path / "masks"
    mask = masks_dir / "mask.tif"
    cache = tmp_path / "window_index.csv"
    _write_rgb_raster(source)
    _write_mask(mask)

    pd.DataFrame(
        [
            {
                "mask_path": str(mask),
                "row_off": 0,
                "col_off": 0,
                "width": 16,
                "height": 16,
            }
        ]
    ).to_csv(cache, index=False)

    # Pass a different base_path; absolute path in CSV must still be used.
    ds = mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
        mbtiles_path=source,
        window_index_cache=str(cache),
        mask_base_path="/some/other/dir",
        patch_size=16,
        stride=16,
    )

    assert ds.windows[0]["mask_path"] == mask


def test_df_attribute_set_when_cache_exists(tmp_path):
    source = tmp_path / "source.tif"
    mask = tmp_path / "masks" / "mask.tif"
    cache = tmp_path / "coreset_v8.csv"
    _write_rgb_raster(source)
    _write_mask(mask)
    pd.DataFrame(
        [
            {
                "mask_path": str(mask),
                "row_off": 0,
                "col_off": 0,
                "patch_size": 16,
                "sampler_weight": 0.25,
            },
            {
                "mask_path": str(mask),
                "row_off": 16,
                "col_off": 0,
                "patch_size": 16,
                "sampler_weight": 0.10,
            },
        ]
    ).to_csv(cache, index=False)

    ds = mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
        mbtiles_path=source,
        window_index_cache=cache,
    )

    assert ds.df is not None
    assert "sampler_weight" in ds.df.columns
    assert len(ds.df) == 2
    assert list(ds.df["sampler_weight"]) == [0.25, 0.10]


def test_df_is_none_without_cache(tmp_path):
    source = tmp_path / "source.tif"
    mask = tmp_path / "masks" / "mask.tif"
    _write_rgb_raster(source)
    _write_mask(mask)

    ds = mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
        mbtiles_path=source,
        mask_paths=[mask],
        patch_size=16,
        stride=16,
    )

    assert ds.df is None


def test_invalid_window_index_cache_extension_raises(tmp_path):
    source = tmp_path / "source.tif"
    mask = tmp_path / "masks" / "mask.tif"
    _write_rgb_raster(source)
    _write_mask(mask)

    with pytest.raises(ValueError, match="window_index_cache"):
        mbtiles_mask_dataset.MBTilesMaskWindowedDataset(
            mbtiles_path=source,
            mask_paths=[mask],
            patch_size=16,
            stride=16,
            window_index_cache=tmp_path / "index.json",
        )
