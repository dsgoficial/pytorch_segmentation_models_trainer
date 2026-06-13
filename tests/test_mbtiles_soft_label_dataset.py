# -*- coding: utf-8 -*-
"""Tests for MBTilesSoftLabelMaskWindowedDataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from rasterio.transform import from_origin

from pytorch_segmentation_models_trainer.custom_losses.soft_label_loss import (
    SoftLabelWeightedCELoss,
)
from pytorch_segmentation_models_trainer.dataset_loader.mbtiles_soft_label_dataset import (
    MBTilesSoftLabelMaskWindowedDataset,
)
from pytorch_segmentation_models_trainer.model_loader.soft_label_model import (
    SoftLabelModel,
)

H, W, C = 32, 32, 4


def _write_raster(path: Path, data: np.ndarray) -> None:
    if data.ndim == 2:
        data = data[np.newaxis]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=str(data.dtype),
        crs="EPSG:3857",
        transform=from_origin(0, data.shape[1], 1, 1),
    ) as dst:
        dst.write(data)


@pytest.fixture()
def soft_mbtiles_setup(tmp_path):
    yy, xx = np.indices((H * 2, W * 2))
    image = np.stack(
        [
            (xx % 255).astype(np.uint8),
            (yy % 255).astype(np.uint8),
            ((xx + yy) % 255).astype(np.uint8),
        ]
    )
    mask = ((xx // 8 + yy // 8) % C).astype(np.uint8)
    lulc_a = ((xx // 16) % C).astype(np.uint8)
    lulc_b = ((yy // 16) % C).astype(np.uint8)

    image_path = tmp_path / "image.tif"
    mask_path = tmp_path / "mask.tif"
    lulc_a_path = tmp_path / "lulc_a.tif"
    lulc_b_path = tmp_path / "lulc_b.tif"
    cache_path = tmp_path / "windows.csv"

    _write_raster(image_path, image)
    _write_raster(mask_path, mask)
    _write_raster(lulc_a_path, lulc_a)
    _write_raster(lulc_b_path, lulc_b)

    pd.DataFrame(
        [
            {"mask_path": str(mask_path), "row_off": 0, "col_off": 0, "patch_size": H},
            {
                "mask_path": str(mask_path),
                "row_off": H,
                "col_off": W,
                "patch_size": H,
            },
        ]
    ).to_csv(cache_path, index=False)

    return {
        "image": image_path,
        "mask": mask_path,
        "lulc_paths": [lulc_a_path, lulc_b_path],
        "cache": cache_path,
    }


def _make_dataset(setup, **kwargs):
    params = {
        "mbtiles_path": setup["image"],
        "mask_paths": [setup["mask"]],
        "lulc_paths": setup["lulc_paths"],
        "window_index_cache": setup["cache"],
        "num_classes": C,
        "return_metadata": False,
    }
    params.update(kwargs)
    return MBTilesSoftLabelMaskWindowedDataset(**params)


class TestMBTilesSoftLabelMaskWindowedDataset:
    def test_contract_shape_dtype_and_probabilities(self, soft_mbtiles_setup):
        ds = _make_dataset(soft_mbtiles_setup)
        item = ds[0]

        assert set(item) == {"image", "mask", "path"}
        assert item["image"].shape == (3, H, W)
        assert item["image"].dtype == torch.float32
        assert item["mask"]["mask"].shape == (C, H, W)
        assert item["mask"]["w_conf"].shape == (1, H, W)
        torch.testing.assert_close(
            item["mask"]["mask"].sum(dim=0),
            torch.ones(H, W),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_without_w_conf_returns_only_soft_mask(self, soft_mbtiles_setup):
        ds = _make_dataset(soft_mbtiles_setup, return_w_conf=False)
        item = ds[0]

        assert set(item["mask"]) == {"mask"}

    def test_batch_sizes(self, soft_mbtiles_setup):
        ds = _make_dataset(soft_mbtiles_setup)
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))

        assert batch["image"].shape == (2, 3, H, W)
        assert batch["mask"]["mask"].shape == (2, C, H, W)
        assert batch["mask"]["w_conf"].shape == (2, 1, H, W)

    def test_gradient_flow_with_soft_label_loss(self, soft_mbtiles_setup):
        ds = _make_dataset(soft_mbtiles_setup)
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))
        model = nn.Conv2d(3, C, kernel_size=1)
        loss, _ = SoftLabelWeightedCELoss(num_classes=C)(
            model(batch["image"]), batch["mask"]
        )
        loss.backward()

        assert model.weight.grad is not None
        assert torch.count_nonzero(model.weight.grad) > 0

    def test_index_error_raised_for_out_of_bounds(self, soft_mbtiles_setup):
        ds = _make_dataset(soft_mbtiles_setup)
        with pytest.raises(IndexError):
            _ = ds[999]

    def test_lightning_model_integration(self, soft_mbtiles_setup):
        dataset_cfg = {
            "_target_": (
                "pytorch_segmentation_models_trainer.dataset_loader"
                ".mbtiles_soft_label_dataset.MBTilesSoftLabelMaskWindowedDataset"
            ),
            "mbtiles_path": str(soft_mbtiles_setup["image"]),
            "mask_paths": [str(soft_mbtiles_setup["mask"])],
            "lulc_paths": [str(p) for p in soft_mbtiles_setup["lulc_paths"]],
            "window_index_cache": str(soft_mbtiles_setup["cache"]),
            "num_classes": C,
            "return_metadata": False,
            "data_loader": {
                "shuffle": False,
                "num_workers": 0,
                "pin_memory": False,
                "drop_last": False,
            },
        }
        cfg = OmegaConf.create(
            {
                "model": {
                    "_target_": "torch.nn.Conv2d",
                    "in_channels": 3,
                    "out_channels": C,
                    "kernel_size": 1,
                },
                "loss": {
                    "_target_": (
                        "pytorch_segmentation_models_trainer.custom_losses"
                        ".soft_label_loss.SoftLabelWeightedCELoss"
                    ),
                    "num_classes": C,
                    "mask_key": "mask",
                    "weight_key": "w_conf",
                },
                "hyperparameters": {"batch_size": 2},
                "train_dataset": dataset_cfg,
                "val_dataset": dataset_cfg,
            }
        )

        lightning_model = SoftLabelModel(cfg, inference_mode=False)
        batch = next(iter(lightning_model.train_dataloader()))
        loss = lightning_model.training_step(batch, 0)
        assert loss.ndim == 0


class TestMBTilesSoftLabelCaching:
    """P_soft disk cache: lazy population, atomic write, cache-hit skips LULC reads."""

    def test_no_cache_files_when_cache_dir_is_none(self, soft_mbtiles_setup, tmp_path):
        ds = _make_dataset(soft_mbtiles_setup)  # cache_dir defaults to None
        _ = ds[0]
        assert not list(tmp_path.glob("**/*.npy"))

    def test_cache_dir_created_automatically(self, soft_mbtiles_setup, tmp_path):
        cache_dir = tmp_path / "new" / "nested" / "cache"
        _make_dataset(soft_mbtiles_setup, cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_cache_file_created_on_first_access(self, soft_mbtiles_setup, tmp_path):
        cache_dir = tmp_path / "cache"
        ds = _make_dataset(soft_mbtiles_setup, cache_dir=cache_dir)
        _ = ds[0]
        assert len(list(cache_dir.glob("*.npy"))) == 1

    def test_different_windows_get_different_cache_files(self, soft_mbtiles_setup, tmp_path):
        cache_dir = tmp_path / "cache"
        ds = _make_dataset(soft_mbtiles_setup, cache_dir=cache_dir)
        _ = ds[0]
        _ = ds[1]
        assert len(list(cache_dir.glob("*.npy"))) == 2

    def test_no_tmp_files_left_after_write(self, soft_mbtiles_setup, tmp_path):
        cache_dir = tmp_path / "cache"
        ds = _make_dataset(soft_mbtiles_setup, cache_dir=cache_dir)
        _ = ds[0]
        assert not list(cache_dir.glob("*.tmp"))

    def test_cached_p_soft_matches_uncached(self, soft_mbtiles_setup, tmp_path):
        cache_dir = tmp_path / "cache"
        ds_no_cache = _make_dataset(soft_mbtiles_setup, return_w_conf=False)
        ds_cached = _make_dataset(
            soft_mbtiles_setup, return_w_conf=False, cache_dir=cache_dir
        )
        torch.testing.assert_close(
            ds_no_cache[0]["mask"]["mask"],
            ds_cached[0]["mask"]["mask"],
        )

    def test_cache_hit_returns_identical_p_soft(self, soft_mbtiles_setup, tmp_path):
        cache_dir = tmp_path / "cache"
        ds = _make_dataset(soft_mbtiles_setup, return_w_conf=False, cache_dir=cache_dir)
        torch.testing.assert_close(
            ds[0]["mask"]["mask"],
            ds[0]["mask"]["mask"],
        )

    def test_cache_hit_skips_lulc_reads(self, soft_mbtiles_setup, tmp_path):
        """After populating the cache, pointing lulc_paths to non-existent files still works."""
        from pathlib import Path as _Path

        cache_dir = tmp_path / "cache"
        ds = _make_dataset(soft_mbtiles_setup, return_w_conf=False, cache_dir=cache_dir)
        _ = ds[0]  # populate cache
        ds.lulc_paths = [_Path("/nonexistent/lulc.tif")]  # would fail on LULC read
        item = ds[0]  # must succeed via cache
        assert item["mask"]["mask"].shape == (C, H, W)

    def test_w_conf_computed_correctly_from_cached_p_soft(
        self, soft_mbtiles_setup, tmp_path
    ):
        cache_dir = tmp_path / "cache"
        ds_ref = _make_dataset(soft_mbtiles_setup, return_w_conf=True)
        ds_cache = _make_dataset(
            soft_mbtiles_setup, return_w_conf=True, cache_dir=cache_dir
        )
        _ = ds_cache[0]  # populate cache
        torch.testing.assert_close(
            ds_ref[0]["mask"]["w_conf"],
            ds_cache[0]["mask"]["w_conf"],
        )
