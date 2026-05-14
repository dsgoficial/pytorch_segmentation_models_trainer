# -*- coding: utf-8 -*-
from contextlib import contextmanager
import json

import numpy as np
import pytest
import torch
import rasterio
from pathlib import Path
from rasterio.transform import from_bounds
from torch.utils.data import DataLoader
from pytorch_segmentation_models_trainer.dataset_loader import (
    image_dataset as image_dataset_module,
)
from pytorch_segmentation_models_trainer.dataset_loader.image_dataset import (
    IterableWindowedImageAutoencoderDataset,
    IterableWindowedImageDataset,
    WindowedImageDataset,
    WindowedImageAutoencoderDataset,
)
from tests.utils import BasicTestCase


def _write_tif(path: Path, width: int, height: int, bands: int = 3, dtype="uint8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    if dtype == "uint8":
        data = rng.integers(0, 255, (bands, height, width), dtype=np.uint8)
    else:
        data = rng.random((bands, height, width)).astype(np.float32)

    transform = from_bounds(0, 0, 1, 1, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)


class TestWindowedDatasets(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = Path(self.make_temp_dir())
        self.img_dir = self.tmp / "images"

        # Create two images:
        # img1: 100x100 -> With patch 50x50 and stride 50, should have 2x2 = 4 patches
        # img2: 150x100 -> With patch 50x50 and stride 50, should have 3x2 = 6 patches
        # Total patches: 10
        _write_tif(self.img_dir / "img1.tif", width=100, height=100)
        _write_tif(self.img_dir / "img2.tif", width=150, height=100)

    def test_windowed_image_dataset_indexing(self):
        ds = WindowedImageDataset(image_dir=self.img_dir, crop_size=[50, 50], stride=50)

        # 100/50 * 100/50 = 4
        # 150/50 * 100/50 = 6
        assert len(ds) == 10

        # Test first patch of first image
        sample = ds[0]
        assert sample["image"].shape == (3, 50, 50)
        assert "path" in sample
        assert "img1.tif" in sample["path"]
        with pytest.raises(IndexError):
            _ = ds[-1]

        # Test first patch of second image (index 4)
        sample = ds[4]
        assert "img2.tif" in sample["path"]
        assert sample["image"].shape == (3, 50, 50)

        # Test last patch
        sample = ds[9]
        assert "img2.tif" in sample["path"]

    def test_windowed_image_autoencoder_dataset(self):
        ds = WindowedImageAutoencoderDataset(
            image_dir=self.img_dir, crop_size=[50, 50], stride=50
        )

        assert len(ds) == 10
        sample = ds[0]
        assert "image" in sample
        assert "target" in sample
        assert "path" in sample

        # Without corruption, image and target should be identical
        assert torch.allclose(sample["image"], sample["target"])
        with pytest.raises(IndexError):
            _ = ds[-1]

    def test_windowed_image_autoencoder_corruption(self):
        # Using a simple corruption (Blur) to verify it only affects "image"
        corruption = [
            {"_target_": "albumentations.Blur", "blur_limit": [11, 11], "p": 1.0}
        ]

        ds = WindowedImageAutoencoderDataset(
            image_dir=self.img_dir,
            crop_size=[50, 50],
            stride=50,
            corruption_augmentation_list=corruption,
        )

        sample = ds[0]
        # Blur should make image different from target
        assert not torch.allclose(sample["image"], sample["target"])
        # Target should remain sharp (relatively)

    def test_windowed_image_dataset_small_images(self):
        # Image smaller than 100x100
        _write_tif(self.img_dir / "too_small.tif", width=50, height=50)

        ds = WindowedImageDataset(
            image_dir=self.img_dir, crop_size=[100, 100], stride=100
        )

        # img1 (100x100) -> 1 patch
        # img2 (150x100) -> 1 patch (if stride is 100, (150-100)//100 + 1 = 1)
        # too_small (50x50) -> 0 patches
        assert len(ds) == 2
        for i in range(len(ds)):
            assert "too_small.tif" not in ds[i]["path"]


def test_windowed_image_autoencoder_verify_windows_excludes_invalid_windows(
    tmp_path, monkeypatch
):
    img_dir = tmp_path / "images"
    _write_tif(img_dir / "img.tif", width=100, height=100)

    original = WindowedImageDataset._is_readable_window

    def fake_is_readable(self, info, row_off, col_off):
        if row_off == 0 and col_off == 50:
            return False
        return original(self, info, row_off, col_off)

    monkeypatch.setattr(
        WindowedImageDataset,
        "_is_readable_window",
        fake_is_readable,
    )

    ds = WindowedImageAutoencoderDataset(
        image_dir=img_dir,
        crop_size=[50, 50],
        stride=50,
        verify_windows=True,
    )

    assert len(ds) == 3
    assert all(
        not (entry["row_off"] == 0 and entry["col_off"] == 50)
        for entry in ds._window_index
    )
    sample = ds[0]
    assert set(sample) == {"image", "target", "path"}
    assert sample["image"].shape == (3, 50, 50)
    assert sample["target"].dtype == torch.float32


def test_windowed_image_autoencoder_window_index_cache_is_reused(tmp_path, monkeypatch):
    img_dir = tmp_path / "images"
    cache_path = tmp_path / "cache" / "windows.json"
    _write_tif(img_dir / "img.tif", width=100, height=100)

    ds = WindowedImageAutoencoderDataset(
        image_dir=img_dir,
        crop_size=[50, 50],
        stride=50,
        verify_windows=True,
        window_index_cache=cache_path,
    )

    assert len(ds) == 4
    assert cache_path.exists()
    with cache_path.open() as f:
        cache = json.load(f)
    assert cache["config"]["crop_size"] == [50, 50]
    assert len(cache["windows"]) == 4

    def fail_if_called(self, info, row_off, col_off):
        raise AssertionError("window verification should have been loaded from cache")

    monkeypatch.setattr(
        WindowedImageDataset,
        "_is_readable_window",
        fail_if_called,
    )
    cached = WindowedImageAutoencoderDataset(
        image_dir=img_dir,
        crop_size=[50, 50],
        stride=50,
        verify_windows=True,
        window_index_cache=cache_path,
    )

    assert len(cached) == 4
    assert cached[3]["image"].shape == (3, 50, 50)


def test_windowed_image_autoencoder_window_index_cache_rebuilds_when_stale(
    tmp_path, monkeypatch
):
    img_dir = tmp_path / "images"
    cache_path = tmp_path / "windows.json"
    _write_tif(img_dir / "img.tif", width=100, height=100)

    WindowedImageAutoencoderDataset(
        image_dir=img_dir,
        crop_size=[50, 50],
        stride=50,
        verify_windows=True,
        window_index_cache=cache_path,
    )

    calls = {"count": 0}
    original = WindowedImageDataset._is_readable_window

    def count_calls(self, info, row_off, col_off):
        calls["count"] += 1
        return original(self, info, row_off, col_off)

    monkeypatch.setattr(WindowedImageDataset, "_is_readable_window", count_calls)
    rebuilt = WindowedImageAutoencoderDataset(
        image_dir=img_dir,
        crop_size=[25, 25],
        stride=25,
        verify_windows=True,
        window_index_cache=cache_path,
    )

    assert len(rebuilt) == 16
    assert calls["count"] == 16


def test_windowed_image_dataset_verify_windows_with_selected_bands(tmp_path):
    img_dir = tmp_path / "images"
    _write_tif(img_dir / "img.tif", width=64, height=64, bands=3)

    ds = WindowedImageDataset(
        image_dir=img_dir,
        crop_size=[32, 32],
        stride=[32, 32],
        selected_bands=[1, 3],
        verify_windows=True,
    )

    assert len(ds) == 4
    sample = ds[0]
    assert sample["image"].shape == (2, 32, 32)
    assert ds.get_path(1).endswith("img.tif")


def test_windowed_image_dataset_invalid_window_options_raise(tmp_path):
    img_dir = tmp_path / "images"
    _write_tif(img_dir / "img.tif", width=64, height=64)

    with pytest.raises(ValueError, match="crop_size"):
        WindowedImageDataset(image_dir=img_dir, crop_size=[32])

    with pytest.raises(ValueError, match="stride"):
        WindowedImageDataset(image_dir=img_dir, crop_size=[32, 32], stride=[16])


def test_windowed_image_dataset_default_crop_and_all_small_images(tmp_path):
    img_dir = tmp_path / "images"
    _write_tif(img_dir / "small.tif", width=32, height=32)

    ds = WindowedImageDataset(image_dir=img_dir)

    assert ds.crop_size == [256, 256]
    assert ds.stride == [256, 256]
    assert len(ds) == 0


def test_windowed_image_dataset_transform_and_read_error_fallback(
    tmp_path, monkeypatch
):
    img_dir = tmp_path / "images"
    _write_tif(img_dir / "img.tif", width=64, height=32)
    ds = WindowedImageDataset(
        image_dir=img_dir,
        crop_size=[32, 32],
        stride=32,
        augmentation_list=[{"_target_": "albumentations.pytorch.ToTensorV2"}],
    )
    original = WindowedImageDataset._read_window
    calls = {"count": 0}

    def fail_once(self, info, row_off, col_off):
        calls["count"] += 1
        if calls["count"] == 1:
            raise rasterio.errors.RasterioIOError("broken window")
        return original(self, info, row_off, col_off)

    monkeypatch.setattr(WindowedImageDataset, "_read_window", fail_once)

    sample = ds[0]

    assert calls["count"] == 2
    assert sample["image"].shape == (3, 32, 32)


def test_windowed_image_dataset_malformed_cache_rebuilds(tmp_path):
    img_dir = tmp_path / "images"
    cache_path = tmp_path / "bad.json"
    _write_tif(img_dir / "img.tif", width=64, height=64)
    cache_path.write_text("{not-json")

    ds = WindowedImageDataset(
        image_dir=img_dir,
        crop_size=[32, 32],
        stride=32,
        verify_windows=True,
        window_index_cache=cache_path,
    )

    assert len(ds) == 4


def test_windowed_image_dataset_cache_with_unknown_image_rebuilds(
    tmp_path, monkeypatch
):
    img_dir = tmp_path / "images"
    cache_path = tmp_path / "windows.json"
    _write_tif(img_dir / "img.tif", width=64, height=64)

    ds = WindowedImageDataset(
        image_dir=img_dir,
        crop_size=[32, 32],
        stride=32,
        verify_windows=True,
        window_index_cache=cache_path,
    )
    ds._window_index[0]["path"] = str(img_dir / "missing.tif")
    ds._save_window_index_cache()

    calls = {"count": 0}
    original = WindowedImageDataset._is_readable_window

    def count_calls(self, info, row_off, col_off):
        calls["count"] += 1
        return original(self, info, row_off, col_off)

    monkeypatch.setattr(WindowedImageDataset, "_is_readable_window", count_calls)
    rebuilt = WindowedImageDataset(
        image_dir=img_dir,
        crop_size=[32, 32],
        stride=32,
        verify_windows=True,
        window_index_cache=cache_path,
    )

    assert len(rebuilt) == 4
    assert calls["count"] == 4


def test_windowed_image_dataset_verify_windows_can_produce_empty_index(
    tmp_path, monkeypatch
):
    img_dir = tmp_path / "images"
    cache_path = tmp_path / "empty.json"
    _write_tif(img_dir / "img.tif", width=64, height=64)

    monkeypatch.setattr(
        WindowedImageDataset,
        "_is_readable_window",
        lambda self, info, row_off, col_off: False,
    )

    ds = WindowedImageDataset(
        image_dir=img_dir,
        crop_size=[32, 32],
        stride=32,
        verify_windows=True,
        window_index_cache=cache_path,
    )

    assert len(ds) == 0
    with cache_path.open() as f:
        cache = json.load(f)
    assert cache["windows"] == []


def test_windowed_image_autoencoder_read_error_fallback(tmp_path, monkeypatch):
    img_dir = tmp_path / "images"
    _write_tif(img_dir / "img.tif", width=64, height=32)
    ds = WindowedImageAutoencoderDataset(
        image_dir=img_dir,
        crop_size=[32, 32],
        stride=32,
    )
    original = WindowedImageDataset._read_window
    calls = {"count": 0}

    def fail_once(self, info, row_off, col_off):
        calls["count"] += 1
        if calls["count"] == 1:
            raise rasterio.errors.RasterioIOError("broken window")
        return original(self, info, row_off, col_off)

    monkeypatch.setattr(WindowedImageDataset, "_read_window", fail_once)

    sample = ds[0]

    assert calls["count"] == 2
    assert sample["target"].shape == (3, 32, 32)


def test_windowed_image_dataset_serializes_rasterio_reads(tmp_path, monkeypatch):
    img_dir = tmp_path / "images"
    lock_dir = tmp_path / "locks"
    _write_tif(img_dir / "img.tif", width=64, height=64)
    calls = []

    @contextmanager
    def fake_read_lock(path, enabled=False, lock_dir=None):
        calls.append((Path(path).name, enabled, Path(lock_dir)))
        yield

    monkeypatch.setattr(
        image_dataset_module,
        "_rasterio_read_lock",
        fake_read_lock,
    )
    ds = WindowedImageDataset(
        image_dir=img_dir,
        crop_size=[32, 32],
        stride=32,
        serialize_rasterio_reads=True,
        rasterio_lock_dir=lock_dir,
        reopen_rasterio_on_read=True,
    )

    sample = ds[0]

    assert sample["image"].shape == (3, 32, 32)
    assert calls == [("img.tif", True, lock_dir)]
    assert len(ds._file_cache) == 0


def test_iterable_windowed_image_dataset_shards_images_by_worker(tmp_path):
    img_dir = tmp_path / "images"
    _write_tif(img_dir / "img1.tif", width=64, height=64)
    _write_tif(img_dir / "img2.tif", width=64, height=64)

    ds = IterableWindowedImageDataset(
        image_dir=img_dir,
        crop_size=[32, 32],
        stride=32,
    )
    loader = DataLoader(ds, batch_size=1, num_workers=2)

    paths = []
    for batch in loader:
        paths.extend(batch["path"])

    assert len(paths) == 8
    assert paths.count(str(img_dir / "img1.tif")) == 4
    assert paths.count(str(img_dir / "img2.tif")) == 4


def test_iterable_windowed_autoencoder_dataset_returns_pairs(tmp_path):
    img_dir = tmp_path / "images"
    _write_tif(img_dir / "img.tif", width=64, height=64)

    ds = IterableWindowedImageAutoencoderDataset(
        image_dir=img_dir,
        crop_size=[32, 32],
        stride=32,
    )

    sample = next(iter(ds))

    assert set(sample) == {"image", "target", "path"}
    assert sample["image"].shape == (3, 32, 32)
    assert torch.allclose(sample["image"], sample["target"])
