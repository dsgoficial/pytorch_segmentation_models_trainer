# -*- coding: utf-8 -*-
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
import torch.nn as nn
from rasterio.transform import from_origin
from torch.utils.data import DataLoader

from pytorch_segmentation_models_trainer.custom_models.generic_autoencoder import (
    GenericAutoencoder,
)
from pytorch_segmentation_models_trainer.dataset_loader.image_dataset import (
    AutoencoderRandomCropDataset,
)
from pytorch_segmentation_models_trainer.model_loader.autoencoder_model import (
    AutoencoderModel,
)


def _write_raster(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data[None, ...]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(data)


def _make_image_folder(tmp_path, dtype=np.uint8):
    image_dir = tmp_path / "images"
    values = [
        np.full((3, 32, 32), 64, dtype=dtype),
        np.full((3, 40, 36), 128, dtype=dtype),
        np.full((3, 48, 44), 255 if dtype == np.uint8 else 1024, dtype=dtype),
    ]
    _write_raster(image_dir / "a" / "img_0.tif", values[0])
    _write_raster(image_dir / "a" / "img_1.tiff", values[1])
    _write_raster(image_dir / "b" / "img_2.TIF", values[2])
    (image_dir / "a" / "img_0.tif.aux.xml").write_text("<xml />")
    return image_dir


def test_discovers_images_recursively_and_normalizes_extensions(tmp_path):
    image_dir = _make_image_folder(tmp_path)

    ds = AutoencoderRandomCropDataset(
        image_dir=image_dir,
        image_extensions=["tif", ".tiff"],
        crop_size=[16, 16],
        samples_per_epoch=7,
    )

    assert len(ds.df) == 3
    assert len(ds) == 7
    assert ds.image_extensions == [".tif", ".tiff"]


def test_empty_image_dir_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="No images found"):
        AutoencoderRandomCropDataset(
            image_dir=tmp_path / "empty",
            crop_size=[16, 16],
            samples_per_epoch=1,
        )


def test_invalid_constructor_arguments_raise(tmp_path):
    image_dir = _make_image_folder(tmp_path)

    with pytest.raises(ValueError, match="crop_size"):
        AutoencoderRandomCropDataset(image_dir=image_dir, crop_size=[16])
    with pytest.raises(ValueError, match="split"):
        AutoencoderRandomCropDataset(
            image_dir=image_dir, split="test", crop_size=[16, 16]
        )
    with pytest.raises(ValueError, match="val_fraction"):
        AutoencoderRandomCropDataset(
            image_dir=image_dir, val_fraction=1.0, crop_size=[16, 16]
        )
    with pytest.raises(ValueError, match="image_dtype"):
        AutoencoderRandomCropDataset(
            image_dir=image_dir, image_dtype="int32", crop_size=[16, 16]
        )
    with pytest.raises(ValueError, match="selected_bands"):
        AutoencoderRandomCropDataset(
            image_dir=image_dir, selected_bands=[0], crop_size=[16, 16]
        )
    with pytest.raises(ValueError, match="input_csv_path, df, or image_dir"):
        AutoencoderRandomCropDataset(crop_size=[16, 16])


def test_default_crop_size_is_used(tmp_path):
    image_dir = tmp_path / "images"
    _write_raster(image_dir / "large.tif", np.ones((3, 256, 256), dtype=np.uint8))

    ds = AutoencoderRandomCropDataset(image_dir=image_dir, samples_per_epoch=1)

    assert ds.crop_size == [256, 256]


def test_rejects_images_smaller_than_crop_size(tmp_path):
    image_dir = tmp_path / "images"
    _write_raster(image_dir / "small.tif", np.ones((3, 8, 8), dtype=np.uint8))

    with pytest.raises(ValueError, match="No image large enough"):
        AutoencoderRandomCropDataset(
            image_dir=image_dir,
            crop_size=[16, 16],
            samples_per_epoch=1,
        )


def test_returns_clean_target_and_cropped_tensor(tmp_path):
    image_dir = _make_image_folder(tmp_path)
    ds = AutoencoderRandomCropDataset(
        image_dir=image_dir,
        crop_size=[16, 16],
        samples_per_epoch=2,
        augmentation_list=[{"_target_": "albumentations.pytorch.ToTensorV2"}],
    )

    item = ds[0]

    assert set(item) == {"image", "target", "path"}
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (3, 16, 16)
    assert item["target"].shape == (3, 16, 16)
    assert item["image"].dtype == torch.uint8
    assert torch.equal(item["image"], item["target"])


def test_without_transform_returns_normalized_float_tensors(tmp_path):
    image_dir = _make_image_folder(tmp_path)
    ds = AutoencoderRandomCropDataset(
        image_dir=image_dir,
        crop_size=[16, 16],
        samples_per_epoch=1,
    )

    item = ds[0]

    assert item["image"].dtype == torch.float32
    assert item["target"].dtype == torch.float32
    assert item["image"].shape == (3, 16, 16)
    assert torch.max(item["image"]) <= 1.0


def test_auto_samples_per_epoch_and_state_roundtrip(tmp_path):
    image_dir = _make_image_folder(tmp_path)
    ds = AutoencoderRandomCropDataset(
        image_dir=image_dir,
        crop_size=[16, 16],
        samples_per_epoch=0,
        file_cache_maxsize=2,
    )

    state = ds.__getstate__()
    ds.__setstate__(state)
    ds._close_cache()

    assert len(ds) > 0
    assert ds.file_cache_maxsize == 2


def test_reset_augmentation_function_path(tmp_path, monkeypatch):
    image_dir = _make_image_folder(tmp_path)
    called = {"gc": 0}
    import pytorch_segmentation_models_trainer.dataset_loader.image_dataset as module

    monkeypatch.setattr(module.gc, "collect", lambda: called.__setitem__("gc", 1))
    ds = AutoencoderRandomCropDataset(
        image_dir=image_dir,
        crop_size=[16, 16],
        samples_per_epoch=1,
        reset_augmentation_function=True,
        augmentation_list=[{"_target_": "albumentations.pytorch.ToTensorV2"}],
    )

    item = ds[0]

    assert item["image"].shape == (3, 16, 16)
    assert called["gc"] == 1


def test_native_selected_band_crop(tmp_path):
    image_dir = _make_image_folder(tmp_path)
    ds = AutoencoderRandomCropDataset(
        image_dir=image_dir,
        crop_size=[16, 16],
        samples_per_epoch=1,
        selected_bands=[1, 2],
        image_dtype="native",
    )

    crop = ds._read_crop(ds.get_path(0), 0, 0)

    assert crop.shape == (16, 16, 2)
    assert crop.dtype == np.uint8


def test_corruption_applies_only_to_input(tmp_path):
    image_dir = tmp_path / "images"
    _write_raster(image_dir / "img.tif", np.full((3, 32, 32), 64, dtype=np.uint8))
    ds = AutoencoderRandomCropDataset(
        image_dir=image_dir,
        crop_size=[16, 16],
        samples_per_epoch=1,
        corruption_augmentation_list=[
            {
                "_target_": "albumentations.RandomBrightnessContrast",
                "brightness_limit": [0.5, 0.5],
                "contrast_limit": [0.0, 0.0],
                "p": 1.0,
            }
        ],
        augmentation_list=[{"_target_": "albumentations.pytorch.ToTensorV2"}],
    )

    item = ds[0]

    assert not torch.equal(item["image"], item["target"])


def test_split_is_deterministic_and_has_no_overlap(tmp_path):
    image_dir = _make_image_folder(tmp_path)

    train_ds = AutoencoderRandomCropDataset(
        image_dir=image_dir,
        split="train",
        val_fraction=0.34,
        split_seed=123,
        crop_size=[16, 16],
        samples_per_epoch=3,
    )
    val_ds = AutoencoderRandomCropDataset(
        image_dir=image_dir,
        split="val",
        val_fraction=0.34,
        split_seed=123,
        crop_size=[16, 16],
        samples_per_epoch=3,
    )

    train_paths = set(train_ds.df["image"].tolist())
    val_paths = set(val_ds.df["image"].tolist())
    assert train_paths
    assert val_paths
    assert train_paths.isdisjoint(val_paths)


def test_single_image_train_split_raises_when_empty(tmp_path):
    image_dir = tmp_path / "images"
    _write_raster(image_dir / "img.tif", np.ones((3, 32, 32), dtype=np.uint8))

    with pytest.raises(ValueError, match="Split 'train' produced no images"):
        AutoencoderRandomCropDataset(
            image_dir=image_dir,
            split="train",
            crop_size=[16, 16],
            samples_per_epoch=1,
        )


def test_dataframe_source_is_supported(tmp_path):
    image_dir = _make_image_folder(tmp_path)
    path = next(image_dir.rglob("*.tif"))
    ds = AutoencoderRandomCropDataset(
        df=pd.DataFrame({"image": [str(path)]}),
        crop_size=[16, 16],
        samples_per_epoch=1,
    )

    assert len(ds.df) == 1


def test_dataloader_batches_different_batch_sizes(tmp_path):
    image_dir = _make_image_folder(tmp_path)
    ds = AutoencoderRandomCropDataset(
        image_dir=image_dir,
        crop_size=[16, 16],
        samples_per_epoch=5,
        augmentation_list=[{"_target_": "albumentations.pytorch.ToTensorV2"}],
    )

    batch = next(iter(DataLoader(ds, batch_size=3)))

    assert batch["image"].shape == (3, 3, 16, 16)
    assert batch["target"].shape == (3, 3, 16, 16)


def test_autoencoder_model_training_step_with_random_crop_dataset(tmp_path):
    image_dir = _make_image_folder(tmp_path)
    ds = AutoencoderRandomCropDataset(
        image_dir=image_dir,
        crop_size=[16, 16],
        samples_per_epoch=2,
        augmentation_list=[{"_target_": "albumentations.pytorch.ToTensorV2"}],
    )
    batch = next(iter(DataLoader(ds, batch_size=2)))
    batch = {
        "image": batch["image"].float() / 255.0,
        "target": batch["target"].float() / 255.0,
    }

    cfg = MagicMock()
    cfg.model = {"_target_": "...", "encoder_name": "resnet18"}
    cfg.loss = {"_target_": "torch.nn.MSELoss"}
    cfg.optimizer = {"_target_": "torch.optim.Adam", "lr": 1e-3}
    cfg.scheduler_list = []
    model_module = GenericAutoencoder(encoder_name="resnet18", pretrained=False)

    with patch.object(AutoencoderModel, "get_model", return_value=model_module):
        with patch.object(
            AutoencoderModel, "get_loss_function", return_value=nn.MSELoss()
        ):
            pl_model = AutoencoderModel(cfg)
            pl_model.log = MagicMock()
            loss = pl_model.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
