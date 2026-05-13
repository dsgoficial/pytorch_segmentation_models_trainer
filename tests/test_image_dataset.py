# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from pytorch_segmentation_models_trainer.dataset_loader.image_dataset import (
    AutoencoderDataset,
    CSVWindowedImageDataset,
    ImageDataset,
    TiledInferenceImageDataset,
)
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    AutoencoderDataset as CompatAutoencoderDataset,
    CSVWindowedImageDataset as CompatCSVWindowedImageDataset,
    ImageDataset as CompatImageDataset,
    TiledInferenceImageDataset as CompatTiledInferenceImageDataset,
)


def _tiny_image_csv(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "test.jpg"
    Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(
        img_path
    )

    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"image": ["test.jpg"]}).to_csv(csv_path, index=False)
    return csv_path, str(img_dir)


def _write_png(path, value=64, shape=(32, 32, 3)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(shape, value, dtype=np.uint8)).save(path)


def test_autoencoder_dataset_contract_without_transform(tmp_path):
    csv_path, img_dir = _tiny_image_csv(tmp_path)
    dataset = AutoencoderDataset(input_csv_path=csv_path, root_dir=img_dir)

    item = dataset[0]

    assert set(item) == {"image", "target", "path"}
    assert isinstance(item["image"], np.ndarray)
    assert item["image"].shape == (32, 32, 3)
    assert np.array_equal(item["image"], item["target"])


def test_autoencoder_dataset_with_corruption(tmp_path):
    csv_path, img_dir = _tiny_image_csv(tmp_path)
    corruption_list = [
        {
            "_target_": "albumentations.RandomBrightnessContrast",
            "brightness_limit": [0.5, 0.5],
            "contrast_limit": [0.0, 0.0],
            "p": 1.0,
        }
    ]
    aug_list = [
        {"_target_": "albumentations.Resize", "height": 64, "width": 64},
        {"_target_": "albumentations.pytorch.ToTensorV2"},
    ]

    dataset = AutoencoderDataset(
        input_csv_path=csv_path,
        root_dir=img_dir,
        augmentation_list=aug_list,
        corruption_augmentation_list=corruption_list,
    )

    item = dataset[0]

    assert item["image"].shape == (3, 64, 64)
    assert item["target"].shape == (3, 64, 64)
    assert not torch.allclose(item["image"], item["target"])


def test_autoencoder_dataset_legacy_import_reexports_new_class():
    assert CompatAutoencoderDataset is AutoencoderDataset


def test_image_dataset_legacy_imports_reexport_new_classes():
    assert CompatImageDataset is ImageDataset
    assert CompatCSVWindowedImageDataset is CSVWindowedImageDataset
    assert CompatTiledInferenceImageDataset is TiledInferenceImageDataset


def test_image_dataset_get_grouped_datasets(tmp_path):
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    _write_png(img_a, value=10)
    _write_png(img_b, value=20)
    df = pd.DataFrame(
        {
            "image": [str(img_a), str(img_b)],
            "fold": ["train", "val"],
        }
    )

    grouped = ImageDataset.get_grouped_datasets(df, group_by_keys=["fold"])

    assert set(grouped) == {"train", "val"}
    assert grouped["train"][0]["image"].shape == (32, 32, 3)


def test_image_dataset_with_transform(tmp_path):
    img = tmp_path / "img.png"
    _write_png(img)
    ds = ImageDataset(
        df=pd.DataFrame({"image": [str(img)]}),
        augmentation_list=[{"_target_": "albumentations.pytorch.ToTensorV2"}],
    )

    item = ds[0]

    assert item["image"].shape == (3, 32, 32)
    assert item["path"] == str(img)


def test_tiled_inference_requires_shape_when_padding(tmp_path):
    img = tmp_path / "img.png"
    _write_png(img)

    with pytest.raises(ValueError, match="model_input_shape"):
        TiledInferenceImageDataset(
            df=pd.DataFrame({"image": [str(img)], "width": [32], "height": [32]}),
            pad_if_needed=True,
        )


def test_tiled_inference_dataset_getitem_and_collate(tmp_path):
    img_a = tmp_path / "a.png"
    img_b = tmp_path / "b.png"
    _write_png(img_a, value=10, shape=(24, 24, 3))
    _write_png(img_b, value=20, shape=(24, 24, 3))
    ds = TiledInferenceImageDataset(
        df=pd.DataFrame(
            {
                "image": [str(img_a), str(img_b)],
                "width": [24, 24],
                "height": [24, 24],
            }
        ),
        normalize_output=False,
        pad_if_needed=True,
        model_input_shape=[16, 16],
        step_shape=[16, 16],
    )

    item0 = ds[0]
    item1 = ds[1]
    batch = TiledInferenceImageDataset.collate_fn([item0, item1])

    assert item0["tiles"].shape[1:] == (3, 16, 16)
    assert item0["tile_image_idx"].dtype == torch.int64
    assert item0["original_shape"] == (24, 24)
    assert batch["tiles"].shape[0] == item0["tiles"].shape[0] + item1["tiles"].shape[0]
    assert batch["path"] == [str(img_a), str(img_b)]
