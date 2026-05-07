# -*- coding: utf-8 -*-
import os
import pytest
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from unittest.mock import MagicMock
from pytorch_segmentation_models_trainer.custom_models.generic_autoencoder import (
    GenericAutoencoder,
)
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    AutoencoderDataset,
)
from pytorch_segmentation_models_trainer.model_loader.autoencoder_model import (
    AutoencoderModel,
)


@pytest.fixture
def tiny_image_csv(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "test.jpg"
    Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(
        img_path
    )

    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"image": ["test.jpg"]}).to_csv(csv_path, index=False)
    return csv_path, str(img_dir)


def test_generic_autoencoder_smp_forward():
    # Test with SMP encoder (resnet18 is standard and fast)
    model = GenericAutoencoder(
        encoder_name="resnet18", use_huggingface=False, in_channels=3, pretrained=False
    )
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    assert output.shape == (1, 3, 256, 256)
    assert isinstance(output, torch.Tensor)


def test_autoencoder_dataset(tiny_image_csv):
    csv_path, img_dir = tiny_image_csv
    dataset = AutoencoderDataset(input_csv_path=csv_path, root_dir=img_dir)
    item = dataset[0]
    assert "image" in item
    assert "target" in item
    assert "path" in item
    # Since no transform, it returns numpy array from PIL
    assert isinstance(item["image"], np.ndarray)
    assert np.array_equal(item["image"], item["target"])


from unittest.mock import MagicMock, patch


def test_autoencoder_dataset_with_corruption(tiny_image_csv):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    csv_path, img_dir = tiny_image_csv

    # Define a corruption that changes values (GaussNoise)
    corruption_list = [
        {"_target_": "albumentations.GaussNoise", "var_limit": [100.0, 100.0], "p": 1.0}
    ]

    # Define base transforms (Resize)
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
    img = item["image"]
    target = item["target"]

    # Check shape (both should be resized to 64x64)
    assert img.shape == (3, 64, 64)
    assert target.shape == (3, 64, 64)

    # Check values (they should be different because of GaussNoise on image only)
    assert not torch.allclose(img, target)


def test_autoencoder_model_step():
    cfg = MagicMock()
    # Mock necessary config attributes
    cfg.model = {"_target_": "...", "encoder_name": "resnet18"}
    cfg.loss = {"_target_": "torch.nn.MSELoss"}
    cfg.optimizer = {"_target_": "torch.optim.Adam", "lr": 1e-3}
    cfg.scheduler_list = []

    # Patch the model creation and loss creation in AutoencoderModel
    model_module = GenericAutoencoder(encoder_name="resnet18", pretrained=False)

    with patch.object(AutoencoderModel, "get_model", return_value=model_module):
        with patch.object(
            AutoencoderModel, "get_loss_function", return_value=nn.MSELoss()
        ):
            pl_model = AutoencoderModel(cfg)
            # Mock log to avoid PL errors during testing
            pl_model.log = MagicMock()

            batch = {
                "image": torch.randn(2, 3, 64, 64),
                "target": torch.randn(2, 3, 64, 64),
            }
            loss = pl_model.training_step(batch, 0)
            assert isinstance(loss, torch.Tensor)
            assert loss >= 0
            assert not torch.isnan(loss)
