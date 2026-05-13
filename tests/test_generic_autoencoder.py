# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from pytorch_segmentation_models_trainer.custom_models.generic_autoencoder import (
    GenericAutoencoder,
)
from pytorch_segmentation_models_trainer.model_loader.autoencoder_model import (
    AutoencoderModel,
)


def test_generic_autoencoder_smp_forward():
    # Test with SMP encoder (resnet18 is standard and fast)
    model = GenericAutoencoder(
        encoder_name="resnet18", use_huggingface=False, in_channels=3, pretrained=False
    )
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    assert output.shape == (1, 3, 256, 256)
    assert isinstance(output, torch.Tensor)


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
