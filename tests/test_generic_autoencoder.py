# -*- coding: utf-8 -*-
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from pytorch_segmentation_models_trainer.custom_models.generic_autoencoder import (
    GenericAutoencoder,
    GenericDecoder,
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


def test_autoencoder_model_computes_metrics_when_configured():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {"_target_": "unused"},
            "loss": {"_target_": "unused"},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler_list": [],
            "metrics": [
                {"_target_": "torchmetrics.MeanSquaredError"},
            ],
        }
    )
    model_module = nn.Conv2d(3, 3, kernel_size=1)

    with patch.object(AutoencoderModel, "get_model", return_value=model_module):
        with patch.object(
            AutoencoderModel, "get_loss_function", return_value=nn.MSELoss()
        ):
            pl_model = AutoencoderModel(cfg)

    pl_model.log = MagicMock()
    pl_model.log_dict = MagicMock()
    batch = {"image": torch.randn(2, 3, 8, 8), "target": torch.randn(2, 3, 8, 8)}

    pl_model.training_step(batch, 0)
    pl_model.validation_step(batch, 0)

    assert pl_model.log_dict.call_count == 2
    first_keys = set(pl_model.log_dict.call_args_list[0].args[0].keys())
    assert any("train/" in k for k in first_keys)
    second_keys = set(pl_model.log_dict.call_args_list[1].args[0].keys())
    assert any("val/" in k for k in second_keys)


def test_generic_decoder_output_activation_default_is_none():
    decoder = GenericDecoder(in_channels=8, out_channels=3, scale_factor=0)
    assert decoder.output_activation is None


def test_generic_decoder_output_activation_invalid_raises():
    with pytest.raises(ValueError, match="output_activation"):
        GenericDecoder(in_channels=8, out_channels=3, output_activation="relu")


@pytest.mark.parametrize("activation", ["sigmoid", "tanh"])
def test_generic_decoder_output_activation_bounds(activation):
    decoder = GenericDecoder(
        in_channels=8, out_channels=3, scale_factor=0, output_activation=activation
    )
    x = torch.randn(2, 8, 16, 16) * 10
    out = decoder(x)
    assert out.shape == (2, 3, 16, 16)
    if activation == "sigmoid":
        assert out.min() >= 0.0
        assert out.max() <= 1.0
    else:
        assert out.min() >= -1.0
        assert out.max() <= 1.0


def test_generic_decoder_no_activation_is_unbounded():
    decoder = GenericDecoder(in_channels=8, out_channels=3, scale_factor=0)
    x = torch.randn(1, 8, 4, 4) * 100
    out = decoder(x)
    assert out.shape == (1, 3, 4, 4)


def test_generic_autoencoder_output_activation_sigmoid():
    model = GenericAutoencoder(
        encoder_name="resnet18",
        use_huggingface=False,
        in_channels=3,
        pretrained=False,
        output_activation="sigmoid",
    )
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == (1, 3, 64, 64)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
