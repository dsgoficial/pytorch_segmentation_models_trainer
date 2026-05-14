# -*- coding: utf-8 -*-
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from pytorch_segmentation_models_trainer.custom_models.generic_autoencoder import (
    GenericAutoencoder,
    GenericDecoder,
    ProgressiveDecoder,
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


# ── ProgressiveDecoder standalone tests ───────────────────────────────────────


@pytest.mark.parametrize("scale_factor", [2, 4, 8, 16, 32])
def test_progressive_decoder_output_shape(scale_factor):
    decoder = ProgressiveDecoder(
        in_channels=64, out_channels=3, scale_factor=scale_factor
    )
    x = torch.randn(2, 64, 4, 4)
    out = decoder(x)
    assert out.shape == (2, 3, 4 * scale_factor, 4 * scale_factor)


def test_progressive_decoder_output_activation_default_is_none():
    decoder = ProgressiveDecoder(in_channels=32, out_channels=3, scale_factor=4)
    assert decoder.output_activation is None


def test_progressive_decoder_invalid_activation_raises():
    with pytest.raises(ValueError, match="output_activation"):
        ProgressiveDecoder(in_channels=32, out_channels=3, output_activation="relu")


def test_progressive_decoder_non_power_of_2_raises():
    with pytest.raises(ValueError, match="power of 2"):
        ProgressiveDecoder(in_channels=32, out_channels=3, scale_factor=3)


@pytest.mark.parametrize("activation", ["sigmoid", "tanh"])
def test_progressive_decoder_output_activation_bounds(activation):
    decoder = ProgressiveDecoder(
        in_channels=32,
        out_channels=3,
        scale_factor=4,
        output_activation=activation,
    )
    x = torch.randn(2, 32, 4, 4) * 10
    out = decoder(x)
    assert out.shape == (2, 3, 16, 16)
    if activation == "sigmoid":
        assert out.min() >= 0.0
        assert out.max() <= 1.0
    else:
        assert out.min() >= -1.0
        assert out.max() <= 1.0


def test_progressive_decoder_no_activation_is_unbounded():
    decoder = ProgressiveDecoder(in_channels=32, out_channels=3, scale_factor=4)
    x = torch.randn(1, 32, 4, 4) * 100
    out = decoder(x)
    assert out.shape == (1, 3, 16, 16)


def test_progressive_decoder_gradient_flow():
    decoder = ProgressiveDecoder(in_channels=16, out_channels=3, scale_factor=4)
    x = torch.randn(1, 16, 8, 8, requires_grad=True)
    out = decoder(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.all(x.grad == 0)


def test_progressive_decoder_dtype_preserved():
    decoder = ProgressiveDecoder(in_channels=16, out_channels=3, scale_factor=4)
    x = torch.randn(1, 16, 4, 4)
    out = decoder(x)
    assert out.dtype == x.dtype


# ── GenericAutoencoder + ProgressiveDecoder integration ───────────────────────


def test_generic_autoencoder_with_progressive_decoder_forward():
    with patch(
        "pytorch_segmentation_models_trainer.custom_models.generic_autoencoder.smp"
    ) as mock_smp:
        mock_encoder = MagicMock()
        mock_encoder.out_channels = [3, 64, 64, 128, 256, 512]
        mock_encoder.return_value = [
            torch.zeros(1, 3, 64, 64),
            torch.zeros(1, 64, 32, 32),
            torch.zeros(1, 64, 16, 16),
            torch.zeros(1, 128, 8, 8),
            torch.zeros(1, 256, 4, 4),
            torch.zeros(1, 512, 2, 2),
        ]
        mock_smp.encoders.get_encoder.return_value = mock_encoder

        model = GenericAutoencoder(
            encoder_name="resnet18",
            use_huggingface=False,
            in_channels=3,
            pretrained=False,
            use_progressive_decoder=True,
        )
        assert isinstance(model.decoder, ProgressiveDecoder)

        x = torch.randn(1, 3, 64, 64)
        out = model(x)
        assert out.shape == (1, 3, 64, 64)


def test_generic_autoencoder_progressive_decoder_with_smp():
    model = GenericAutoencoder(
        encoder_name="resnet18",
        use_huggingface=False,
        in_channels=3,
        pretrained=False,
        use_progressive_decoder=True,
    )
    assert isinstance(model.decoder, ProgressiveDecoder)
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 256, 256)


def test_generic_autoencoder_progressive_decoder_gradient_flow():
    model = GenericAutoencoder(
        encoder_name="resnet18",
        use_huggingface=False,
        in_channels=3,
        pretrained=False,
        use_progressive_decoder=True,
    )
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    out.sum().backward()
    for name, param in model.decoder.named_parameters():
        assert param.grad is not None, f"No gradient for decoder.{name}"


@pytest.mark.parametrize("activation", [None, "sigmoid", "tanh"])
def test_generic_autoencoder_progressive_decoder_activations(activation):
    model = GenericAutoencoder(
        encoder_name="resnet18",
        pretrained=False,
        use_progressive_decoder=True,
        output_activation=activation,
    )
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 64, 64)
    if activation == "sigmoid":
        assert out.min() >= 0.0 and out.max() <= 1.0
    elif activation == "tanh":
        assert out.min() >= -1.0 and out.max() <= 1.0


# ── GenericVariationalAutoencoder + ProgressiveDecoder integration ─────────────


def test_variational_autoencoder_with_progressive_decoder():
    from pytorch_segmentation_models_trainer.custom_models.variational_autoencoder import (
        GenericVariationalAutoencoder,
        VariationalAutoencoderOutput,
    )

    with patch(
        "pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.smp"
    ) as mock_smp:
        mock_encoder = MagicMock()
        mock_encoder.out_channels = [3, 64, 64, 128]
        mock_encoder.return_value = [
            torch.zeros(1, 3, 64, 64),
            torch.zeros(1, 64, 32, 32),
            torch.zeros(1, 64, 16, 16),
            torch.zeros(1, 128, 8, 8),
        ]
        mock_smp.encoders.get_encoder.return_value = mock_encoder

        model = GenericVariationalAutoencoder(
            encoder_name="resnet18",
            use_huggingface=False,
            in_channels=3,
            latent_dim=16,
            pretrained=False,
            encoder_depth=3,
            use_progressive_decoder=True,
        )
        assert isinstance(model.decoder, ProgressiveDecoder)

        x = torch.randn(1, 3, 64, 64)
        output = model(x)
        assert isinstance(output, VariationalAutoencoderOutput)
        assert output.reconstruction.shape == (1, 3, 64, 64)
        assert output.mu.shape[0] == 1
        assert output.z.shape == output.mu.shape


def test_variational_autoencoder_progressive_decoder_with_real_smp():
    from pytorch_segmentation_models_trainer.custom_models.variational_autoencoder import (
        GenericVariationalAutoencoder,
        VariationalAutoencoderOutput,
    )

    model = GenericVariationalAutoencoder(
        encoder_name="resnet18",
        use_huggingface=False,
        in_channels=3,
        latent_dim=8,
        pretrained=False,
        encoder_depth=3,
        use_progressive_decoder=True,
    )
    assert isinstance(model.decoder, ProgressiveDecoder)

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        output = model(x)
    assert isinstance(output, VariationalAutoencoderOutput)
    assert output.reconstruction.shape == (1, 3, 64, 64)


def test_variational_autoencoder_progressive_decoder_gradient_flow():
    from pytorch_segmentation_models_trainer.custom_models.variational_autoencoder import (
        GenericVariationalAutoencoder,
    )

    model = GenericVariationalAutoencoder(
        encoder_name="resnet18",
        pretrained=False,
        encoder_depth=3,
        latent_dim=8,
        use_progressive_decoder=True,
    )
    x = torch.randn(1, 3, 64, 64)
    output = model(x)
    output.reconstruction.sum().backward()
    for name, param in model.decoder.named_parameters():
        assert param.grad is not None, f"No gradient for decoder.{name}"
