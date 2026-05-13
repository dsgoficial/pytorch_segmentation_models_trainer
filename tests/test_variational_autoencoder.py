# -*- coding: utf-8 -*-
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.custom_losses.autoencoder_losses import (
    VariationalAutoencoderLoss,
)
from pytorch_segmentation_models_trainer.custom_models.variational_autoencoder import (
    GenericVariationalAutoencoder,
    VariationalAutoencoderOutput,
)
from pytorch_segmentation_models_trainer.model_loader.variational_autoencoder_model import (
    VariationalAutoencoderModel,
)


class TinyEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=8):
        super().__init__()
        self.out_channels = [in_channels, out_channels]
        self.net = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return [x, self.net(x)]


class TinyHFEncoder(nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.out_channels = 8
        self.net = nn.Conv2d(3, 8, kernel_size=3, stride=4, padding=1)

    def forward(self, x):
        # HuggingFace encoders usually return a list/tuple of hidden states
        return [self.net(x)]


class TinyVAEOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x):
        reconstruction = self.proj(x)
        latent_shape = (x.shape[0], 2, max(1, x.shape[2] // 4), max(1, x.shape[3] // 4))
        mu = reconstruction.mean() * torch.ones(latent_shape, device=x.device)
        logvar = torch.zeros_like(mu)
        return VariationalAutoencoderOutput(reconstruction, mu, logvar, mu)


def test_generic_variational_autoencoder_smp_contract_and_gradients():
    with patch(
        "pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.smp"
    ) as smp_mock:
        smp_mock.encoders.get_encoder.return_value = TinyEncoder()
        model = GenericVariationalAutoencoder(
            encoder_name="tiny", in_channels=3, latent_dim=4, pretrained=False
        )

    x = torch.randn(2, 3, 32, 32)
    output = model(x)

    assert isinstance(output, VariationalAutoencoderOutput)
    assert output.reconstruction.shape == x.shape
    assert output.mu.shape == (2, 4, 16, 16)
    assert output.logvar.shape == output.mu.shape
    assert output.z.shape == output.mu.shape

    loss = output.reconstruction.mean() + output.mu.mean() + output.logvar.mean()
    loss.backward()

    assert model.mu_proj.weight.grad is not None
    assert model.logvar_proj.weight.grad is not None
    assert model.decoder.conv1.weight.grad is not None


def test_generic_variational_autoencoder_huggingface_adapter_path():
    with patch(
        "pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.HuggingFaceEncoderAdapter",
        TinyHFEncoder,
    ):
        model = GenericVariationalAutoencoder(
            encoder_name="hf-tiny",
            use_huggingface=True,
            in_channels=3,
            latent_dim=None,
        )

    x = torch.randn(1, 3, 32, 32)
    output = model(x)

    assert output.reconstruction.shape == x.shape
    assert output.mu.shape == output.logvar.shape == output.z.shape
    assert output.mu.shape[1] == 8
    # HF scale_factor should be 4 (32/8)
    assert model._hf_scale_factor == 4
    assert model.decoder.scale_factor == 4


def test_generic_variational_autoencoder_encoder_depth():
    with patch(
        "pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.smp"
    ) as smp_mock:
        smp_mock.encoders.get_encoder.return_value = TinyEncoder()

        # Test depth 3
        model = GenericVariationalAutoencoder(
            encoder_name="tiny", encoder_depth=3, pretrained=False
        )
        assert model.scale_factor == 8
        smp_mock.encoders.get_encoder.assert_called_with(
            "tiny", in_channels=3, depth=3, weights=None
        )

        # Test depth 4
        model = GenericVariationalAutoencoder(
            encoder_name="tiny", encoder_depth=4, pretrained=False
        )
        assert model.scale_factor == 16

    # Backward compatibility: default encoder_depth=None -> 5
    with patch(
        "pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.smp"
    ) as smp_mock:
        smp_mock.encoders.get_encoder.return_value = TinyEncoder()
        model = GenericVariationalAutoencoder(
            encoder_name="tiny", encoder_depth=None, pretrained=False
        )
        assert model.scale_factor == 32
        smp_mock.encoders.get_encoder.assert_called_with(
            "tiny", in_channels=3, depth=5, weights=None
        )


def test_generic_variational_autoencoder_invalid_configs():
    with patch(
        "pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.smp"
    ) as smp_mock:
        smp_mock.encoders.get_encoder.return_value = TinyEncoder()

        # encoder_depth with HF
        with pytest.raises(ValueError, match="encoder_depth"):
            GenericVariationalAutoencoder(
                encoder_name="tiny", use_huggingface=True, encoder_depth=3
            )

        # feature_level with SMP
        with pytest.raises(ValueError, match="feature_level"):
            GenericVariationalAutoencoder(
                encoder_name="tiny", use_huggingface=False, feature_level=0
            )

        # invalid encoder_depth
        with pytest.raises(ValueError, match="encoder_depth"):
            GenericVariationalAutoencoder(encoder_name="tiny", encoder_depth=6)
        with pytest.raises(ValueError, match="encoder_depth"):
            GenericVariationalAutoencoder(encoder_name="tiny", encoder_depth=0)


def test_generic_variational_autoencoder_huggingface_feature_level():
    class MultiLevelHFEncoder(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.out_channels = 8

        def forward(self, x):
            # Return two levels: 2x downsampled and 4x downsampled
            return [
                torch.randn(x.shape[0], 8, x.shape[2] // 2, x.shape[3] // 2),
                torch.randn(x.shape[0], 8, x.shape[2] // 4, x.shape[3] // 4),
            ]

    with patch(
        "pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.HuggingFaceEncoderAdapter",
        MultiLevelHFEncoder,
    ):
        # Test level 0 (2x downsampling)
        model = GenericVariationalAutoencoder(
            encoder_name="hf-multi", use_huggingface=True, feature_level=0
        )
        x = torch.randn(1, 3, 32, 32)
        model(x)
        assert model._hf_scale_factor == 2

        # Test level 1 or -1 (4x downsampling)
        model = GenericVariationalAutoencoder(
            encoder_name="hf-multi", use_huggingface=True, feature_level=-1
        )
        model(x)
        assert model._hf_scale_factor == 4


def test_generic_variational_autoencoder_huggingface_non_square_scale():
    class NonSquareHFEncoder(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.out_channels = 8

        def forward(self, x):
            return [torch.randn(x.shape[0], 8, x.shape[2] // 2, x.shape[3] // 4)]

    with patch(
        "pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.HuggingFaceEncoderAdapter",
        NonSquareHFEncoder,
    ):
        model = GenericVariationalAutoencoder(
            encoder_name="hf-nonsquare", use_huggingface=True
        )
        x = torch.randn(1, 3, 32, 32)
        with pytest.raises(RuntimeError, match="Non-square scale factor"):
            model(x)


def test_generic_variational_autoencoder_reparameterize_is_differentiable():
    model = object.__new__(GenericVariationalAutoencoder)
    mu = torch.zeros(2, 3, 4, 4, requires_grad=True)
    logvar = torch.zeros_like(mu, requires_grad=True)

    z = GenericVariationalAutoencoder.reparameterize(model, mu, logvar)
    assert z.shape == mu.shape

    z.sum().backward()
    assert mu.grad is not None
    assert logvar.grad is not None


def test_generic_variational_autoencoder_requires_smp_when_missing():
    with patch(
        "pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.smp",
        None,
    ):
        with pytest.raises(ImportError, match="segmentation-models-pytorch"):
            GenericVariationalAutoencoder(encoder_name="resnet18")


def test_generic_variational_autoencoder_rejects_too_small_latent_dim():
    with patch(
        "pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.smp"
    ) as smp_mock:
        smp_mock.encoders.get_encoder.return_value = TinyEncoder()
        with pytest.raises(ValueError, match="latent_dim"):
            GenericVariationalAutoencoder(
                encoder_name="tiny", latent_dim=1, pretrained=False
            )


def test_variational_autoencoder_loss_mse_components_and_gradients():
    loss_fn = VariationalAutoencoderLoss(reconstruction_loss="mse", beta=0.5)
    target = torch.zeros(2, 3, 8, 8)
    reconstruction = torch.ones_like(target, requires_grad=True)
    mu = torch.zeros(2, 4, 2, 2, requires_grad=True)
    logvar = torch.zeros_like(mu, requires_grad=True)
    output = VariationalAutoencoderOutput(reconstruction, mu, logvar, mu)

    result = loss_fn(output, target)

    assert set(result) == {"loss", "reconstruction_loss", "kl_loss"}
    assert result["loss"].shape == torch.Size([])
    assert torch.isclose(result["reconstruction_loss"], torch.tensor(1.0))
    assert torch.isclose(result["kl_loss"], torch.tensor(0.0))

    result["loss"].backward()
    assert reconstruction.grad is not None
    assert mu.grad is not None
    assert logvar.grad is not None


@pytest.mark.parametrize("reconstruction_loss", ["l1", "bce_with_logits"])
def test_variational_autoencoder_loss_reconstruction_modes(reconstruction_loss):
    loss_fn = VariationalAutoencoderLoss(reconstruction_loss=reconstruction_loss)
    target = torch.zeros(1, 1, 4, 4)
    reconstruction = torch.full_like(target, 0.25, requires_grad=True)
    mu = torch.ones(1, 2, 1, 1, requires_grad=True)
    logvar = torch.zeros_like(mu, requires_grad=True)
    output = VariationalAutoencoderOutput(reconstruction, mu, logvar, mu)

    result = loss_fn(output, target)

    assert result["loss"] > result["reconstruction_loss"]
    assert result["kl_loss"] > 0


def test_variational_autoencoder_loss_validates_reconstruction_mode():
    with pytest.raises(ValueError, match="reconstruction_loss"):
        VariationalAutoencoderLoss(reconstruction_loss="ssim")


def test_variational_autoencoder_loss_validates_model_output_type():
    loss_fn = VariationalAutoencoderLoss()
    with pytest.raises(TypeError, match="VariationalAutoencoderOutput"):
        loss_fn(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))


def test_variational_autoencoder_model_logs_train_val_and_test_components():
    cfg = OmegaConf.create(
        {
            "model": {"_target_": "unused"},
            "loss": {"_target_": "unused"},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler_list": [],
        }
    )
    domain_model = TinyVAEOutputModel()
    loss_fn = MagicMock()
    loss_fn.side_effect = lambda output, target: {
        "loss": output.reconstruction.mean() + target.mean(),
        "reconstruction_loss": torch.tensor(0.25),
        "kl_loss": torch.tensor(0.5),
    }

    with patch.object(
        VariationalAutoencoderModel, "get_model", return_value=domain_model
    ):
        with patch.object(
            VariationalAutoencoderModel, "get_loss_function", return_value=loss_fn
        ):
            pl_model = VariationalAutoencoderModel(cfg)

    pl_model.log = MagicMock()
    batch = {"image": torch.randn(2, 3, 8, 8), "target": torch.randn(2, 3, 8, 8)}

    train_loss = pl_model.training_step(batch, 0)
    val_loss = pl_model.validation_step(batch, 0)
    test_loss = pl_model.test_step(batch, 0)

    assert train_loss.shape == torch.Size([])
    assert val_loss.shape == torch.Size([])
    assert test_loss.shape == torch.Size([])
    assert loss_fn.call_count == 3
    logged_names = [call.args[0] for call in pl_model.log.call_args_list]
    assert "train/loss" in logged_names
    assert "train/reconstruction_loss" in logged_names
    assert "train/kl_loss" in logged_names
    assert "val/loss" in logged_names
    assert "test/loss" in logged_names


def test_variational_autoencoder_model_accepts_plain_loss_tensor():
    cfg = OmegaConf.create(
        {
            "model": {"_target_": "unused"},
            "loss": {"_target_": "unused"},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler_list": [],
        }
    )
    domain_model = nn.Conv2d(3, 3, kernel_size=1)

    with patch.object(
        VariationalAutoencoderModel, "get_model", return_value=domain_model
    ):
        with patch.object(
            VariationalAutoencoderModel,
            "get_loss_function",
            return_value=nn.MSELoss(),
        ):
            pl_model = VariationalAutoencoderModel(cfg)

    pl_model.log = MagicMock()
    batch = {"image": torch.randn(1, 3, 4, 4)}

    loss = pl_model.training_step(batch, 0)

    assert loss.shape == torch.Size([])
    assert "train/loss" in [call.args[0] for call in pl_model.log.call_args_list]
