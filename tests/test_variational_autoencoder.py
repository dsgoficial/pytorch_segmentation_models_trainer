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

    assert set(result) == {
        "loss",
        "reconstruction_loss",
        "kl_loss",
        "weighted_reconstruction_loss",
        "weighted_kl_loss",
    }
    assert result["loss"].shape == torch.Size([])
    assert torch.isclose(result["reconstruction_loss"], torch.tensor(1.0))
    assert torch.isclose(result["kl_loss"], torch.tensor(0.0))

    result["loss"].backward()
    assert reconstruction.grad is not None
    assert mu.grad is not None
    assert logvar.grad is not None


@pytest.mark.parametrize("reconstruction_loss", ["l1", "bce_with_logits", "smooth_l1"])
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


def test_variational_autoencoder_loss_smooth_l1_scalar_and_gradients():
    loss_fn = VariationalAutoencoderLoss(
        reconstruction_loss="smooth_l1", smooth_l1_beta=0.1
    )
    target = torch.zeros(2, 3, 8, 8)
    reconstruction = torch.ones_like(target, requires_grad=True)
    mu = torch.zeros(2, 4, 2, 2, requires_grad=True)
    logvar = torch.zeros_like(mu, requires_grad=True)
    output = VariationalAutoencoderOutput(reconstruction, mu, logvar, mu)

    result = loss_fn(output, target)

    assert result["loss"].shape == torch.Size([])
    assert result["reconstruction_loss"] > 0

    result["loss"].backward()
    assert reconstruction.grad is not None
    assert mu.grad is not None
    assert logvar.grad is not None


def test_variational_autoencoder_loss_smooth_l1_beta_affects_value():
    target = torch.zeros(1, 1, 4, 4)
    reconstruction = torch.full_like(target, 0.05, requires_grad=False)
    mu = torch.zeros(1, 2, 1, 1)
    logvar = torch.zeros_like(mu)
    output = VariationalAutoencoderOutput(reconstruction, mu, logvar, mu)

    # small beta → L1 regime for error=0.05; large beta → L2 regime
    loss_small_beta = VariationalAutoencoderLoss(
        reconstruction_loss="smooth_l1", smooth_l1_beta=0.01
    )
    loss_large_beta = VariationalAutoencoderLoss(
        reconstruction_loss="smooth_l1", smooth_l1_beta=1.0
    )

    result_small = loss_small_beta(output, target)
    result_large = loss_large_beta(output, target)

    # In L1 regime: loss ≈ |e| - beta/2; in L2 regime: loss ≈ e²/(2*beta)
    # Both give different values for same input
    assert not torch.isclose(
        result_small["reconstruction_loss"], result_large["reconstruction_loss"]
    )


def test_variational_autoencoder_loss_smooth_l1_governed_by_reconstruction_weight():
    target = torch.zeros(1, 1, 4, 4)
    reconstruction = torch.full_like(target, 0.5, requires_grad=False)
    mu = torch.zeros(1, 2, 1, 1)
    logvar = torch.zeros_like(mu)
    output = VariationalAutoencoderOutput(reconstruction, mu, logvar, mu)

    loss_w1 = VariationalAutoencoderLoss(
        reconstruction_loss="smooth_l1", reconstruction_weight=1.0, beta=0.0
    )
    loss_w2 = VariationalAutoencoderLoss(
        reconstruction_loss="smooth_l1", reconstruction_weight=2.0, beta=0.0
    )

    result_w1 = loss_w1(output, target)
    result_w2 = loss_w2(output, target)

    assert torch.isclose(result_w2["loss"], 2.0 * result_w1["loss"])
    assert torch.isclose(
        result_w2["weighted_reconstruction_loss"],
        2.0 * result_w1["weighted_reconstruction_loss"],
    )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_variational_autoencoder_loss_ms_ssim_contract_and_gradients(batch_size):
    loss_fn = VariationalAutoencoderLoss(
        reconstruction_loss="ms_ssim",
        beta=0.25,
        ms_ssim_data_range=1.0,
    )
    target = torch.rand(batch_size, 3, 32, 32)
    reconstruction = torch.rand_like(target, requires_grad=True)
    mu = torch.zeros(batch_size, 4, 8, 8, requires_grad=True)
    logvar = torch.zeros_like(mu, requires_grad=True)
    output = VariationalAutoencoderOutput(reconstruction, mu, logvar, mu)

    result = loss_fn(output, target)

    assert set(result) == {
        "loss",
        "reconstruction_loss",
        "kl_loss",
        "weighted_reconstruction_loss",
        "weighted_kl_loss",
        "ms_ssim_loss",
    }
    assert result["loss"].shape == torch.Size([])
    assert result["loss"].dtype == reconstruction.dtype
    assert torch.isfinite(result["loss"])
    assert torch.isclose(result["reconstruction_loss"], result["ms_ssim_loss"])

    result["loss"].backward()
    assert reconstruction.grad is not None
    assert mu.grad is not None
    assert logvar.grad is not None


def test_variational_autoencoder_loss_ms_ssim_identical_target_is_lower():
    loss_fn = VariationalAutoencoderLoss(
        reconstruction_loss="ms_ssim",
        ms_ssim_data_range=1.0,
        beta=0.0,
    )
    target = torch.rand(1, 3, 32, 32)
    mu = torch.zeros(1, 2, 8, 8)
    logvar = torch.zeros_like(mu)

    perfect = VariationalAutoencoderOutput(target.clone(), mu, logvar, mu)
    degraded = VariationalAutoencoderOutput(1.0 - target, mu, logvar, mu)

    perfect_loss = loss_fn(perfect, target)["loss"]
    degraded_loss = loss_fn(degraded, target)["loss"]

    assert perfect_loss < degraded_loss


def test_variational_autoencoder_loss_ms_ssim_random_inputs_are_not_exorbitant():
    loss_fn = VariationalAutoencoderLoss(
        reconstruction_loss="ms_ssim",
        ms_ssim_data_range=1.0,
        beta=0.0,
    )
    target = torch.rand(2, 3, 64, 64)
    reconstruction = torch.rand_like(target, requires_grad=True)
    mu = torch.zeros(2, 4, 16, 16)
    logvar = torch.zeros_like(mu)
    output = VariationalAutoencoderOutput(reconstruction, mu, logvar, mu)

    result = loss_fn(output, target)

    assert result["ms_ssim_loss"].item() <= 2.0
    assert result["loss"].item() <= 2.0


def test_variational_autoencoder_loss_ms_ssim_denormalizes_imagenet_inputs():
    target_01 = torch.rand(1, 3, 64, 64)
    reconstruction_01 = torch.rand_like(target_01)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)
    target_norm = (target_01 - mean_tensor) / std_tensor
    reconstruction_norm = (reconstruction_01 - mean_tensor) / std_tensor
    mu = torch.zeros(1, 4, 16, 16)
    output_01 = VariationalAutoencoderOutput(reconstruction_01, mu, mu, mu)
    output_norm = VariationalAutoencoderOutput(reconstruction_norm, mu, mu, mu)

    plain_loss = VariationalAutoencoderLoss(
        reconstruction_loss="ms_ssim",
        ms_ssim_data_range=1.0,
        beta=0.0,
    )
    denorm_loss = VariationalAutoencoderLoss(
        reconstruction_loss="ms_ssim",
        ms_ssim_data_range=1.0,
        ms_ssim_input_is_normalized=True,
        ms_ssim_mean=mean,
        ms_ssim_std=std,
        beta=0.0,
    )

    expected = plain_loss(output_01, target_01)["ms_ssim_loss"]
    actual = denorm_loss(output_norm, target_norm)["ms_ssim_loss"]

    assert actual.item() == pytest.approx(expected.item(), rel=1e-5)


def test_smooth_l1_ms_ssim_denormalization_only_affects_ms_ssim_component():
    target_01 = torch.rand(1, 3, 64, 64)
    reconstruction_01 = torch.rand_like(target_01)
    mean = [0.5, 0.5, 0.5]
    std = [0.25, 0.25, 0.25]
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)
    target_norm = (target_01 - mean_tensor) / std_tensor
    reconstruction_norm = (reconstruction_01 - mean_tensor) / std_tensor
    mu = torch.zeros(1, 4, 16, 16)
    output_norm = VariationalAutoencoderOutput(reconstruction_norm, mu, mu, mu)

    loss_fn = VariationalAutoencoderLoss(
        reconstruction_loss="smooth_l1_ms_ssim",
        smooth_l1_weight=0.8,
        ms_ssim_weight=0.2,
        ms_ssim_input_is_normalized=True,
        ms_ssim_mean=mean,
        ms_ssim_std=std,
        beta=0.0,
    )
    result = loss_fn(output_norm, target_norm)

    expected_smooth_l1 = torch.nn.functional.smooth_l1_loss(
        reconstruction_norm, target_norm, beta=loss_fn.smooth_l1_beta
    )
    assert result["smooth_l1_loss"].item() == pytest.approx(expected_smooth_l1.item())
    assert result["ms_ssim_loss"].item() <= 2.0


def test_variational_autoencoder_loss_smooth_l1_ms_ssim_components_and_weights():
    loss_fn = VariationalAutoencoderLoss(
        reconstruction_loss="smooth_l1_ms_ssim",
        smooth_l1_beta=0.2,
        smooth_l1_weight=0.75,
        ms_ssim_weight=0.25,
        ms_ssim_data_range=1.0,
        beta=0.0,
    )
    target = torch.zeros(2, 3, 32, 32)
    reconstruction = torch.full_like(target, 0.25, requires_grad=True)
    mu = torch.zeros(2, 4, 8, 8, requires_grad=True)
    logvar = torch.zeros_like(mu, requires_grad=True)
    output = VariationalAutoencoderOutput(reconstruction, mu, logvar, mu)

    result = loss_fn(output, target)

    assert {
        "smooth_l1_loss",
        "ms_ssim_loss",
        "weighted_smooth_l1_loss",
        "weighted_ms_ssim_loss",
    }.issubset(result)
    expected_reconstruction = (
        result["weighted_smooth_l1_loss"] + result["weighted_ms_ssim_loss"]
    )
    assert torch.isclose(result["reconstruction_loss"], expected_reconstruction)
    assert torch.isclose(result["loss"], expected_reconstruction)
    assert torch.isclose(
        result["weighted_smooth_l1_loss"], 0.75 * result["smooth_l1_loss"]
    )
    assert torch.isclose(result["weighted_ms_ssim_loss"], 0.25 * result["ms_ssim_loss"])

    result["loss"].backward()
    assert reconstruction.grad is not None
    assert mu.grad is not None
    assert logvar.grad is not None


def test_variational_autoencoder_loss_ms_ssim_defaults_to_pure_structural_term():
    loss_fn = VariationalAutoencoderLoss(reconstruction_loss="smooth_l1_ms_ssim")

    assert loss_fn.ms_ssim_alpha == pytest.approx(1.0)
    assert loss_fn.ms_ssim_compensation == pytest.approx(1.0)
    assert loss_fn.ms_ssim_loss.alpha == pytest.approx(1.0)
    assert loss_fn.ms_ssim_loss.compensation == pytest.approx(1.0)


def test_variational_autoencoder_loss_rejects_negative_component_weights():
    with pytest.raises(ValueError, match="smooth_l1_weight"):
        VariationalAutoencoderLoss(
            reconstruction_loss="smooth_l1_ms_ssim", smooth_l1_weight=-0.1
        )
    with pytest.raises(ValueError, match="ms_ssim_weight"):
        VariationalAutoencoderLoss(
            reconstruction_loss="smooth_l1_ms_ssim", ms_ssim_weight=-0.1
        )


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
        "weighted_reconstruction_loss": torch.tensor(0.25),
        "weighted_kl_loss": torch.tensor(0.5),
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
    assert "train/weighted_reconstruction_loss" in logged_names
    assert "train/weighted_kl_loss" in logged_names
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


def test_variational_autoencoder_model_logs_smooth_l1_ms_ssim_components():
    cfg = OmegaConf.create(
        {
            "model": {"_target_": "unused"},
            "loss": {"_target_": "unused"},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.001},
            "scheduler_list": [],
        }
    )
    domain_model = TinyVAEOutputModel()
    loss_fn = VariationalAutoencoderLoss(
        reconstruction_loss="smooth_l1_ms_ssim",
        smooth_l1_weight=0.8,
        ms_ssim_weight=0.2,
        ms_ssim_data_range=1.0,
        beta=0.0,
    )

    with patch.object(
        VariationalAutoencoderModel, "get_model", return_value=domain_model
    ):
        with patch.object(
            VariationalAutoencoderModel, "get_loss_function", return_value=loss_fn
        ):
            pl_model = VariationalAutoencoderModel(cfg)

    pl_model.log = MagicMock()
    batch = {"image": torch.rand(1, 3, 32, 32), "target": torch.rand(1, 3, 32, 32)}

    loss = pl_model.training_step(batch, 0)

    assert loss.shape == torch.Size([])
    logged_names = [call.args[0] for call in pl_model.log.call_args_list]
    assert "train/loss" in logged_names
    assert "train/smooth_l1_loss" in logged_names
    assert "train/ms_ssim_loss" in logged_names
    assert "train/weighted_smooth_l1_loss" in logged_names
    assert "train/weighted_ms_ssim_loss" in logged_names


@pytest.mark.parametrize(
    "mu_val,logvar_val",
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (2.0, -2.0),
        (0.0, 3.0),
    ],
)
def test_kl_loss_always_nonneg(mu_val, logvar_val):
    loss_fn = VariationalAutoencoderLoss(reconstruction_loss="mse")
    mu = torch.full((2, 4, 4, 4), mu_val)
    logvar = torch.full((2, 4, 4, 4), logvar_val)
    kl = loss_fn._compute_kl_loss(mu, logvar)
    assert kl >= 0.0, f"KL negative at mu={mu_val}, logvar={logvar_val}: {kl}"


def test_kl_loss_mean_scale_comparable_to_mse():
    loss_fn = VariationalAutoencoderLoss(reconstruction_loss="mse", beta=1.0)
    target = torch.rand(4, 3, 16, 16)
    reconstruction = torch.rand_like(target, requires_grad=True)
    mu = torch.randn(4, 4, 4, 4, requires_grad=True)
    logvar = torch.zeros_like(mu, requires_grad=True)
    output = VariationalAutoencoderOutput(reconstruction, mu, logvar, mu)

    result = loss_fn(output, target)

    assert result["reconstruction_loss"] < 1.0
    assert result["kl_loss"] >= 0.0


def test_vae_output_activation_sigmoid_bounds_reconstruction():
    with patch(
        "pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.smp"
    ) as smp_mock:
        smp_mock.encoders.get_encoder.return_value = TinyEncoder()
        model = GenericVariationalAutoencoder(
            encoder_name="tiny",
            in_channels=3,
            latent_dim=4,
            pretrained=False,
            output_activation="sigmoid",
        )

    x = torch.randn(2, 3, 32, 32) * 5
    output = model(x)

    assert output.reconstruction.min() >= 0.0
    assert output.reconstruction.max() <= 1.0


def test_vae_logvar_clamp_limits_values():
    with patch(
        "pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.smp"
    ) as smp_mock:
        smp_mock.encoders.get_encoder.return_value = TinyEncoder()
        model = GenericVariationalAutoencoder(
            encoder_name="tiny",
            in_channels=3,
            latent_dim=4,
            pretrained=False,
            logvar_clamp=(-2.0, 2.0),
        )

    x = torch.randn(2, 3, 32, 32)
    mu, logvar = model.encode(x)

    assert logvar.min() >= -2.0
    assert logvar.max() <= 2.0


# ──────────────────────────────────────────────────────────────────────────────
# free_bits tests
# ──────────────────────────────────────────────────────────────────────────────


def test_free_bits_floors_collapsed_kl():
    # mu=0, logvar=0 → kl_per_dim=0 < free_bits=0.5 → clamped to 0.5
    loss_fn = VariationalAutoencoderLoss(free_bits=0.5)
    mu = torch.zeros(2, 4, 2, 2)
    logvar = torch.zeros_like(mu)
    kl = loss_fn._compute_kl_loss(mu, logvar)
    assert kl.item() == pytest.approx(0.5)


def test_free_bits_no_effect_when_kl_above_threshold():
    # large mu → kl >> free_bits → result identical with and without floor
    loss_fn_with = VariationalAutoencoderLoss(free_bits=0.1)
    loss_fn_without = VariationalAutoencoderLoss(free_bits=0.0)
    mu = torch.full((2, 4, 2, 2), 3.0)
    logvar = torch.zeros_like(mu)
    assert loss_fn_with._compute_kl_loss(mu, logvar).item() == pytest.approx(
        loss_fn_without._compute_kl_loss(mu, logvar).item()
    )


def test_free_bits_zero_matches_analytic_formula():
    loss_fn = VariationalAutoencoderLoss(free_bits=0.0)
    mu = torch.randn(2, 4, 2, 2)
    logvar = torch.randn(2, 4, 2, 2)
    kl = loss_fn._compute_kl_loss(mu, logvar)
    expected = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    assert kl.item() == pytest.approx(expected.item())


def test_free_bits_blocks_gradient_for_collapsed_dims():
    # Collapsed dim (kl=0 < free_bits=0.5): clamp output is constant → grad=0
    mu = torch.zeros(1, 1, 1, 1, requires_grad=True)
    logvar = torch.zeros(1, 1, 1, 1, requires_grad=True)
    loss_fn = VariationalAutoencoderLoss(free_bits=0.5)
    loss_fn._compute_kl_loss(mu, logvar).backward()
    assert mu.grad.abs().max().item() == pytest.approx(0.0)
    assert logvar.grad.abs().max().item() == pytest.approx(0.0)


def test_free_bits_allows_gradient_for_active_dims():
    # Active dim (kl > free_bits): gradient flows normally
    mu = torch.full((1, 1, 1, 1), 3.0, requires_grad=True)
    logvar = torch.zeros(1, 1, 1, 1, requires_grad=True)
    loss_fn = VariationalAutoencoderLoss(free_bits=0.1)
    loss_fn._compute_kl_loss(mu, logvar).backward()
    assert mu.grad.abs().max().item() > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# kl_balance tests
# ──────────────────────────────────────────────────────────────────────────────


def test_kl_balance_scales_weighted_kl_by_dimension_ratio():
    # target: (2,3,8,8) → data_numel=192; mu: (2,4,2,2) → latent_numel=16; ratio=12
    target = torch.zeros(2, 3, 8, 8)
    reconstruction = torch.zeros_like(target)
    mu = torch.ones(2, 4, 2, 2, requires_grad=True)
    logvar = torch.zeros(2, 4, 2, 2, requires_grad=True)
    output = VariationalAutoencoderOutput(reconstruction, mu, logvar, mu)

    result_balanced = VariationalAutoencoderLoss(kl_balance=True)(output, target)
    result_plain = VariationalAutoencoderLoss(kl_balance=False)(output, target)

    ratio = float(target[0].numel()) / float(mu[0].numel())  # 12.0
    assert result_balanced["weighted_kl_loss"].item() == pytest.approx(
        result_plain["weighted_kl_loss"].item() * ratio, rel=1e-5
    )


def test_kl_balance_does_not_affect_raw_kl_loss_key():
    target = torch.zeros(2, 3, 8, 8)
    reconstruction = torch.zeros_like(target)
    mu = torch.ones(2, 4, 2, 2)
    logvar = torch.zeros_like(mu)
    output = VariationalAutoencoderOutput(reconstruction, mu, logvar, mu)

    r_with = VariationalAutoencoderLoss(kl_balance=True)(output, target)
    r_without = VariationalAutoencoderLoss(kl_balance=False)(output, target)

    assert r_with["kl_loss"].item() == pytest.approx(r_without["kl_loss"].item())


def test_kl_balance_false_is_default():
    assert VariationalAutoencoderLoss().kl_balance is False


def test_free_bits_default_is_zero():
    assert VariationalAutoencoderLoss().free_bits == pytest.approx(0.0)


def test_free_bits_and_kl_balance_compose():
    # mu=0 → kl_per_dim=0 < free_bits=0.5 → kl_loss=0.5
    # kl_balance: ratio = (3*8*8) / (4*2*2) = 192/16 = 12
    # weighted_kl_loss = beta * kl_loss * ratio = 1.0 * 0.5 * 12 = 6.0
    target = torch.zeros(2, 3, 8, 8)
    reconstruction = torch.zeros_like(target)
    mu = torch.zeros(2, 4, 2, 2)
    logvar = torch.zeros_like(mu)
    output = VariationalAutoencoderOutput(reconstruction, mu, logvar, mu)

    result = VariationalAutoencoderLoss(free_bits=0.5, kl_balance=True, beta=1.0)(
        output, target
    )

    assert result["kl_loss"].item() == pytest.approx(0.5)
    ratio = float(target[0].numel()) / float(mu[0].numel())
    assert result["weighted_kl_loss"].item() == pytest.approx(0.5 * ratio)


def test_vae_logvar_clamp_default_is_none():
    with patch(
        "pytorch_segmentation_models_trainer.custom_models.variational_autoencoder.smp"
    ) as smp_mock:
        smp_mock.encoders.get_encoder.return_value = TinyEncoder()
        model = GenericVariationalAutoencoder(
            encoder_name="tiny", in_channels=3, pretrained=False
        )

    assert model.logvar_clamp is None


def test_variational_autoencoder_model_computes_metrics_when_configured():
    import torchmetrics

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
    domain_model = TinyVAEOutputModel()
    loss_fn = MagicMock()
    loss_fn.side_effect = lambda output, target: {
        "loss": output.reconstruction.mean() + target.mean(),
        "reconstruction_loss": torch.tensor(0.25),
        "kl_loss": torch.tensor(0.5),
        "weighted_reconstruction_loss": torch.tensor(0.25),
        "weighted_kl_loss": torch.tensor(0.5),
    }

    with patch.object(
        VariationalAutoencoderModel, "get_model", return_value=domain_model
    ):
        with patch.object(
            VariationalAutoencoderModel, "get_loss_function", return_value=loss_fn
        ):
            pl_model = VariationalAutoencoderModel(cfg)

    pl_model.log = MagicMock()
    pl_model.log_dict = MagicMock()
    batch = {"image": torch.randn(2, 3, 8, 8), "target": torch.randn(2, 3, 8, 8)}

    pl_model.training_step(batch, 0)
    pl_model.validation_step(batch, 0)

    assert pl_model.log_dict.call_count == 2
    first_call_keys = set(pl_model.log_dict.call_args_list[0].args[0].keys())
    assert any("train/" in k for k in first_call_keys)
    second_call_keys = set(pl_model.log_dict.call_args_list[1].args[0].keys())
    assert any("val/" in k for k in second_call_keys)
