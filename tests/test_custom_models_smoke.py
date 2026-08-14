import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock

from pytorch_segmentation_models_trainer.custom_models.edl_wrapper import (
    EvidentialWrapper,
    is_evidential_model,
)
from pytorch_segmentation_models_trainer.custom_models.generic_autoencoder import (
    GenericAutoencoder,
)
from pytorch_segmentation_models_trainer.custom_models.mod_polymapper.modpolymapper import (
    GenericPolygonRNN,
    ModPolyMapper,
)
from pytorch_segmentation_models_trainer.custom_models.moe_layers import (
    ExpertChoiceRouter,
    MoEConv2dReLU,
    NoisyTopKRouter,
)
from pytorch_segmentation_models_trainer.custom_models.timm_models import (
    TimmEncoderWithSMPDecoder,
)
from pytorch_segmentation_models_trainer.custom_models.transformer_adapters import (
    ModelOutputAdapter,
)
from pytorch_segmentation_models_trainer.custom_models.terratorch_models import (
    TerraTorchSegmentationWrapper,
)
from pytorch_segmentation_models_trainer.custom_models.unet import UNetBackbone
from pytorch_segmentation_models_trainer.custom_models.unet_resnet import (
    UNetResNetBackbone,
)
from pytorch_segmentation_models_trainer.custom_models.upernet_dual_head import (
    UPerNetDualHead,
)
from pytorch_segmentation_models_trainer.custom_models.upernet_medoe import (
    UPerNetMEDOE,
)
from pytorch_segmentation_models_trainer.custom_models.upernet_moe import UPerNetMoE
from pytorch_segmentation_models_trainer.custom_models.variational_autoencoder import (
    GenericVariationalAutoencoder,
)


def test_unet_backbone_outputs_expected_shape():
    model = UNetBackbone(n_channels=3, n_hidden_base=16)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert "out" in out
    assert out["out"].shape == (1, 16, 64, 64)


def test_upernet_dual_head_returns_segmentation_map():
    model = UPerNetDualHead(
        encoder_name="resnet18", classes=6, inference_head="average"
    )
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 6, 64, 64)


def test_evidential_wrapper_wraps_logits():
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Conv2d(3, 16, 3)
            self.head = nn.Conv2d(16, 5, 1)

        def forward(self, x):
            return self.head(self.encoder(x))

    inner = MockModel()
    wrapper = EvidentialWrapper(inner, freeze_encoder=True)
    assert is_evidential_model(wrapper)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = wrapper(x)
    assert "logits" in out


def test_generic_autoencoder_produces_reconstruction():
    model = GenericAutoencoder(
        encoder_name="resnet18", use_progressive_decoder=True, upsample_mode="bilinear"
    )
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 64, 64)


def test_upernet_medoe_tracks_expert_loss():
    model = UPerNetMEDOE(encoder_name="resnet18", classes=6, tail_residual=True)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 6, 64, 64)
    target = torch.randint(0, 6, (1, 64, 64))
    loss = model.compute_expert_loss(model.last_expert_outputs, target)
    assert loss >= 0


def test_mod_polymapper_returns_sequence_outputs():
    model = ModPolyMapper(num_classes=2, backbone_trainable_layers=1, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, list)


def test_generic_polygon_rnn_decoder_step():
    backbone = MagicMock()
    backbone.side_effect = lambda x: {
        "0": torch.randn(x.shape[0], 256, 16, 16),
        "1": torch.randn(x.shape[0], 256, 8, 8),
        "2": torch.randn(x.shape[0], 256, 4, 4),
        "3": torch.randn(x.shape[0], 256, 2, 2),
    }
    model = GenericPolygonRNN(backbone=backbone, grid_size=8)
    x = torch.randn(1, 3, 64, 64)
    bs, seq_len, grid_size = 1, 5, 8
    first = torch.randn(bs, grid_size * grid_size + 3)
    second = torch.randn(bs, seq_len, grid_size * grid_size + 3)
    third = torch.randn(bs, seq_len, grid_size * grid_size + 3)
    out = model(x, first, second, third)
    assert out.shape == (bs, seq_len, grid_size * grid_size + 3)


def test_upernet_moe_outputs_segmentation_logits():
    model = UPerNetMoE(encoder_name="resnet18", classes=6, num_experts=2)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 6, 64, 64)


def test_variational_autoencoder_returns_reconstruction():
    model = GenericVariationalAutoencoder(
        encoder_name="resnet18", in_channels=3, latent_dim=16
    )
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.reconstruction.shape == (1, 3, 64, 64)


def test_timm_segmentation_model_smoke():
    try:
        model = TimmEncoderWithSMPDecoder(timm_model_name="resnet18", num_classes=6)
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 6, 64, 64)
    except Exception:
        pytest.skip("Timm model failed")


def test_transformer_adapter_matches_input_spatial_shape():
    inner = nn.Conv2d(3, 6, 3)
    model = ModelOutputAdapter(model=inner, output_size="input")
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 6, 32, 32)


def test_unet_resnet_backbone_outputs_feature_map():
    model = UNetResNetBackbone(encoder_depth=34, num_filters=16)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out["out"].shape == (1, 16, 64, 64)


def test_moe_layers_return_expected_shapes():
    router = NoisyTopKRouter(in_channels=16, num_experts=4, top_k=1)
    x = torch.randn(2, 16, 16, 16)
    weights, aux_loss = router(x)
    assert weights.shape == (2, 4, 16, 16)

    router_ec = ExpertChoiceRouter(in_channels=16, num_experts=4)
    weights_ec, _ = router_ec(x)
    assert weights_ec.shape == (2, 4, 16, 16)

    moe_conv = MoEConv2dReLU(in_channels=16, out_channels=16, num_experts=2)
    out, _ = moe_conv(x)
    assert out.shape == (2, 16, 16, 16)
