"""Tests for the UNetFormer port (custom_models/unetformer.py): smp-style
3-step interface, input sizes below 256 px (window padding fallback), aux
head, and building blocks."""
import pytest
import torch
import torch.nn.functional as F

from pytorch_segmentation_models_trainer.custom_models.unetformer import (
    GLTBlock,
    GlobalLocalAttention,
    UNetFormerBackbone,
)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return UNetFormerBackbone(encoder_name="resnet18", encoder_weights=None, classes=6).eval()


@pytest.mark.parametrize("batch,size", [(1, 256), (2, 256), (2, 128), (1, 160), (1, 224)])
def test_forward_contract_any_multiple_of_32(model, batch, size):
    with torch.no_grad():
        out = model(torch.randn(batch, 3, size, size))
    assert out.shape == (batch, 6, size, size)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_three_step_smp_interface_matches_forward(model):
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        feats = model.encoder(x)
        logits = model.segmentation_head(model.decoder(feats))
        assert torch.equal(logits, model(x))
    assert [tuple(f.shape[-2:]) for f in feats][-4:] == [(32, 32), (16, 16), (8, 8), (4, 4)]


def test_gradient_reaches_encoder_and_decoder():
    torch.manual_seed(0)
    m = UNetFormerBackbone(encoder_name="resnet18", encoder_weights=None, classes=6).train()
    m(torch.randn(2, 3, 128, 128)).mean().backward()
    assert m.encoder.conv1.weight.grad.abs().sum() > 0
    assert m.decoder.b4.attn.qkv[0].weight.grad.abs().sum() > 0


def test_pad_uses_reflect_when_possible_identical_to_original():
    attn = GlobalLocalAttention(dim=16, num_heads=2, window_size=8)
    x = torch.randn(1, 16, 12, 12)  # pad 4 < 12: reflect, as in GeoSeg
    ref = F.pad(F.pad(x, (0, 4, 0, 0), mode="reflect"), (0, 0, 0, 4), mode="reflect")
    assert torch.equal(attn.pad(x, 8), ref)


def test_pad_falls_back_to_replicate_when_map_smaller_than_window():
    attn = GlobalLocalAttention(dim=16, num_heads=2, window_size=8)
    x = torch.randn(1, 16, 4, 4)  # pad 4 == 4: reflect impossible
    out = attn.pad(x, 8)
    assert out.shape == (1, 16, 8, 8)
    ref = F.pad(F.pad(x, (0, 4, 0, 0), mode="replicate"), (0, 0, 0, 4), mode="replicate")
    assert torch.equal(out, ref)


def test_pad_noop_when_divisible():
    attn = GlobalLocalAttention(dim=16, num_heads=2, window_size=8)
    x = torch.randn(1, 16, 16, 8)
    assert attn.pad(x, 8) is x


def test_attention_without_relative_position_embedding():
    attn = GlobalLocalAttention(dim=16, num_heads=2, window_size=8,
                                relative_pos_embedding=False).eval()
    assert not hasattr(attn, "relative_position_bias_table")
    with torch.no_grad():
        assert attn(torch.randn(1, 16, 8, 8)).shape == (1, 16, 8, 8)


def test_gltblock_with_drop_path():
    blk = GLTBlock(dim=16, num_heads=2, window_size=8, drop_path=0.1).train()
    assert type(blk.drop_path).__name__ == "DropPath"
    assert blk(torch.randn(2, 16, 8, 8)).shape == (2, 16, 8, 8)


def test_aux_head_only_in_training():
    torch.manual_seed(0)
    m = UNetFormerBackbone(encoder_name="resnet18", encoder_weights=None, classes=6,
                           use_aux_head=True).train()
    out = m(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 6, 128, 128)
    assert m.decoder._last_aux.shape == (2, 6, 128, 128)
    m.eval()
    del m.decoder._last_aux
    with torch.no_grad():
        m(torch.randn(1, 3, 128, 128))
    assert not hasattr(m.decoder, "_last_aux")


def test_example_yaml_instantiates_with_hydra():
    from pathlib import Path

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path(__file__).parents[1] / "conf" / "examples" / "unetformer_resnet18.yaml")
    cfg.model.encoder_weights = None  # no network in tests
    m = instantiate(cfg.model)
    assert isinstance(m, UNetFormerBackbone)
