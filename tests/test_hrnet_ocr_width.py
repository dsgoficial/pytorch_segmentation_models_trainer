"""Tests for HRNet-OCR of arbitrary width (``build_hrnet_ocr``) and the
``branch_hook`` argument of ``HRNetOCR.forward`` (context injection into a
resolution branch before stage 3)."""
import pytest
import torch

from pytorch_segmentation_models_trainer.custom_models.hrnet_models import seg_hrnet_ocr
from pytorch_segmentation_models_trainer.custom_models.hrnet_models.seg_hrnet_ocr import (
    HRNetOCR,
    build_hrnet_ocr,
)


@pytest.fixture(scope="module")
def w18():
    torch.manual_seed(0)
    return build_hrnet_ocr(width=18, n_classes=6).eval()


def test_build_returns_hrnet_ocr_with_scaled_branches(w18):
    assert isinstance(w18, HRNetOCR)
    assert list(w18.stage4_cfg.num_channels) == [18, 36, 72, 144]
    n = sum(p.numel() for p in w18.parameters())
    assert 11e6 < n < 13e6  # HRNet-W18-OCR ~12.1M


@pytest.mark.parametrize("batch", [1, 2])
def test_forward_contract(w18, batch):
    out = w18(torch.randn(batch, 3, 64, 64))
    assert set(out) == {"aux", "out"}
    assert out["out"].shape == (batch, 6, 16, 16)
    assert out["aux"].shape == (batch, 6, 16, 16)
    assert out["out"].dtype == torch.float32


def test_branch_hook_receives_stage3_inputs_and_identity_is_noop(w18):
    x = torch.randn(1, 3, 64, 64)
    seen = []

    def hook(xs):
        seen.append([tuple(t.shape[1:]) for t in xs])
        return xs

    with torch.no_grad():
        ref = w18(x)["out"]
        got = w18(x, branch_hook=hook)["out"]
    assert seen == [[(18, 16, 16), (36, 8, 8), (72, 4, 4)]]
    assert torch.equal(ref, got)


def test_branch_hook_changes_output_and_carries_gradient(w18):
    x = torch.randn(1, 3, 64, 64)
    delta = torch.zeros(1, 36, 8, 8, requires_grad=True)

    def hook(xs):
        return [xs[0], xs[1] + delta + 1.0, xs[2]]

    with torch.no_grad():
        ref = w18(x)["out"]
    out = w18(x, branch_hook=hook)["out"]
    assert not torch.allclose(ref, out)
    out.sum().backward()
    assert delta.grad is not None and delta.grad.abs().sum() > 0


def test_imagenet_timm_weights_are_loaded(monkeypatch):
    import timm

    src = timm.create_model("hrnet_w18", pretrained=False)
    with torch.no_grad():
        src.conv1.weight.fill_(0.5)
    calls = {}

    def fake_create_model(name, pretrained):
        calls["args"] = (name, pretrained)
        return src

    monkeypatch.setattr(seg_hrnet_ocr.timm, "create_model", fake_create_model)
    m = build_hrnet_ocr(width=18, n_classes=6, imagenet_timm=True)
    assert calls["args"] == ("hrnet_w18", True)
    assert torch.equal(m.conv1.weight, src.conv1.weight)
    # every backbone tensor of the timm model with a matching name/shape is copied
    assert m.loaded_imagenet_keys == len(
        [k for k, v in src.state_dict().items()
         if k in m.state_dict() and m.state_dict()[k].shape == v.shape])
    assert m.loaded_imagenet_keys > 1000


def test_invalid_width_raises():
    with pytest.raises(ValueError):
        build_hrnet_ocr(width=0)


def test_example_yaml_instantiates_with_hydra():
    from pathlib import Path

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path(__file__).parents[1] / "conf" / "examples" / "hrnet_ocr_w18.yaml")
    cfg.model.imagenet_timm = False  # no network in tests
    m = instantiate(cfg.model)
    assert isinstance(m, HRNetOCR) and m.n_classes == 6
