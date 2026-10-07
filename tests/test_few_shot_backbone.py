# -*- coding: utf-8 -*-
"""Tests for the frozen linear-head segmenter and the base GFSS method API."""

import pytest
import segmentation_models_pytorch as smp
import torch

from pytorch_segmentation_models_trainer.few_shot.backbone import (
    FrozenLinearHeadSegmenter,
)
from pytorch_segmentation_models_trainer.few_shot.base_method import BaseGFSSMethod
from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy
from pytorch_segmentation_models_trainer.few_shot.methods.base_only import BaseOnly


def _upernet(classes=3):
    torch.manual_seed(0)
    return smp.UPerNet(
        encoder_name="resnet18",
        encoder_weights=None,
        classes=classes,
        decoder_channels=32,
    )


class TestFrozenLinearHeadSegmenter:
    def test_features_and_logits_reproduce_model(self):
        model = _upernet().eval()
        seg = FrozenLinearHeadSegmenter(model)
        x = torch.randn(2, 3, 64, 64)
        feats = seg.features(x)
        assert feats.shape == (2, 32, 16, 16)
        assert seg.weight.shape == (3, 32) and seg.bias.shape == (3,)
        logits = seg.linear(feats, seg.weight, seg.bias)
        torch.testing.assert_close(seg.upsample(logits, x.shape[-2:]), model(x))
        torch.testing.assert_close(seg.base_logits(x), model(x))

    def test_frozen_and_stays_in_eval(self):
        seg = FrozenLinearHeadSegmenter(_upernet())
        assert not any(p.requires_grad for p in seg.parameters())
        seg.train()
        assert not seg.training and not seg.model.training
        assert seg.features(torch.randn(1, 3, 64, 64)).grad_fn is None

    def test_weight_and_bias_are_detached_copies(self):
        seg = FrozenLinearHeadSegmenter(_upernet())
        w = seg.weight
        w += 1.0
        assert not torch.equal(w, seg.weight)

    def test_head_without_bias_gives_zero_bias(self):
        model = _upernet()
        model.segmentation_head[0].bias = None
        seg = FrozenLinearHeadSegmenter(model)
        assert torch.equal(seg.bias, torch.zeros(3))

    def test_non_linear_head_rejected(self):
        unet = smp.Unet(encoder_name="resnet18", encoder_weights=None, classes=3)
        with pytest.raises(ValueError, match="1x1"):
            FrozenLinearHeadSegmenter(unet)

    def test_upsample_resizes_to_target(self):
        seg = FrozenLinearHeadSegmenter(_upernet())
        out = seg.upsample(torch.randn(1, 3, 16, 16), (64, 64))
        assert out.shape == (1, 3, 64, 64)
        out = seg.upsample(torch.randn(1, 3, 16, 16), (60, 60))
        assert out.shape == (1, 3, 60, 60)


class TestBaseMethodApi:
    def test_defaults(self):
        h = ClassHierarchy({1: [1, 2]}, num_base_classes=2)
        m = BaseOnly()
        m.setup(h, torch.randn(2, 8), torch.zeros(2), not_novel_index=254)
        assert m.hierarchy is h and m.not_novel_index == 254
        assert m.transductive is False
        feats = torch.randn(1, 8, 4, 4)
        masks = torch.zeros(1, 4, 4, dtype=torch.long)
        assert m.init_from_support(feats, masks) is None
        assert m.support_loss(feats, masks) is None
        assert m.adapt_to_query(feats, masks, feats) is None

    def test_abstract_forward(self):
        with pytest.raises(TypeError):
            BaseGFSSMethod()


class TestBaseOnly:
    def test_novel_never_predicted_and_base_logits_kept(self):
        h = ClassHierarchy({1: [1, 2]}, num_base_classes=2)
        w, b = torch.randn(2, 8), torch.randn(2)
        m = BaseOnly(**{"unused": 1})
        m.setup(h, w, b)
        feats = torch.randn(3, 8, 4, 4)
        logits = m(feats)
        assert logits.shape == (3, 3, 4, 4)
        base = FrozenLinearHeadSegmenter.linear(feats, w, b)
        torch.testing.assert_close(logits[:, :2], base)
        assert (logits.argmax(1) != 2).all()
        assert torch.isfinite(logits).all()


class TestEvidentialBase:
    def test_wrapper_is_unwrapped_and_flagged(self):
        from pytorch_segmentation_models_trainer.custom_models.edl_wrapper import (
            EvidentialWrapper,
        )

        inner = _upernet()
        seg = FrozenLinearHeadSegmenter(EvidentialWrapper(inner))
        assert seg.evidential is True
        assert seg.model is inner
        x = torch.randn(1, 3, 64, 64)
        torch.testing.assert_close(seg.base_logits(x), inner.eval()(x))
        assert FrozenLinearHeadSegmenter(_upernet()).evidential is False


class TestBaseProbabilities:
    def test_softmax_and_dirichlet_mean(self):
        h = ClassHierarchy({1: [1, 2]}, num_base_classes=2)
        logits = torch.randn(1, 2, 3, 3)
        m = BaseOnly()
        m.setup(h, torch.randn(2, 4), torch.zeros(2))
        assert m.base_output == "softmax"
        torch.testing.assert_close(m.base_probabilities(logits), logits.softmax(1))
        m.setup(h, torch.randn(2, 4), torch.zeros(2), base_output="evidential")
        alpha = torch.nn.functional.softplus(logits) + 1
        torch.testing.assert_close(m.base_alpha(logits), alpha)
        torch.testing.assert_close(
            m.base_probabilities(logits), alpha / alpha.sum(1, keepdim=True)
        )

    def test_invalid_base_output(self):
        h = ClassHierarchy({1: [1, 2]}, num_base_classes=2)
        with pytest.raises(ValueError, match="base_output"):
            BaseOnly().setup(h, torch.randn(2, 4), torch.zeros(2), base_output="nig")
