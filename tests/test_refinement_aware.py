# -*- coding: utf-8 -*-
"""Tests for the refinement-aware base training wrapper (R2-C)."""

import pytest
import segmentation_models_pytorch as smp
import torch

from pytorch_segmentation_models_trainer.custom_models.refinement_aware import (
    RefinementAwareWrapper,
    canny_targets,
)


def _inner(classes=3):
    return smp.UPerNet(
        encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=classes
    )


def _wrapper(**kw):
    kw.setdefault("superclass_index", 1)
    return RefinementAwareWrapper(_inner(), **kw)


def _images():
    x = torch.zeros(2, 3, 64, 64)
    x[:, :, :, 32:] = 3.0  # vertical step edge in normalized space
    return x


def _masks():
    m = torch.ones(2, 64, 64, dtype=torch.long)  # all superclass
    m[:, :8] = 0
    m[:, -2:] = 255
    return m


class TestCanny:
    def test_step_edge_detected_at_feature_resolution(self):
        t = canny_targets(_images(), size=(16, 16))
        assert t.shape == (2, 1, 16, 16)
        assert t[:, :, 4:12, 7:9].sum() > 0
        assert t[:, :, :, :5].sum() == 0 and t[:, :, :, 11:].sum() == 0

    def test_flat_image_has_no_edges(self):
        assert canny_targets(torch.zeros(1, 3, 32, 32), size=(8, 8)).sum() == 0


class TestWrapper:
    def test_forward_returns_logits_and_stores_features(self):
        w = _wrapper()
        out = w(_images())
        assert out.shape == (2, 3, 64, 64)
        assert w.last_features.shape[1] == 256
        assert w.last_images is not None

    def test_validation_of_arguments(self):
        with pytest.raises(ValueError):
            _wrapper(num_subprototypes=1, subprototype_weight=1.0)
        with pytest.raises(ValueError):
            _wrapper(edge_region="foo")

    def test_auxiliary_losses_are_finite_and_have_gradients(self):
        w = _wrapper(edge_weight=1.0, subprototype_weight=1.0, num_subprototypes=4)
        w(_images())
        losses = w.compute_auxiliary_losses(_masks())
        assert set(losses) == {"edge", "subproto"}
        total = sum(losses.values())
        assert torch.isfinite(total)
        total.backward()
        assert w.edge_head.weight.grad is not None
        assert w.subprototypes.grad is not None
        assert any(p.grad is not None for p in w.model.decoder.parameters())
        assert w.last_features is None  # freed after use

    def test_disabled_terms_are_absent(self):
        w = _wrapper(edge_weight=0.0, subprototype_weight=0.0)
        assert w.edge_head is None and w.subprototypes is None
        w(_images())
        assert w.compute_auxiliary_losses(_masks()) == {}

    def test_no_superclass_pixels_gives_zero_subproto_loss(self):
        w = _wrapper(edge_weight=0.0, subprototype_weight=1.0)
        w(_images())
        m = torch.zeros(2, 64, 64, dtype=torch.long)
        assert w.compute_auxiliary_losses(m)["subproto"] == 0.0

    def test_edge_region_superclass_masks_edge_loss(self):
        w = _wrapper(edge_weight=1.0, subprototype_weight=0.0, edge_region="superclass")
        w(_images())
        m = torch.zeros(2, 64, 64, dtype=torch.long)  # no superclass
        assert w.compute_auxiliary_losses(m)["edge"] == 0.0

    def test_without_forward_returns_empty(self):
        assert _wrapper().compute_auxiliary_losses(_masks()) == {}

    def test_subprototype_loss_prefers_sharp_balanced_assignments(self):
        w = _wrapper(edge_weight=0.0, subprototype_weight=1.0, num_subprototypes=2)
        with torch.no_grad():
            w.subprototypes.copy_(torch.eye(2, 256) * 10)
        f = torch.zeros(1, 256, 2, 2)
        f[0, 0, :, 0] = 1.0  # half the pixels on prototype 0
        f[0, 1, :, 1] = 1.0  # half on prototype 1
        sup = torch.ones(1, 2, 2, dtype=torch.bool)
        good = w._subprototype_loss(f, sup)
        f2 = torch.zeros(1, 256, 2, 2)
        f2[0, 0] = 1.0  # all on prototype 0 (collapse)
        bad = w._subprototype_loss(f2, sup)
        assert good < bad

    def test_gfss_unwrap_returns_inner_model(self):
        w = _wrapper()
        assert w.gfss_unwrap() is w.model


def test_frozen_linear_head_segmenter_unwraps_refinement_wrapper():
    from pytorch_segmentation_models_trainer.few_shot.backbone import (
        FrozenLinearHeadSegmenter,
    )

    w = _wrapper()
    seg = FrozenLinearHeadSegmenter(w)
    assert seg.model is w.model and not seg.evidential


def test_model_adds_auxiliary_losses(tmp_path):
    from unittest.mock import MagicMock

    from omegaconf import OmegaConf

    from pytorch_segmentation_models_trainer.model_loader.model import Model

    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "pytorch_segmentation_models_trainer.custom_models.refinement_aware.RefinementAwareWrapper",
                "superclass_index": 1,
                "model": {
                    "_target_": "segmentation_models_pytorch.UPerNet",
                    "encoder_name": "resnet18",
                    "encoder_weights": None,
                    "in_channels": 3,
                    "classes": 3,
                },
            },
            "loss": {"_target_": "torch.nn.CrossEntropyLoss", "ignore_index": 255},
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
            "hyperparameters": {"batch_size": 2},
        }
    )
    m = Model(cfg)
    m.log = MagicMock()
    loss = m.training_step({"image": _images(), "mask": _masks()}, 0)
    loss = loss["loss"] if isinstance(loss, dict) else loss
    assert torch.isfinite(loss)
    names = [c.args[0] for c in m.log.call_args_list]
    assert "losses/train_edge" in names and "losses/train_subproto" in names
