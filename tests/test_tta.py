# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-04-12
        copyright            : (C) 2026 by Philipe Borba
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

Unit tests for the TTA module.
"""

import pytest
import torch

from pytorch_segmentation_models_trainer.tools.tta.tta import (
    D4_AUGMENTATIONS,
    FLIP_H,
    FLIP_V,
    ROT0,
    ROT180,
    ROT270,
    ROT90,
    ROTATION_AUGMENTATIONS,
    _augment,
    _deaugment,
    apply_tta,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, C, H, W = 2, 3, 16, 16


@pytest.fixture
def batch():
    torch.manual_seed(42)
    return torch.randn(B, C, H, W)


@pytest.fixture
def identity_model():
    """Model that returns its input unchanged."""
    return lambda x: x


@pytest.fixture
def constant_model():
    """Model that always returns a fixed constant tensor (all ones)."""

    def _model(x):
        return torch.ones_like(x)

    return _model


# ---------------------------------------------------------------------------
# _augment / _deaugment: round-trip identity for all D4 augmentations
# ---------------------------------------------------------------------------


class TestAugmentDeaugmentRoundTrip:
    @pytest.mark.parametrize("aug", D4_AUGMENTATIONS)
    def test_roundtrip_is_identity(self, batch, aug):
        """deaugment(augment(x, aug), aug) must equal x for all augmentations."""
        augmented = _augment(batch, aug)
        recovered = _deaugment(augmented, aug)
        assert torch.allclose(recovered, batch), (
            f"Round-trip failed for '{aug}': "
            f"max diff = {(recovered - batch).abs().max().item()}"
        )

    @pytest.mark.parametrize("aug", D4_AUGMENTATIONS)
    def test_augment_preserves_shape(self, batch, aug):
        """Augmentation must not change the tensor shape."""
        assert _augment(batch, aug).shape == batch.shape

    @pytest.mark.parametrize("aug", D4_AUGMENTATIONS)
    def test_deaugment_preserves_shape(self, batch, aug):
        """De-augmentation must not change the tensor shape."""
        assert _deaugment(_augment(batch, aug), aug).shape == batch.shape

    def test_augment_unknown_raises(self, batch):
        with pytest.raises(ValueError, match="Unknown TTA augmentation"):
            _augment(batch, "invalid_aug")

    def test_deaugment_unknown_raises(self, batch):
        with pytest.raises(ValueError, match="Unknown TTA augmentation"):
            _deaugment(batch, "invalid_aug")


# ---------------------------------------------------------------------------
# apply_tta: basic behaviour
# ---------------------------------------------------------------------------


class TestApplyTTABasic:
    def test_empty_augmentations_raises(self, batch, identity_model):
        with pytest.raises(ValueError, match="must not be empty"):
            apply_tta(identity_model, batch, augmentations=[])

    def test_unknown_augmentation_raises(self, batch, identity_model):
        with pytest.raises(ValueError, match="Unknown TTA augmentations"):
            apply_tta(identity_model, batch, augmentations=["rot0", "bad_aug"])

    def test_output_shape_preserved_tensor(self, batch, identity_model):
        """Output shape must match input batch shape."""
        result = apply_tta(identity_model, batch, augmentations=ROTATION_AUGMENTATIONS)
        assert result.shape == batch.shape

    def test_output_shape_preserved_dict(self, batch):
        """Dict-output model: each value shape must be preserved."""

        def dict_model(x):
            return {"seg": x, "extra": x * 2}

        result = apply_tta(dict_model, batch, augmentations=ROTATION_AUGMENTATIONS)
        assert isinstance(result, dict)
        assert result["seg"].shape == batch.shape
        assert result["extra"].shape == batch.shape

    def test_rot0_only_equals_direct_call(self, batch, identity_model):
        """TTA with only rot0 must equal a direct model call."""
        result = apply_tta(identity_model, batch, augmentations=[ROT0])
        assert torch.allclose(result, identity_model(batch))


# ---------------------------------------------------------------------------
# apply_tta: constant model — TTA must not change result
# ---------------------------------------------------------------------------


class TestApplyTTAConstantModel:
    @pytest.mark.parametrize(
        "augmentations",
        [
            ROTATION_AUGMENTATIONS,
            D4_AUGMENTATIONS,
            [ROT90, ROT180],
            [FLIP_H, FLIP_V],
        ],
    )
    def test_constant_model_is_unaffected_by_tta(
        self, batch, constant_model, augmentations
    ):
        """A model that always returns ones should give ones regardless of TTA."""
        result = apply_tta(constant_model, batch, augmentations=augmentations)
        expected = constant_model(batch)
        assert torch.allclose(
            result, expected
        ), f"TTA changed the output of a constant model with {augmentations}"


# ---------------------------------------------------------------------------
# apply_tta: identity model — average of de-augmented predictions equals input
# ---------------------------------------------------------------------------


class TestApplyTTAIdentityModel:
    @pytest.mark.parametrize(
        "augmentations",
        [
            ROTATION_AUGMENTATIONS,
            D4_AUGMENTATIONS,
            [ROT90, ROT270],  # subset: two rotations that cancel
            [FLIP_H],  # single flip
        ],
    )
    def test_identity_model_returns_input(self, batch, identity_model, augmentations):
        """With an identity model, averaging de-augmented preds gives back input."""
        result = apply_tta(identity_model, batch, augmentations=augmentations)
        assert torch.allclose(result, batch, atol=1e-6), (
            f"Identity-model TTA failed for {augmentations}: "
            f"max diff = {(result - batch).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# apply_tta: skip_keys (frame-field crossfield scenario)
# ---------------------------------------------------------------------------


class TestApplyTTASkipKeys:
    def test_skipped_key_comes_from_identity_pass(self, batch):
        """A key in skip_keys must come unchanged from the identity (rot0) pass."""
        call_log = []

        def dict_model(x):
            call_log.append(x.clone())
            return {"seg": x, "crossfield": x * 10}

        augmentations = [ROT0, ROT90, ROT180, ROT270]
        result = apply_tta(
            dict_model,
            batch,
            augmentations=augmentations,
            skip_keys=frozenset({"crossfield"}),
        )

        # crossfield must be the raw output of the rot0 pass (x * 10 for batch)
        expected_crossfield = batch * 10
        assert torch.allclose(
            result["crossfield"], expected_crossfield, atol=1e-6
        ), "crossfield should equal the identity-pass output (batch * 10)"

    def test_non_skipped_key_is_averaged(self, batch, identity_model):
        """Non-skipped dict keys must still be properly de-augmented and averaged."""

        def dict_model(x):
            return {"seg": x, "crossfield": x * 0}

        result = apply_tta(
            dict_model,
            batch,
            augmentations=ROTATION_AUGMENTATIONS,
            skip_keys=frozenset({"crossfield"}),
        )
        # seg: identity model → result should equal batch
        assert torch.allclose(result["seg"], batch, atol=1e-6)

    def test_skip_keys_without_rot0_uses_first_pass(self, batch):
        """When rot0 is absent, skipped keys come from the first augmentation's pass."""
        call_order = []

        def dict_model(x):
            # Return a unique value per call to track which pass is used
            val = torch.full_like(x, float(len(call_order)))
            call_order.append(val.clone())
            return {"seg": x, "crossfield": val}

        augmentations = [ROT90, ROT180]  # no ROT0
        result = apply_tta(
            dict_model,
            batch,
            augmentations=augmentations,
            skip_keys=frozenset({"crossfield"}),
        )
        # crossfield should be from the first pass (ROT90 pass, call_order[0])
        assert torch.allclose(result["crossfield"], call_order[0])


# ---------------------------------------------------------------------------
# apply_tta: subset augmentations
# ---------------------------------------------------------------------------


class TestApplyTTASubsets:
    @pytest.mark.parametrize(
        "augmentations",
        [
            [ROT0, ROT180],
            [FLIP_H, FLIP_V],
            [ROT90, ROT270],
            [ROT0, FLIP_H],
            [ROT0, ROT90, ROT180, ROT270, FLIP_H, FLIP_V],
        ],
    )
    def test_subset_preserves_shape(self, batch, identity_model, augmentations):
        result = apply_tta(identity_model, batch, augmentations=augmentations)
        assert result.shape == batch.shape

    def test_single_augmentation_matches_direct_model_call(self, batch):
        """Single-augment TTA with a non-identity aug must equal deaugment(model(aug(x)))."""
        torch.manual_seed(0)
        weight = torch.randn(C, C, 1, 1)
        bias = torch.zeros(C)

        def linear_model(x):
            return torch.nn.functional.conv2d(x, weight, bias, padding=0)

        for aug in [ROT90, ROT180, ROT270, FLIP_H, FLIP_V]:
            augmented = _augment(batch, aug)
            expected = _deaugment(linear_model(augmented), aug)
            result = apply_tta(linear_model, batch, augmentations=[aug])
            assert torch.allclose(
                result, expected, atol=1e-5
            ), f"Single-aug TTA mismatch for '{aug}'"
