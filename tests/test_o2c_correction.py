# -*- coding: utf-8 -*-
"""Tests for online_object_correction (O2C, AIO2 §2.4, multiclass generalization)."""

import numpy as np
import pytest
import torch

from pytorch_segmentation_models_trainer.utils.o2c_correction import (
    online_object_correction,
)

# Classes used throughout: 0=background/other, 1=grassland, 2=cropland, 3=urban(fixed)
H, W = 6, 6


def _probs_from_argmax(
    argmax_grid: np.ndarray, n_classes: int, confident: float = 0.9
) -> torch.Tensor:
    """Build a (C, H, W) softmax-like prob tensor peaked at argmax_grid's class per pixel."""
    c_h_w = np.zeros((n_classes, *argmax_grid.shape), dtype=np.float32)
    remainder = (1.0 - confident) / max(n_classes - 1, 1)
    c_h_w[:] = remainder
    for c in range(n_classes):
        c_h_w[c][argmax_grid == c] = confident
    return torch.from_numpy(c_h_w)


class TestValidation:
    def test_invalid_conflict_resolution_raises(self):
        noisy = torch.zeros(H, W, dtype=torch.long)
        probs = torch.ones(4, H, W) / 4
        with pytest.raises(ValueError):
            online_object_correction(noisy, probs, [1, 2], conflict_resolution="bogus")

    def test_even_filter_size_raises(self):
        noisy = torch.zeros(H, W, dtype=torch.long)
        probs = torch.ones(4, H, W) / 4
        with pytest.raises(ValueError):
            online_object_correction(noisy, probs, [1, 2], filter_size=4)

    def test_empty_classes_to_correct_raises(self):
        noisy = torch.zeros(H, W, dtype=torch.long)
        probs = torch.ones(4, H, W) / 4
        with pytest.raises(ValueError):
            online_object_correction(noisy, probs, [])

    def test_duplicate_classes_to_correct_raises(self):
        noisy = torch.zeros(H, W, dtype=torch.long)
        probs = torch.ones(4, H, W) / 4
        with pytest.raises(ValueError):
            online_object_correction(noisy, probs, [1, 1])

    def test_out_of_range_class_raises(self):
        noisy = torch.zeros(H, W, dtype=torch.long)
        probs = torch.ones(4, H, W) / 4
        with pytest.raises(ValueError):
            online_object_correction(noisy, probs, [1, 99])


class TestOutputContract:
    def test_output_shape_matches_input(self):
        noisy = torch.zeros(H, W, dtype=torch.long)
        probs = torch.ones(4, H, W) / 4
        out = online_object_correction(noisy, probs, [1, 2])
        assert out.shape == (H, W)

    def test_output_dtype_matches_input(self):
        noisy = torch.zeros(H, W, dtype=torch.int32)
        probs = torch.ones(4, H, W) / 4
        out = online_object_correction(noisy, probs, [1, 2])
        assert out.dtype == torch.int32

    def test_no_proposals_returns_original_unchanged(self):
        noisy = torch.randint(0, 4, (H, W), dtype=torch.long)
        # Teacher agrees with noisy everywhere except never confidently
        # predicts anything above 0.5 for any class -> no proposals at all.
        probs = torch.full((4, H, W), 0.4)
        out = online_object_correction(noisy, probs, [1, 2])
        assert torch.equal(out, noisy)


class TestBasicAddAndDiscard:
    def test_missed_object_gets_added(self):
        noisy = np.zeros((H, W), dtype=np.int64)  # all class 0
        noisy_t = torch.from_numpy(noisy)
        # teacher confidently predicts class 1 in a 3x3 block, no overlap
        # with any existing class-1 pixel in noisy (there are none).
        argmax = np.zeros((H, W), dtype=np.int64)
        argmax[1:4, 1:4] = 1
        probs = _probs_from_argmax(argmax, n_classes=3)

        out = online_object_correction(noisy_t, probs, classes_to_correct=[0, 1])
        out_np = out.numpy()
        assert (out_np[1:4, 1:4] == 1).all()
        # everything outside the block stays class 0
        mask = np.ones((H, W), dtype=bool)
        mask[1:4, 1:4] = False
        assert (out_np[mask] == 0).all()

    def test_object_overlapping_existing_label_is_discarded(self):
        noisy = np.zeros((H, W), dtype=np.int64)
        noisy[1:4, 1:4] = 1  # already labelled class 1 here
        noisy_t = torch.from_numpy(noisy)
        # teacher predicts the SAME block as class 1 -> overlaps -> discarded
        argmax = np.zeros((H, W), dtype=np.int64)
        argmax[1:4, 1:4] = 1
        probs = _probs_from_argmax(argmax, n_classes=3)

        out = online_object_correction(noisy_t, probs, classes_to_correct=[0, 1])
        assert torch.equal(out, noisy_t)  # nothing changes, already marked


class TestFixedClasses:
    """The mechanism this whole module exists to get right (see docstring)."""

    def test_pixel_outside_classes_to_correct_is_never_touched(self):
        noisy = np.zeros((H, W), dtype=np.int64)
        noisy[2:4, 2:4] = 3  # class 3 = "urban", NOT in classes_to_correct
        noisy_t = torch.from_numpy(noisy)
        # Teacher confidently proposes class 1 exactly over that urban block
        # (no overlap with any class-1 pixel in noisy, since noisy has none)
        argmax = np.zeros((H, W), dtype=np.int64)
        argmax[2:4, 2:4] = 1
        probs = _probs_from_argmax(argmax, n_classes=4)

        out = online_object_correction(noisy_t, probs, classes_to_correct=[0, 1])
        out_np = out.numpy()
        # Must remain class 3 — protected, regardless of the proposal.
        assert (out_np[2:4, 2:4] == 3).all()

    def test_target_class_outside_classes_to_correct_is_never_proposed(self):
        # classes_to_correct=[0] only -> class 2 is never even considered as
        # a target, even though the teacher is confident about it.
        noisy = np.zeros((H, W), dtype=np.int64)
        noisy_t = torch.from_numpy(noisy)
        argmax = np.zeros((H, W), dtype=np.int64)
        argmax[1:3, 1:3] = 2
        probs = _probs_from_argmax(argmax, n_classes=3)

        out = online_object_correction(noisy_t, probs, classes_to_correct=[0])
        assert torch.equal(out, noisy_t)


class TestConflictResolution:
    def _disputed_setup(self):
        noisy = np.zeros((H, W), dtype=np.int64)
        noisy_t = torch.from_numpy(noisy)
        # Both class 1 and class 2 predicted confidently over the SAME block
        # -> both channels propose the same pixels -> conflict.
        probs = np.full((3, H, W), 0.1, dtype=np.float32)
        probs[1, 1:4, 1:4] = 0.6
        probs[2, 1:4, 1:4] = 0.9  # class 2 more confident there
        return noisy_t, torch.from_numpy(probs)

    def test_priority_order_first_class_wins(self):
        noisy_t, probs = self._disputed_setup()
        # 0 must stay eligible too -> it's the disputed pixels' original
        # class; without it the "fixed classes" restore would wipe the
        # whole disputed region regardless of the 1-vs-2 tie being tested.
        out = online_object_correction(
            noisy_t,
            probs,
            classes_to_correct=[1, 2, 0],
            conflict_resolution="priority_order",
        )
        assert (out.numpy()[1:4, 1:4] == 1).all()

    def test_priority_order_respects_list_order_not_confidence(self):
        noisy_t, probs = self._disputed_setup()
        out = online_object_correction(
            noisy_t,
            probs,
            classes_to_correct=[2, 1, 0],
            conflict_resolution="priority_order",
        )
        # class 2 listed first now -> wins, even though nothing about
        # confidence changed (only list order did).
        assert (out.numpy()[1:4, 1:4] == 2).all()

    def test_teacher_confidence_ignores_list_order(self):
        noisy_t, probs = self._disputed_setup()
        out_a = online_object_correction(
            noisy_t,
            probs,
            classes_to_correct=[1, 2, 0],
            conflict_resolution="teacher_confidence",
        )
        out_b = online_object_correction(
            noisy_t,
            probs,
            classes_to_correct=[2, 1, 0],
            conflict_resolution="teacher_confidence",
        )
        # class 2 has higher probability at the disputed pixels -> wins
        # regardless of list order.
        assert (out_a.numpy()[1:4, 1:4] == 2).all()
        assert (out_b.numpy()[1:4, 1:4] == 2).all()


class TestBoundaryErosion:
    def test_isolated_single_pixel_proposal_is_eroded_away(self):
        noisy = np.zeros((H, W), dtype=np.int64)
        noisy_t = torch.from_numpy(noisy)
        argmax = np.zeros((H, W), dtype=np.int64)
        argmax[3, 3] = 1  # single isolated pixel
        probs = _probs_from_argmax(argmax, n_classes=3)

        out_hard = online_object_correction(noisy_t, probs, [0, 1], filter_size=-1)
        out_eroded = online_object_correction(noisy_t, probs, [0, 1], filter_size=3)

        assert out_hard.numpy()[3, 3] == 1  # kept without erosion
        assert out_eroded.numpy()[3, 3] == 0  # eroded away (too thin to survive)

    def test_large_solid_block_keeps_interior_after_erosion(self):
        noisy = np.zeros((H, W), dtype=np.int64)
        noisy_t = torch.from_numpy(noisy)
        argmax = np.zeros((H, W), dtype=np.int64)
        argmax[1:5, 1:5] = 1  # solid 4x4 block
        probs = _probs_from_argmax(argmax, n_classes=3)

        out = online_object_correction(noisy_t, probs, [0, 1], filter_size=3)
        # interior pixel of the block (not touching any border of the block)
        assert out.numpy()[2, 2] == 1

    def test_filter_size_disabled_by_default(self):
        noisy = np.zeros((H, W), dtype=np.int64)
        noisy_t = torch.from_numpy(noisy)
        argmax = np.zeros((H, W), dtype=np.int64)
        argmax[3, 3] = 1
        probs = _probs_from_argmax(argmax, n_classes=3)

        out = online_object_correction(noisy_t, probs, [0, 1])  # filter_size default
        assert out.numpy()[3, 3] == 1


class TestDeviceAndDtypePreservation:
    def test_cpu_roundtrip(self):
        noisy = torch.zeros(H, W, dtype=torch.long, device="cpu")
        probs = torch.ones(3, H, W) / 3
        out = online_object_correction(noisy, probs, [0, 1])
        assert out.device == noisy.device
