# -*- coding: utf-8 -*-
"""Tests for the mother → children class hierarchy used by GFSS methods."""

import pytest
import torch
from omegaconf import OmegaConf

from pytorch_segmentation_models_trainer.few_shot.hierarchy import ClassHierarchy


class TestConstruction:
    def test_single_mother(self):
        h = ClassHierarchy({3: [3, 5]}, num_base_classes=5)
        assert h.num_base_classes == 5
        assert h.num_classes == 6
        assert h.novel_classes == [5]
        assert h.mothers == [3]
        assert h.children_of(3) == [3, 5]

    def test_multiple_mothers(self):
        h = ClassHierarchy({3: [3, 5], 4: [4, 6, 7]}, num_base_classes=5)
        assert h.num_classes == 8
        assert h.novel_classes == [5, 6, 7]
        assert h.mothers == [3, 4]

    def test_background_mother_reproduces_standard_gfss(self):
        h = ClassHierarchy({0: [0, 3, 4]}, num_base_classes=3)
        assert h.novel_classes == [3, 4]
        assert h.mother_of(3) == 0
        assert h.mother_of(4) == 0

    def test_accepts_dictconfig_and_string_keys(self):
        cfg = OmegaConf.create({"h": {"3": [3, 5]}})
        h = ClassHierarchy(cfg.h, num_base_classes=5)
        assert h.mother_of(5) == 3

    def test_mother_of_base_class_is_itself(self):
        h = ClassHierarchy({3: [3, 5]}, num_base_classes=5)
        for c in range(5):
            assert h.mother_of(c) == c

    def test_new_to_old_lut(self):
        h = ClassHierarchy({3: [3, 5], 1: [1, 6]}, num_base_classes=5)
        assert h.new_to_old.tolist() == [0, 1, 2, 3, 4, 3, 1]
        assert h.new_to_old.dtype == torch.long


class TestValidation:
    def test_empty_mapping_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ClassHierarchy({}, num_base_classes=5)

    def test_mother_out_of_range_raises(self):
        with pytest.raises(ValueError, match="mother"):
            ClassHierarchy({7: [7, 8]}, num_base_classes=5)

    def test_mother_must_be_one_of_its_children(self):
        with pytest.raises(ValueError, match="must be one of its children"):
            ClassHierarchy({3: [5, 6]}, num_base_classes=5)

    def test_child_shared_by_two_mothers_raises(self):
        with pytest.raises(ValueError, match="more than one mother"):
            ClassHierarchy({3: [3, 5], 4: [4, 5]}, num_base_classes=5)

    def test_base_class_as_child_of_other_mother_raises(self):
        with pytest.raises(ValueError, match="base class"):
            ClassHierarchy({3: [3, 2, 5]}, num_base_classes=5)

    def test_non_contiguous_novel_indices_raise(self):
        with pytest.raises(ValueError, match="contiguous"):
            ClassHierarchy({3: [3, 6]}, num_base_classes=5)

    def test_mother_without_novel_child_raises(self):
        with pytest.raises(ValueError, match="no novel"):
            ClassHierarchy({3: [3]}, num_base_classes=5)


class TestProjectNewToOld:
    def test_sums_children_into_mother(self):
        h = ClassHierarchy({3: [3, 5]}, num_base_classes=5)
        p = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.1, 0.4]).view(1, 6, 1, 1)
        out = h.project_new_to_old(p)
        assert out.shape == (1, 5, 1, 1)
        torch.testing.assert_close(
            out.flatten(), torch.tensor([0.1, 0.1, 0.1, 0.6, 0.1])
        )

    def test_multiple_mothers_and_mass_preserved(self):
        h = ClassHierarchy({3: [3, 5], 0: [0, 6]}, num_base_classes=5)
        p = torch.softmax(torch.randn(2, 7, 4, 4), dim=1)
        out = h.project_new_to_old(p)
        assert out.shape == (2, 5, 4, 4)
        torch.testing.assert_close(out.sum(1), torch.ones(2, 4, 4))
        torch.testing.assert_close(out[:, 3], p[:, 3] + p[:, 5])
        torch.testing.assert_close(out[:, 0], p[:, 0] + p[:, 6])

    def test_supports_extra_leading_dims(self):
        h = ClassHierarchy({3: [3, 5]}, num_base_classes=5)
        p = torch.softmax(torch.randn(2, 3, 6, 4, 4), dim=2)
        out = h.project_new_to_old(p, dim=2)
        assert out.shape == (2, 3, 5, 4, 4)

    def test_gradient_flows(self):
        h = ClassHierarchy({3: [3, 5]}, num_base_classes=5)
        logits = torch.randn(1, 6, 2, 2, requires_grad=True)
        h.project_new_to_old(torch.softmax(logits, 1)).sum().backward()
        assert logits.grad is not None

    def test_wrong_channel_count_raises(self):
        h = ClassHierarchy({3: [3, 5]}, num_base_classes=5)
        with pytest.raises(ValueError, match="6 channels"):
            h.project_new_to_old(torch.zeros(1, 5, 2, 2))


class TestMapOldTargetToMother:
    def test_maps_novel_labels_to_mothers_keeping_ignore(self):
        h = ClassHierarchy({3: [3, 5], 1: [1, 6]}, num_base_classes=5)
        y = torch.tensor([[0, 5, 6, 255, 3]])
        assert h.to_old_labels(y).tolist() == [[0, 3, 1, 255, 3]]


def test_repr_mentions_mapping():
    h = ClassHierarchy({3: [3, 5]}, num_base_classes=5)
    assert "3: [3, 5]" in repr(h) and "num_base_classes=5" in repr(h)
