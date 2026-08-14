"""
Tests for MEDOE (Multi-Expert Decoder and Output Ensemble):
  - ExpertBranch output shapes
  - UPerNetMEDOE forward pass, gate, ensemble
  - Masked target generation
  - Expert loss computation (normal, all-ignore, single-class)
  - Diagnostics
  - Compatibility with MixStyleCallback (encoder accessible)
"""

import unittest
import warnings

# Windows DLL fix — must import before torch
import pytorch_segmentation_models_trainer  # noqa: F401

import torch

from pytorch_segmentation_models_trainer.custom_models.upernet_medoe import (
    ExpertBranch,
    UPerNetMEDOE,
)


class TestExpertBranch(unittest.TestCase):

    def test_output_shape(self):
        branch = ExpertBranch(256, 256, num_classes=6, upsampling=4)
        x = torch.randn(2, 256, 16, 16)
        out = branch(x)
        self.assertEqual(out.shape, (2, 6, 64, 64))

    def test_output_shape_no_upsample(self):
        branch = ExpertBranch(128, 128, num_classes=3, upsampling=1)
        x = torch.randn(1, 128, 8, 8)
        out = branch(x)
        self.assertEqual(out.shape, (1, 3, 8, 8))

    def test_grad_flows(self):
        branch = ExpertBranch(64, 64, num_classes=6)
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        out = branch(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.abs().sum() > 0)


class TestUPerNetMEDOE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.model = UPerNetMEDOE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=6,
            class_groups={
                "HEAD": [0, 1, 2, 3, 4, 5],
                "BODY": [2, 3, 5],
                "TAIL": [2],
            },
            expert_loss_weight=0.5,
        )

    def test_forward_shape(self):
        self.model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (2, 6, 128, 128))

    def test_num_experts(self):
        self.assertEqual(self.model.num_experts, 3)
        self.assertEqual(len(self.model.experts), 3)
        self.assertEqual(self.model.group_names, ["HEAD", "BODY", "TAIL"])

    def test_encoder_accessible(self):
        """MixStyleCallback accesses pl_module.model.encoder."""
        self.assertTrue(hasattr(self.model, "encoder"))

    def test_gate_weights_sum_to_one(self):
        self.model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            _ = self.model(x)
        gw = self.model.last_gate_weights
        self.assertEqual(gw.shape[1], 3)
        sums = gw.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_expert_outputs_stored(self):
        self.model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            _ = self.model(x)
        self.assertIsNotNone(self.model.last_expert_outputs)
        self.assertEqual(len(self.model.last_expert_outputs), 3)
        for eo in self.model.last_expert_outputs:
            self.assertEqual(eo.shape, (2, 6, 128, 128))

    def test_backward_pass(self):
        self.model.train()
        x = torch.randn(2, 3, 128, 128)
        out = self.model(x)
        loss = out.sum()
        loss.backward()
        # Gate grads
        self.assertIsNotNone(self.model.gate.weight.grad)
        # Expert grads
        for expert in self.model.experts:
            for p in expert.parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad)
                    break

    def test_model_name(self):
        self.assertIn("upernet-medoe", self.model.name)


class TestMaskedTarget(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.model = UPerNetMEDOE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=6,
            class_groups={
                "HEAD": [0, 1, 2, 3, 4, 5],
                "BODY": [2, 3, 5],
                "TAIL": [2],
            },
        )

    def test_head_mask_cumulative(self):
        """HEAD sees all classes — nothing should be masked."""
        target = torch.tensor([[0, 1, 2, 3, 4, 5]])
        masked = self.model._make_masked_target(target, 0)
        for j in range(6):
            self.assertEqual(masked[0, j].item(), j)

    def test_body_mask_cumulative(self):
        """BODY sees [2,3,5] — classes 0,1,4 should be masked."""
        target = torch.tensor([[0, 1, 2, 3, 4, 5]])
        masked = self.model._make_masked_target(target, 1)
        self.assertEqual(masked[0, 2].item(), 2)
        self.assertEqual(masked[0, 3].item(), 3)
        self.assertEqual(masked[0, 5].item(), 5)
        for j in [0, 1, 4]:
            self.assertEqual(masked[0, j].item(), 255)

    def test_tail_mask(self):
        """TAIL sees only [2] — all others masked."""
        target = torch.tensor([[0, 1, 2, 3, 4, 5]])
        masked = self.model._make_masked_target(target, 2)
        self.assertEqual(masked[0, 2].item(), 2)
        for j in [0, 1, 3, 4, 5]:
            self.assertEqual(masked[0, j].item(), 255)

    def test_preserves_existing_ignore(self):
        target = torch.tensor([[255, 4, 2]])
        # BODY: [2,3,5] — test that 255 stays 255, class 4 is masked, class 2 kept
        masked = self.model._make_masked_target(target, 1)  # BODY
        self.assertEqual(masked[0, 0].item(), 255)  # was already 255
        self.assertEqual(masked[0, 1].item(), 255)  # class 4 not in BODY
        self.assertEqual(masked[0, 2].item(), 2)  # class 2 in BODY

    def test_all_ignore_input(self):
        target = torch.full((1, 4), 255, dtype=torch.long)
        masked = self.model._make_masked_target(target, 0)
        self.assertTrue((masked == 255).all())


class TestExpertLoss(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.model = UPerNetMEDOE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=6,
            class_groups={
                "HEAD": [0, 1, 2, 3, 4, 5],
                "BODY": [2, 3, 5],
                "TAIL": [2],
            },
            expert_loss_weight=0.5,
        )

    def test_normal_loss(self):
        self.model.train()
        x = torch.randn(2, 3, 128, 128)
        _ = self.model(x)
        target = torch.randint(0, 6, (2, 128, 128))
        loss = self.model.compute_expert_loss(self.model.last_expert_outputs, target)
        self.assertGreater(loss.item(), 0)

    def test_all_ignore_ce_zero_suppression_nonzero(self):
        """All-ignore target: CE term=0 but suppression loss may be >0."""
        self.model.train()
        x = torch.randn(2, 3, 128, 128)
        _ = self.model(x)
        target = torch.full((2, 128, 128), 255, dtype=torch.long)
        loss = self.model.compute_expert_loss(self.model.last_expert_outputs, target)
        # Loss can be >0 due to L2 suppression on non-assigned class logits
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_single_class_only_tail(self):
        """Only terreno_exposto (class 2) pixels — only TAIL expert should contribute."""
        self.model.train()
        x = torch.randn(2, 3, 128, 128)
        _ = self.model(x)
        target = torch.full((2, 128, 128), 255, dtype=torch.long)
        target[:, :32, :] = 2  # only terreno_exposto
        loss = self.model.compute_expert_loss(self.model.last_expert_outputs, target)
        self.assertGreater(loss.item(), 0)

    def test_suppression_loss_nonzero_for_tail(self):
        """TAIL expert (class [2]) should have non-zero L2 suppression on other logits."""
        self.model.train()
        x = torch.randn(2, 3, 128, 128)
        _ = self.model(x)
        supp = self.model._compute_suppression_loss(
            self.model.last_expert_outputs[2], 2  # TAIL expert
        )
        self.assertGreater(supp.item(), 0)

    def test_suppression_loss_zero_for_head(self):
        """HEAD expert sees all classes — no suppression needed."""
        self.model.train()
        x = torch.randn(2, 3, 128, 128)
        _ = self.model(x)
        supp = self.model._compute_suppression_loss(
            self.model.last_expert_outputs[0], 0  # HEAD expert
        )
        self.assertEqual(supp.item(), 0.0)

    def test_grad_flows_through_expert_loss(self):
        self.model.train()
        self.model.zero_grad()
        x = torch.randn(2, 3, 128, 128)
        _ = self.model(x)
        target = torch.randint(0, 6, (2, 128, 128))
        loss = self.model.compute_expert_loss(self.model.last_expert_outputs, target)
        loss.backward()
        # Expert branch should have grads
        has_grad = False
        for expert in self.model.experts:
            for p in expert.parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    has_grad = True
                    break
        self.assertTrue(has_grad)


class TestMEDOEDiagnostics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.model = UPerNetMEDOE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=6,
            class_groups={
                "HEAD": [0, 1, 2, 3, 4, 5],
                "BODY": [2, 3, 5],
                "TAIL": [2],
            },
        )

    def test_basic_diagnostics(self):
        self.model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            _ = self.model(x)
        diag = self.model.get_medoe_diagnostics()
        self.assertIn("medoe/gate_HEAD_mean", diag)
        self.assertIn("medoe/gate_BODY_mean", diag)
        self.assertIn("medoe/gate_TAIL_mean", diag)
        self.assertIn("medoe/gate_entropy", diag)

    def test_diagnostics_with_masks(self):
        self.model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            _ = self.model(x)
        masks = torch.randint(0, 6, (2, 128, 128))
        diag = self.model.get_medoe_diagnostics(hard_masks=masks)
        # HEAD classes: 4, 0, 1
        self.assertIn("medoe/expert_HEAD_iou/class_4", diag)
        self.assertIn("medoe/expert_HEAD_iou/class_0", diag)
        self.assertIn("medoe/expert_HEAD_iou/class_1", diag)
        # TAIL class: 2
        self.assertIn("medoe/expert_TAIL_iou/class_2", diag)

    def test_diagnostics_empty_without_forward(self):
        model = UPerNetMEDOE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=6,
        )
        diag = model.get_medoe_diagnostics()
        self.assertEqual(len(diag), 0)

    def test_gate_weights_detached(self):
        self.model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            _ = self.model(x)
        self.assertFalse(self.model.last_gate_weights.requires_grad)


if __name__ == "__main__":
    unittest.main()
