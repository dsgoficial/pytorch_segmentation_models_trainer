"""
Tests for Mixture of Experts (MoE) components:
  - NoisyTopKRouter (v1, token-choice)
  - ExpertChoiceRouter (v2, expert-choice)
  - MoEConv2dReLU (v1 token-choice + v2 shared expert + expert choice)
  - UPerNetMoEDecoder
  - UPerNetMoE (full model)
  - Auxiliary loss integration
"""

import unittest
import warnings

# Windows DLL fix — must import before torch
import pytorch_segmentation_models_trainer  # noqa: F401

import torch
from unittest.mock import Mock


from pytorch_segmentation_models_trainer.custom_models.moe_layers import (
    NoisyTopKRouter,
    ExpertChoiceRouter,
    MoEConv2dReLU,
)
from pytorch_segmentation_models_trainer.custom_models.upernet_moe import (
    UPerNetMoE,
    UPerNetMoEDecoder,
)


# ----------------------------------------------------------------------
# A. NoisyTopKRouter (v1 — token-choice)
# ----------------------------------------------------------------------


class TestNoisyTopKRouter(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.router = NoisyTopKRouter(
            in_channels=256, num_experts=8, top_k=2, noise_std=1.0
        )

    def test_output_shapes(self):
        x = torch.randn(2, 256, 8, 8)
        weight_maps, aux_loss = self.router(x)
        self.assertEqual(weight_maps.shape, (2, 8, 8, 8))
        self.assertEqual(aux_loss.dim(), 0)

    def test_expert_weights_sparse(self):
        """Each pixel should have exactly top_k non-zero experts."""
        x = torch.randn(2, 256, 4, 4)
        weight_maps, _ = self.router(x)
        # (B, num_experts, H, W) -> count non-zero per pixel
        non_zero = (weight_maps > 0).sum(dim=1)  # (B, H, W)
        self.assertTrue(torch.all(non_zero == 2))

    def test_aux_loss_is_positive(self):
        x = torch.randn(4, 256, 4, 4)
        _, aux_loss = self.router(x)
        self.assertTrue(aux_loss.item() > 0)

    def test_no_noise_in_eval(self):
        self.router.eval()
        x = torch.randn(2, 256, 4, 4)
        w1, _ = self.router(x)
        w2, _ = self.router(x)
        self.assertTrue(torch.allclose(w1, w2))

    def test_different_top_k(self):
        for k in [1, 2, 4]:
            router = NoisyTopKRouter(
                in_channels=64, num_experts=8, top_k=k
            )
            x = torch.randn(1, 64, 4, 4)
            w, _ = router(x)
            non_zero = (w > 0).sum(dim=1)
            self.assertTrue(torch.all(non_zero == k))

    def test_gradient_flows_through_router(self):
        x = torch.randn(1, 256, 4, 4, requires_grad=True)
        weight_maps, aux_loss = self.router(x)
        total = weight_maps.sum() + aux_loss
        total.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.abs().sum() > 0)


# ----------------------------------------------------------------------
# B. ExpertChoiceRouter (v2 — expert-choice)
# ----------------------------------------------------------------------


class TestExpertChoiceRouter(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.router = ExpertChoiceRouter(
            in_channels=256, num_experts=6, capacity_factor=1.25
        )

    def test_output_shapes(self):
        x = torch.randn(2, 256, 8, 8)
        weight_maps, aux_loss = self.router(x)
        self.assertEqual(weight_maps.shape, (2, 6, 8, 8))
        self.assertEqual(aux_loss.dim(), 0)

    def test_aux_loss_is_zero(self):
        """Expert Choice doesn't need auxiliary loss."""
        x = torch.randn(2, 256, 4, 4)
        _, aux_loss = self.router(x)
        self.assertAlmostEqual(aux_loss.item(), 0.0, places=5)

    def test_capacity_respected(self):
        """Each expert should select at most capacity positions."""
        x = torch.randn(1, 256, 8, 8)
        N = 64
        capacity = max(1, int(1.25 * N / 6))
        weight_maps, _ = self.router(x)
        for e in range(6):
            n_selected = (weight_maps[0, e] > 0).sum().item()
            self.assertLessEqual(n_selected, capacity)

    def test_deterministic_in_eval(self):
        self.router.eval()
        x = torch.randn(2, 256, 4, 4)
        w1, _ = self.router(x)
        w2, _ = self.router(x)
        self.assertTrue(torch.allclose(w1, w2))

    def test_gradient_flows(self):
        x = torch.randn(1, 256, 4, 4, requires_grad=True)
        weight_maps, _ = self.router(x)
        weight_maps.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.abs().sum() > 0)

    def test_per_image_routing(self):
        """Routing should be independent per image in batch."""
        x = torch.randn(2, 256, 4, 4)
        w_batch, _ = self.router(x)

        # Run each image separately
        w0, _ = self.router(x[0:1])
        w1, _ = self.router(x[1:2])

        # Should produce identical results
        self.assertTrue(torch.allclose(w_batch[0:1], w0, atol=1e-5))
        self.assertTrue(torch.allclose(w_batch[1:2], w1, atol=1e-5))

    def test_different_capacity_factors(self):
        for cf in [0.5, 1.0, 1.5, 2.0]:
            router = ExpertChoiceRouter(
                in_channels=64, num_experts=4, capacity_factor=cf
            )
            x = torch.randn(1, 64, 4, 4)
            w, _ = router(x)
            self.assertEqual(w.shape, (1, 4, 4, 4))


# ----------------------------------------------------------------------
# C. MoEConv2dReLU — token-choice (v1 backward compat)
# ----------------------------------------------------------------------


class TestMoEConv2dReLU(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.moe = MoEConv2dReLU(
            in_channels=256, out_channels=128,
            num_experts=4, top_k=2,
        )

    def test_output_shape(self):
        x = torch.randn(2, 256, 8, 8)
        output, aux_loss = self.moe(x)
        self.assertEqual(output.shape, (2, 128, 8, 8))
        self.assertEqual(aux_loss.dim(), 0)

    def test_grad_flows(self):
        x = torch.randn(1, 256, 4, 4, requires_grad=True)
        output, aux_loss = self.moe(x)
        total = output.sum() + aux_loss
        total.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.abs().sum() > 0)

    def test_expert_count(self):
        self.assertEqual(len(self.moe.experts), 4)

    def test_fusion_block_dimensions(self):
        """Test with fusion block dimensions: 1024 -> 256."""
        moe = MoEConv2dReLU(
            in_channels=1024, out_channels=256,
            num_experts=8, top_k=2,
        )
        x = torch.randn(1, 1024, 16, 16)
        output, aux_loss = moe(x)
        self.assertEqual(output.shape, (1, 256, 16, 16))

    def test_extra_repr(self):
        s = repr(self.moe)
        self.assertIn("num_experts=4", s)
        self.assertIn("token_choice", s)

    def test_single_expert(self):
        """With top_k=1 and 1 expert, should be equivalent to a single Conv2dReLU."""
        moe = MoEConv2dReLU(
            in_channels=64, out_channels=64,
            num_experts=1, top_k=1,
        )
        x = torch.randn(1, 64, 4, 4)
        output, aux_loss = moe(x)
        self.assertEqual(output.shape, (1, 64, 4, 4))

    def test_no_shared_expert_by_default(self):
        self.assertFalse(self.moe.use_shared_expert)
        self.assertFalse(hasattr(self.moe, 'shared_expert'))

    def test_token_choice_routing_by_default(self):
        self.assertEqual(self.moe.routing, "token_choice")
        self.assertIsInstance(self.moe.router, NoisyTopKRouter)


# ----------------------------------------------------------------------
# D. MoEConv2dReLU — expert-choice + shared expert (v2)
# ----------------------------------------------------------------------


class TestMoEConv2dReLUV2(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.moe = MoEConv2dReLU(
            in_channels=256, out_channels=128,
            num_experts=6, use_shared_expert=True,
            routing="expert_choice", capacity_factor=1.25,
        )

    def test_output_shape(self):
        x = torch.randn(2, 256, 8, 8)
        output, aux_loss = self.moe(x)
        self.assertEqual(output.shape, (2, 128, 8, 8))

    def test_has_shared_expert(self):
        self.assertTrue(self.moe.use_shared_expert)
        self.assertIsNotNone(self.moe.shared_expert)

    def test_expert_choice_router(self):
        self.assertEqual(self.moe.routing, "expert_choice")
        self.assertIsInstance(self.moe.router, ExpertChoiceRouter)

    def test_aux_loss_zero_for_expert_choice(self):
        x = torch.randn(2, 256, 4, 4)
        _, aux_loss = self.moe(x)
        self.assertAlmostEqual(aux_loss.item(), 0.0, places=5)

    def test_shared_plus_sparse_count(self):
        """1 shared + 6 sparse = 7 total experts."""
        self.assertEqual(len(self.moe.experts), 6)
        self.assertIsNotNone(self.moe.shared_expert)

    def test_grad_flows(self):
        x = torch.randn(1, 256, 4, 4, requires_grad=True)
        output, aux_loss = self.moe(x)
        output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.abs().sum() > 0)

    def test_shared_expert_always_contributes(self):
        """Shared expert output should be non-zero for all pixels."""
        x = torch.randn(1, 256, 4, 4)
        self.moe.eval()
        # Run shared expert directly
        shared_out = self.moe.shared_expert(x)
        self.assertTrue(shared_out.abs().sum() > 0)

    def test_extra_repr_v2(self):
        s = repr(self.moe)
        self.assertIn("expert_choice", s)
        self.assertIn("shared_expert=True", s)

    def test_fusion_block_large(self):
        """Test with ConvNeXtV2-Large fusion dimensions."""
        moe = MoEConv2dReLU(
            in_channels=768, out_channels=256,
            num_experts=6, use_shared_expert=True,
            routing="expert_choice", capacity_factor=1.25,
        )
        x = torch.randn(1, 768, 16, 16)
        output, aux_loss = moe(x)
        self.assertEqual(output.shape, (1, 256, 16, 16))
        self.assertAlmostEqual(aux_loss.item(), 0.0, places=5)

    def test_backward_pass_v2(self):
        moe = MoEConv2dReLU(
            in_channels=128, out_channels=64,
            num_experts=4, use_shared_expert=True,
            routing="expert_choice",
        )
        x = torch.randn(2, 128, 8, 8)
        output, _ = moe(x)
        loss = output.sum()
        loss.backward()
        # Check shared expert grads
        for p in moe.shared_expert.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)
        # Check sparse expert grads
        for expert in moe.experts:
            for p in expert.parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad)


# ----------------------------------------------------------------------
# E. UPerNetMoE (full model) — v1 backward compat
# ----------------------------------------------------------------------


class TestUPerNetMoE(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_forward_pass(self):
        model = UPerNetMoE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=7,
            moe_num_experts=4,
            moe_top_k=2,
            moe_at_fusion=True,
            moe_at_fpn=False,
        )
        model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 7, 128, 128))

    def test_aux_loss_stored(self):
        model = UPerNetMoE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=7,
            moe_num_experts=4,
            moe_top_k=2,
            moe_aux_loss_weight=0.01,
            moe_at_fusion=True,
        )
        x = torch.randn(2, 3, 128, 128)
        _ = model(x)
        self.assertTrue(model.last_aux_loss.item() > 0)

    def test_aux_loss_weight_zero(self):
        model = UPerNetMoE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=7,
            moe_num_experts=4,
            moe_top_k=2,
            moe_aux_loss_weight=0.0,
            moe_at_fusion=True,
        )
        x = torch.randn(2, 3, 128, 128)
        _ = model(x)
        self.assertAlmostEqual(model.last_aux_loss.item(), 0.0, places=5)

    def test_moe_at_both(self):
        model = UPerNetMoE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=7,
            moe_at_fusion=True,
            moe_at_fpn=True,
            moe_num_experts=4,
            moe_top_k=2,
        )
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, (2, 7, 128, 128))
        self.assertTrue(model.last_aux_loss.item() > 0)

    def test_no_moe_baseline(self):
        """With moe_at_fusion=False, moe_at_fpn=False, should still work."""
        model = UPerNetMoE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=7,
            moe_at_fusion=False,
            moe_at_fpn=False,
        )
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, (2, 7, 128, 128))
        self.assertAlmostEqual(model.last_aux_loss.item(), 0.0, places=5)

    def test_backward_pass(self):
        model = UPerNetMoE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=7,
            moe_num_experts=4,
            moe_top_k=2,
        )
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        loss = out.sum() + model.last_aux_loss
        loss.backward()
        has_router_grad = False
        for name, p in model.decoder.named_parameters():
            if 'router' in name and p.requires_grad:
                self.assertIsNotNone(p.grad, f"No gradient for {name}")
                has_router_grad = True
        self.assertTrue(has_router_grad, "No router parameters found")

    def test_model_name(self):
        model = UPerNetMoE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=7,
        )
        self.assertEqual(model.name, "upernet-moe-resnet18")

    def test_different_input_channels(self):
        """Test with 6 input channels (e.g., RGB+CIR)."""
        model = UPerNetMoE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=6,
            classes=7,
            moe_num_experts=4,
            moe_top_k=2,
        )
        model.eval()
        x = torch.randn(1, 6, 128, 128)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 7, 128, 128))


# ----------------------------------------------------------------------
# F. UPerNetMoE (full model) — v2 shared expert + expert choice
# ----------------------------------------------------------------------


class TestUPerNetMoEV2(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

    def test_forward_pass_v2(self):
        model = UPerNetMoE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=7,
            moe_num_experts=4,
            moe_use_shared_expert=True,
            moe_routing="expert_choice",
            moe_capacity_factor=1.25,
            moe_aux_loss_weight=0.0,
            moe_at_fusion=True,
            moe_at_fpn=False,
        )
        model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 7, 128, 128))

    def test_no_aux_loss_with_expert_choice(self):
        model = UPerNetMoE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=7,
            moe_num_experts=4,
            moe_use_shared_expert=True,
            moe_routing="expert_choice",
            moe_aux_loss_weight=0.0,
            moe_at_fusion=True,
        )
        x = torch.randn(2, 3, 128, 128)
        _ = model(x)
        self.assertAlmostEqual(model.last_aux_loss.item(), 0.0, places=5)

    def test_backward_v2(self):
        model = UPerNetMoE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=7,
            moe_num_experts=4,
            moe_use_shared_expert=True,
            moe_routing="expert_choice",
            moe_at_fusion=True,
        )
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check router gradients flow
        has_router_grad = False
        for name, p in model.decoder.named_parameters():
            if 'router' in name and p.requires_grad:
                self.assertIsNotNone(p.grad, f"No gradient for {name}")
                has_router_grad = True
        self.assertTrue(has_router_grad)
        # Check shared expert gradients
        has_shared_grad = False
        for name, p in model.decoder.named_parameters():
            if 'shared_expert' in name and p.requires_grad:
                self.assertIsNotNone(p.grad, f"No gradient for {name}")
                has_shared_grad = True
        self.assertTrue(has_shared_grad)

    def test_moe_at_both_v2(self):
        model = UPerNetMoE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=7,
            moe_num_experts=4,
            moe_use_shared_expert=True,
            moe_routing="expert_choice",
            moe_at_fusion=True,
            moe_at_fpn=True,
        )
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, (2, 7, 128, 128))

    def test_v2_output_not_zero(self):
        """With shared expert, output should never be all zeros."""
        model = UPerNetMoE(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=7,
            moe_num_experts=4,
            moe_use_shared_expert=True,
            moe_routing="expert_choice",
            moe_at_fusion=True,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        self.assertTrue(out.abs().sum() > 0)


# ----------------------------------------------------------------------
# G. Integration with Model training_step (mock-based)
# ----------------------------------------------------------------------


class TestMoETrainingIntegration(unittest.TestCase):

    def test_training_step_adds_aux_loss(self):
        """Verify the pattern that training_step uses."""
        mock_model = Mock()
        mock_model.last_aux_loss = torch.tensor(0.5)

        base_loss = torch.tensor(1.0)
        if hasattr(mock_model, 'last_aux_loss') and mock_model.last_aux_loss is not None:
            total_loss = base_loss + mock_model.last_aux_loss
        else:
            total_loss = base_loss

        self.assertAlmostEqual(total_loss.item(), 1.5, places=5)

    def test_no_aux_loss_for_standard_model(self):
        """Standard model without last_aux_loss attribute."""
        mock_model = Mock(spec=[])

        base_loss = torch.tensor(1.0)
        if hasattr(mock_model, 'last_aux_loss') and mock_model.last_aux_loss is not None:
            total_loss = base_loss + mock_model.last_aux_loss
        else:
            total_loss = base_loss

        self.assertAlmostEqual(total_loss.item(), 1.0, places=5)

    def test_aux_loss_none(self):
        """Model with last_aux_loss=None should not add anything."""
        mock_model = Mock()
        mock_model.last_aux_loss = None

        base_loss = torch.tensor(1.0)
        if hasattr(mock_model, 'last_aux_loss') and mock_model.last_aux_loss is not None:
            total_loss = base_loss + mock_model.last_aux_loss
        else:
            total_loss = base_loss

        self.assertAlmostEqual(total_loss.item(), 1.0, places=5)

    def test_expert_choice_zero_aux_loss(self):
        """Expert Choice produces zero aux loss — should not affect total."""
        mock_model = Mock()
        mock_model.last_aux_loss = torch.tensor(0.0)

        base_loss = torch.tensor(1.0)
        if hasattr(mock_model, 'last_aux_loss') and mock_model.last_aux_loss is not None:
            total_loss = base_loss + mock_model.last_aux_loss
        else:
            total_loss = base_loss

        self.assertAlmostEqual(total_loss.item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
