"""
Mixture of Experts (MoE) layers for convolutional segmentation decoders.

Implements:
  - NoisyTopKRouter: token-choice top-k gating (v1, backward compatible)
  - ExpertChoiceRouter: expert-choice routing with guaranteed load balance
  - MoEConv2dReLU: MoE conv block with optional shared expert

v2 improvements:
  - ExpertChoiceRouter: each expert selects its top-C tokens per image,
    guaranteeing perfect load balance without auxiliary loss.
    (Zhou et al., NeurIPS 2022)
  - Shared expert: one dense expert always active on all pixels provides
    a stable baseline; sparse experts add class-specific refinements.
    (Dai et al., DeepSeekMoE 2024)
  - Per-image routing: routing decisions are independent across batch
    elements, ensuring consistent behavior between train and eval.

Backward compatible: default params reproduce v1 behavior.

References:
  - Shazeer et al., "Outrageously Large Neural Networks", ICLR 2017
  - Fedus et al., "Switch Transformers", JMLR 2022
  - Zhou et al., "Mixture-of-Experts with Expert Choice Routing", NeurIPS 2022
  - Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization", 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Any, Dict, Tuple, Union

from segmentation_models_pytorch.base.modules import Conv2dReLU

# ----------------------------------------------------------------------
# Routers
# ----------------------------------------------------------------------


class NoisyTopKRouter(nn.Module):
    """Noisy top-k gating router for spatial MoE (token-choice).

    For each spatial position (pixel), computes gating weights over
    num_experts experts, selects top_k, and returns sparse dispatch
    weights + auxiliary load-balancing loss.

    Args:
        in_channels: Number of input feature channels.
        num_experts: Total number of experts.
        top_k: Number of experts to activate per token.
        noise_std: Standard deviation multiplier for learnable noise
                   added during training for exploration.
    """

    def __init__(
        self,
        in_channels: int,
        num_experts: int = 8,
        top_k: int = 2,
        noise_std: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        self.gate = nn.Linear(in_channels, num_experts, bias=True)
        self.w_noise = nn.Linear(in_channels, num_experts, bias=False)

        nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.w_noise.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (B, C, H, W)

        Returns:
            expert_weight_maps: (B, num_experts, H, W) per-expert gate
                weights. Non-selected experts have zero weight.
            aux_loss: Scalar load-balancing auxiliary loss.
        """
        B, C, H, W = x.shape
        N = H * W

        x_flat = x.permute(0, 2, 3, 1).reshape(B * N, C)

        logits = self.gate(x_flat)

        if self.training and self.noise_std > 0:
            noise_logits = self.w_noise(x_flat)
            noise = torch.randn_like(logits) * F.softplus(noise_logits) * self.noise_std
            logits = logits + noise

        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        dispatch_weights = F.softmax(top_k_logits, dim=-1)

        # Load-balancing auxiliary loss (Switch Transformer style)
        probs = F.softmax(logits, dim=-1)
        expert_mask = torch.zeros_like(probs)
        expert_mask.scatter_(1, top_k_indices, 1.0)
        f = expert_mask.mean(dim=0)
        P = probs.mean(dim=0)
        aux_loss = self.num_experts * (f * P).sum()

        # Build dense weight maps
        full_weights = torch.zeros(
            B * N, self.num_experts, device=x.device, dtype=x.dtype
        )
        full_weights.scatter_(1, top_k_indices, dispatch_weights)

        expert_weight_maps = (
            full_weights.view(B, N, self.num_experts)
            .permute(0, 2, 1)
            .view(B, self.num_experts, H, W)
        )

        return expert_weight_maps, aux_loss


class ExpertChoiceRouter(nn.Module):
    """Expert Choice routing (Zhou et al., NeurIPS 2022).

    Each expert selects its top-C spatial positions per image,
    guaranteeing perfect load balance without auxiliary loss.

    Unlike token-choice (where each pixel picks experts), here each
    expert picks its best pixels. This eliminates load imbalance and
    removes the need for auxiliary loss tuning.

    Routing is per-image (not per-batch) for consistent behavior
    between training and inference.

    Args:
        in_channels: Number of input feature channels.
        num_experts: Total number of sparse experts.
        capacity_factor: Multiplier for expert capacity. 1.0 means each
            expert handles exactly N/num_experts positions per image.
            >1.0 allows overlap (positions selected by multiple experts).
    """

    def __init__(
        self,
        in_channels: int,
        num_experts: int = 6,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        self.gate = nn.Linear(in_channels, num_experts, bias=True)
        nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.gate.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (B, C, H, W)

        Returns:
            expert_weight_maps: (B, num_experts, H, W) per-expert gate
                weights. Non-selected positions have zero weight.
            aux_loss: Always zero (no aux loss needed).
        """
        B, C, H, W = x.shape
        N = H * W

        x_flat = x.permute(0, 2, 3, 1).reshape(B * N, C)
        logits = self.gate(x_flat).view(B, N, self.num_experts)

        # Expert Choice: softmax over spatial positions per image
        # S[b, e, n] = probability that expert e assigns to position n
        S = F.softmax(logits.permute(0, 2, 1), dim=2)  # (B, num_experts, N)

        capacity = max(1, int(self.capacity_factor * N / self.num_experts))
        capacity = min(capacity, N)

        # Vectorized top-C selection across all experts and batch elements
        S_flat = S.reshape(B * self.num_experts, N)
        top_vals, top_idx = torch.topk(S_flat, capacity, dim=1)

        weight_maps = torch.zeros_like(S_flat)
        weight_maps.scatter_(1, top_idx, top_vals)

        expert_weight_maps = weight_maps.view(B, self.num_experts, H, W)

        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return expert_weight_maps, aux_loss


# ----------------------------------------------------------------------
# MoE Convolutional Block
# ----------------------------------------------------------------------


class MoEConv2dReLU(nn.Module):
    """Mixture of Experts convolutional block.

    Drop-in replacement for Conv2dReLU that routes each spatial position
    to a subset of Conv2dReLU experts.

    Supports two modes:
      - Token Choice (v1, default): each pixel picks top_k experts.
        Requires auxiliary load-balancing loss.
      - Expert Choice (v2): each expert picks its top-C pixels.
        No auxiliary loss needed, guaranteed balance.

    Optional shared expert (DeepSeekMoE style): one dense Conv2dReLU
    always processes all pixels, providing a stable baseline. Sparse
    experts add class-specific refinements on top.

    Uses gradient checkpointing for sparse experts: each expert is
    recomputed during backward instead of storing activations.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        padding: Convolution padding.
        num_experts: Number of sparse experts.
        top_k: Number of experts per token (token_choice only).
        noise_std: Noise multiplier for gating (token_choice only).
        capacity_factor: Expert capacity multiplier (expert_choice only).
        use_shared_expert: If True, adds a shared dense expert.
        routing: "token_choice" or "expert_choice".
        use_norm: Normalization type for Conv2dReLU experts.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        num_experts: int = 8,
        top_k: int = 2,
        noise_std: float = 1.0,
        capacity_factor: float = 1.25,
        use_shared_expert: bool = False,
        routing: str = "token_choice",
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = use_shared_expert
        if routing not in ("token_choice", "expert_choice"):
            raise ValueError(
                f"routing must be 'token_choice' or 'expert_choice', got '{routing}'"
            )
        self.routing = routing

        # Router
        if routing == "expert_choice":
            self.router = ExpertChoiceRouter(
                in_channels=in_channels,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
            )
        else:
            self.router = NoisyTopKRouter(
                in_channels=in_channels,
                num_experts=num_experts,
                top_k=top_k,
                noise_std=noise_std,
            )

        # Shared expert (dense, always active)
        if use_shared_expert:
            self.shared_expert = Conv2dReLU(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                use_norm=use_norm,
            )

        # Sparse experts
        self.experts = nn.ModuleList(
            [
                Conv2dReLU(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    use_norm=use_norm,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, C_in, H, W)

        Returns:
            output: Mixed expert output (B, C_out, H, W)
            aux_loss: Scalar load-balancing loss (0 for expert_choice)
        """
        B, C, H, W = x.shape

        # 1. Shared expert baseline (if enabled)
        if self.use_shared_expert:
            output = self.shared_expert(x)
        else:
            output = torch.zeros(
                B,
                self.out_channels,
                H,
                W,
                device=x.device,
                dtype=x.dtype,
            )

        # 2. Route
        expert_weight_maps, aux_loss = self.router(x)
        # expert_weight_maps: (B, num_experts, H, W)

        # Store for diagnostics (detached to avoid graph retention)
        self.last_expert_weight_maps = expert_weight_maps.detach()

        # 3. Sparse experts with gradient checkpointing
        for e, expert in enumerate(self.experts):
            w_map = expert_weight_maps[:, e : e + 1, :, :]  # (B, 1, H, W)

            if self.training:
                expert_out = grad_checkpoint(expert, x, use_reentrant=False)
            else:
                expert_out = expert(x)

            output = output + expert_out * w_map

        return output, aux_loss

    def extra_repr(self) -> str:
        parts = [
            f"in_channels={self.in_channels}",
            f"out_channels={self.out_channels}",
            f"num_experts={self.num_experts}",
            f"routing={self.routing}",
        ]
        if self.use_shared_expert:
            parts.append("shared_expert=True")
        if self.routing == "token_choice":
            parts.append(f"top_k={self.top_k}")
        return ", ".join(parts)
