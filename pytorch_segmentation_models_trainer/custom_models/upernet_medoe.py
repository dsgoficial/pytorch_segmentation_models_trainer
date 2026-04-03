"""
UPerNet with Multi-Expert Decoder and Output Ensemble (MEDOE).

Forces expert specialization via cumulative/nested class sets and masked
loss per expert. HEAD sees all classes (baseline), BODY sees body+tail,
TAIL sees only tail. Each expert receives gradients only from its assigned
classes. An auxiliary L2 loss suppresses logits for non-assigned classes.

Ref: MEDOE — Multi-Expert Decoder and Output Ensemble (arXiv:2308.08213)

Usage via Hydra YAML:
    model:
      _target_: pytorch_segmentation_models_trainer.custom_models.upernet_medoe.UPerNetMEDOE
      encoder_name: tu-convnextv2_tiny.fcmae_ft_in22k_in1k
      encoder_weights: true
      in_channels: 3
      classes: 6
      class_groups:
        HEAD: [0, 1, 2, 3, 4, 5]
        BODY: [2, 3, 5]
        TAIL: [2]
      expert_loss_weight: 0.5
      aux_suppression_weight: 0.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.encoders import get_encoder

from pytorch_segmentation_models_trainer.custom_models.upernet_moe import (
    UPerNetMoEDecoder,
)


class ExpertBranch(nn.Module):
    """Single expert branch: Conv2dReLU refinement + SegmentationHead."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        upsampling: int = 4,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
    ):
        super().__init__()
        self.refine = md.Conv2dReLU(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )
        self.head = SegmentationHead(
            in_channels=hidden_channels,
            out_channels=num_classes,
            kernel_size=1,
            upsampling=upsampling,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.refine(x))


class UPerNetMEDOE(nn.Module):
    """UPerNet with Multi-Expert Decoder and Output Ensemble.

    Architecture:
        Shared encoder -> Shared UPerNet decoder -> N expert branches + gate
        -> gated ensemble output.

    Each expert branch produces logits for ALL classes, but receives
    masked loss only for its assigned class group. The ensemble output
    receives full loss on all classes.

    Args:
        encoder_name: Name of timm/smp encoder.
        encoder_depth: Number of encoder stages.
        encoder_weights: Pretrained weights.
        decoder_channels: Decoder intermediate channels.
        decoder_use_norm: Normalization for decoder conv blocks.
        in_channels: Input image channels.
        classes: Number of output classes.
        activation: Output activation function.
        upsampling: Upsampling factor for segmentation heads.
        class_groups: Dict mapping group names to lists of class indices.
            Should be cumulative/nested: HEAD sees all classes, BODY sees
            body+tail, TAIL sees only tail.
            Default: HEAD=[0..5], BODY=[2,3,5], TAIL=[2] for 6-class LULC.
        expert_loss_weight: Weight for expert CE losses.
        expert_weights: Optional per-expert loss multipliers. List of floats
            with length == number of experts. E.g. [1, 1, 3] gives TAIL 3x
            more weight. Default: equal weights.
        aux_suppression_weight: Weight for L2 suppression of non-assigned
            class logits (paper default: 0.2).
        gate_entropy_weight: Weight for gate entropy regularization. Penalizes
            gate collapse (all weight on one expert). 0 = disabled.
        label_smoothing: Label smoothing for expert CE loss.
        ignore_index: Index for ignored pixels.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_channels: int = 256,
        decoder_use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        in_channels: int = 3,
        classes: int = 6,
        activation: Optional[Union[str, Callable]] = None,
        upsampling: int = 4,
        class_groups: Optional[Dict[str, List[int]]] = None,
        expert_loss_weight: float = 0.5,
        expert_weights: Optional[List[float]] = None,
        aux_suppression_weight: float = 0.2,
        gate_entropy_weight: float = 0.0,
        label_smoothing: float = 0.1,
        ignore_index: int = 255,
        **kwargs: Any,
    ):
        super().__init__()

        self.num_classes = classes
        self.ignore_index = ignore_index
        self.expert_loss_weight = expert_loss_weight
        self.aux_suppression_weight = aux_suppression_weight
        self.gate_entropy_weight = gate_entropy_weight

        # Convert OmegaConf to plain Python if needed
        if class_groups is not None:
            try:
                from omegaconf import DictConfig, OmegaConf

                if isinstance(class_groups, DictConfig):
                    class_groups = OmegaConf.to_container(
                        class_groups, resolve=True
                    )
            except ImportError:
                pass
        else:
            # Cumulative/nested sets (paper design):
            # HEAD sees all classes, BODY sees body+tail, TAIL sees only tail
            class_groups = {
                "HEAD": [0, 1, 2, 3, 4, 5],
                "BODY": [2, 3, 5],
                "TAIL": [2],
            }

        self.class_groups = class_groups
        self.group_names = list(class_groups.keys())
        self.num_experts = len(self.group_names)

        # Per-expert loss weights (e.g. [1, 1, 3] to give TAIL 3x more weight)
        if expert_weights is not None:
            if len(expert_weights) != self.num_experts:
                raise ValueError(
                    f"expert_weights length {len(expert_weights)} != "
                    f"num_experts {self.num_experts}"
                )
            total = sum(expert_weights)
            self.expert_weights = [w * self.num_experts / total for w in expert_weights]
        else:
            self.expert_weights = [1.0] * self.num_experts

        # Pre-compute class-to-group membership masks and their complements
        for i, (name, indices) in enumerate(class_groups.items()):
            mask = torch.zeros(classes, dtype=torch.bool)
            for idx in indices:
                mask[idx] = True
            self.register_buffer(f"_group_mask_{i}", mask)
            # Complement: classes NOT in this group (for L2 suppression)
            self.register_buffer(f"_group_complement_{i}", ~mask)

        # Encoder (shared)
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **{k: v for k, v in kwargs.items() if v is not None},
        )

        # Shared decoder (plain UPerNet, no MoE)
        self.decoder = UPerNetMoEDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            use_norm=decoder_use_norm,
            moe_at_fusion=False,
            moe_at_fpn=False,
        )

        # Expert branches
        self.experts = nn.ModuleList(
            [
                ExpertBranch(
                    in_channels=decoder_channels,
                    hidden_channels=decoder_channels,
                    num_classes=classes,
                    upsampling=upsampling,
                    use_norm=decoder_use_norm,
                )
                for _ in range(self.num_experts)
            ]
        )

        # Gate network (at decoder resolution, upsampled later)
        self.gate = nn.Conv2d(decoder_channels, self.num_experts, kernel_size=1)

        # Expert-specific loss (simple CE, robust to sparse masked targets)
        self.expert_criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

        # State for model.py integration
        self.last_expert_outputs: Optional[List[torch.Tensor]] = None
        self.last_gate_weights: Optional[torch.Tensor] = None
        self.last_aux_loss = torch.tensor(0.0)

        self.classification_head = None
        self.name = f"upernet-medoe-{encoder_name}"
        self._initialize()

    def _initialize(self):
        init.initialize_decoder(self.decoder)
        for expert in self.experts:
            init.initialize_head(expert.head)
        nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.gate.weight)

    def _make_masked_target(
        self, target: torch.Tensor, group_idx: int
    ) -> torch.Tensor:
        """Create masked target for an expert group.

        Pixels NOT belonging to the expert's class group are set to
        ignore_index so CrossEntropyLoss skips them.
        """
        group_mask = getattr(self, f"_group_mask_{group_idx}")
        valid = target != self.ignore_index
        in_group = group_mask[target.clamp(0, self.num_classes - 1)] & valid
        masked = target.clone()
        masked[~in_group] = self.ignore_index
        return masked

    def _compute_suppression_loss(
        self, expert_out: torch.Tensor, group_idx: int
    ) -> torch.Tensor:
        """L2 penalty on logits of non-assigned classes.

        Encourages expert to produce near-zero logits for classes it
        should not specialize in, reducing interference in the ensemble.
        """
        complement = getattr(self, f"_group_complement_{group_idx}")  # (C,)
        if not complement.any():
            # HEAD expert sees all classes — no suppression needed
            return (expert_out * 0).sum()  # graph-connected zero
        # expert_out: (B, C, H, W) — select non-assigned class channels
        interfering_logits = expert_out[:, complement]  # (B, C', H, W)
        return interfering_logits.pow(2).mean()

    def _compute_gate_entropy_reg(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """Entropy regularization on gate weights to prevent collapse.

        Maximizes entropy = encourages the gate to distribute weight
        across experts instead of putting everything on HEAD.

        Returns negative entropy (to be minimized).
        """
        # gate_weights: (B, E, H, W), already softmax
        entropy = -(gate_weights * torch.log(gate_weights.clamp(min=1e-8))).sum(dim=1)
        # Maximize entropy → minimize negative entropy
        max_entropy = torch.log(torch.tensor(float(self.num_experts)))
        return max_entropy - entropy.mean()

    def compute_expert_loss(
        self,
        expert_outputs: List[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked expert losses + L2 suppression + gate entropy reg.

        Each expert's CE loss is computed only on pixels of its assigned
        class group, weighted by expert_weights. An auxiliary L2 loss
        penalizes logits of non-assigned classes. Gate entropy regularization
        prevents gate collapse to a single expert.

        Returns:
            Weighted combination of all loss terms (scalar).
        """
        ce_losses = []
        ce_weights = []
        suppression_losses = []
        for i, expert_out in enumerate(expert_outputs):
            masked_target = self._make_masked_target(target, i)
            if (masked_target != self.ignore_index).any():
                ce_losses.append(self.expert_criterion(expert_out, masked_target))
                ce_weights.append(self.expert_weights[i])
            suppression_losses.append(
                self._compute_suppression_loss(expert_out, i)
            )

        if not ce_losses:
            ce_term = torch.zeros(1, device=target.device, dtype=torch.float32).squeeze()
        else:
            # Weighted mean of expert CE losses
            weights_t = torch.tensor(ce_weights, device=target.device, dtype=torch.float32)
            ce_term = (torch.stack(ce_losses) * weights_t).sum() / weights_t.sum()

        suppression_term = torch.stack(suppression_losses).mean()

        total = self.expert_loss_weight * ce_term + self.aux_suppression_weight * suppression_term

        # Gate entropy regularization (needs grad-connected weights)
        if self.gate_entropy_weight > 0 and self._gate_weights_with_grad is not None:
            gate_reg = self._compute_gate_entropy_reg(self._gate_weights_with_grad)
            total = total + self.gate_entropy_weight * gate_reg
            self._gate_weights_with_grad = None  # free after use

        return total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns ensemble logits.

        Stores expert outputs and gate weights for loss computation
        and diagnostics.
        """
        features = self.encoder(x)
        decoder_output, aux_loss = self.decoder(features)

        expert_outputs = [expert(decoder_output) for expert in self.experts]

        gate_logits = self.gate(decoder_output)
        # Interpolate logits before softmax so probabilities sum to 1 at full resolution
        gate_logits = F.interpolate(
            gate_logits,
            size=expert_outputs[0].shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        gate_weights = F.softmax(gate_logits, dim=1)

        ensemble = torch.zeros_like(expert_outputs[0])
        for i, expert_out in enumerate(expert_outputs):
            ensemble = ensemble + gate_weights[:, i : i + 1] * expert_out

        self.last_expert_outputs = expert_outputs
        # Keep graph-connected version for gate entropy regularization
        self._gate_weights_with_grad = gate_weights
        self.last_gate_weights = gate_weights.detach()
        self.last_aux_loss = aux_loss

        return ensemble

    @torch.no_grad()
    def get_medoe_diagnostics(
        self, hard_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Collect MEDOE diagnostics for TensorBoard.

        Returns:
            - medoe/gate_{name}_mean: mean gate weight per expert
            - medoe/gate_entropy: spatial routing entropy
            - medoe/expert_{name}_iou/class_{c}: per-expert per-class IoU
              (only if hard_masks provided, validation only)
        """
        diagnostics: Dict[str, torch.Tensor] = {}
        if self.last_gate_weights is None:
            return diagnostics

        gw = self.last_gate_weights

        for i, name in enumerate(self.group_names):
            diagnostics[f"medoe/gate_{name}_mean"] = gw[:, i].mean()

        entropy = -(gw * torch.log(gw.clamp(min=1e-8))).sum(dim=1)
        diagnostics["medoe/gate_entropy"] = entropy.mean()

        if hard_masks is not None and self.last_expert_outputs is not None:
            for i, (name, class_indices) in enumerate(
                self.class_groups.items()
            ):
                expert_pred = self.last_expert_outputs[i].detach().argmax(
                    dim=1
                )
                for c in class_indices:
                    pred_c = expert_pred == c
                    gt_c = hard_masks == c
                    intersection = (pred_c & gt_c).sum().float()
                    union = (pred_c | gt_c).sum().float()
                    iou = intersection / union.clamp(min=1.0)
                    diagnostics[f"medoe/expert_{name}_iou/class_{c}"] = iou

        return diagnostics
