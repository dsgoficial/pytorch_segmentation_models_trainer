"""
UPerNet with Multi-Expert Decoder and Output Ensemble (MEDOE).

Forces expert specialization via cumulative/nested class sets and masked
loss per expert. HEAD sees all classes (baseline), BODY sees body+tail,
TAIL operates as a residual correction on its assigned class channels.
An auxiliary L2 loss suppresses logits for non-assigned classes.
OHEM selects the hardest pixels per batch for the ensemble loss.

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
      tail_residual: true
      ohem_ratio: 0.25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Union

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
    masked loss only for its assigned class group. The last expert
    (TAIL) can optionally operate as a residual correction instead of
    an independent prediction.

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
        expert_weights: Optional per-expert loss multipliers.
        aux_suppression_weight: Weight for L2 suppression of non-assigned
            class logits (paper default: 0.2).
        gate_entropy_weight: Weight for gate entropy regularization.
        tail_residual: If True, the last expert (TAIL) produces an additive
            residual correction on its class channels instead of an
            independent prediction blended via gate. Default: False.
        tail_residual_scale: Initial scale for residual correction.
        ohem_ratio: Fraction of hardest pixels to use for the ensemble loss.
            0.0 = disabled (use all pixels). E.g. 0.25 = top 25% hardest.
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
        tail_residual: bool = False,
        tail_residual_scale: float = 0.1,
        ohem_ratio: float = 0.0,
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
        self.tail_residual = tail_residual
        self.ohem_ratio = ohem_ratio

        # Convert OmegaConf to plain Python if needed
        if class_groups is not None:
            try:
                from omegaconf import DictConfig, OmegaConf

                if isinstance(class_groups, DictConfig):
                    class_groups = OmegaConf.to_container(class_groups, resolve=True)
            except ImportError:
                pass
        else:
            class_groups = {
                "HEAD": [0, 1, 2, 3, 4, 5],
                "BODY": [2, 3, 5],
                "TAIL": [2],
            }

        self.class_groups = class_groups
        self.group_names = list(class_groups.keys())
        self.num_experts = len(self.group_names)

        # For tail_residual mode, gate has one fewer expert (TAIL is not gated)
        self._num_gated_experts = (
            self.num_experts - 1 if tail_residual else self.num_experts
        )
        # TAIL expert index and its class channels
        self._tail_idx = self.num_experts - 1
        self._tail_classes = class_groups[self.group_names[self._tail_idx]]

        # Per-expert loss weights
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

        # Gate network — fewer channels if tail_residual (TAIL not gated)
        self.gate = nn.Conv2d(decoder_channels, self._num_gated_experts, kernel_size=1)

        # Learnable scale for residual correction
        if tail_residual:
            self.tail_scale = nn.Parameter(torch.tensor(tail_residual_scale))

        # Expert-specific loss (simple CE, robust to sparse masked targets)
        self.expert_criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

        # State for model.py integration
        self.last_expert_outputs: Optional[List[torch.Tensor]] = None
        self.last_gate_weights: Optional[torch.Tensor] = None
        self._gate_weights_with_grad: Optional[torch.Tensor] = None
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

    def _make_masked_target(self, target: torch.Tensor, group_idx: int) -> torch.Tensor:
        """Create masked target for an expert group."""
        group_mask = getattr(self, f"_group_mask_{group_idx}")
        valid = target != self.ignore_index
        in_group = group_mask[target.clamp(0, self.num_classes - 1)] & valid
        masked = target.clone()
        masked[~in_group] = self.ignore_index
        return masked

    def _compute_suppression_loss(
        self, expert_out: torch.Tensor, group_idx: int
    ) -> torch.Tensor:
        """L2 penalty on logits of non-assigned classes."""
        complement = getattr(self, f"_group_complement_{group_idx}")
        if not complement.any():
            return (expert_out * 0).sum()  # graph-connected zero
        interfering_logits = expert_out[:, complement]
        return interfering_logits.pow(2).mean()

    def _compute_gate_entropy_reg(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """Entropy regularization on gate weights to prevent collapse."""
        n_experts = gate_weights.shape[1]
        entropy = -(gate_weights * torch.log(gate_weights.clamp(min=1e-8))).sum(dim=1)
        max_entropy = torch.log(
            torch.tensor(float(n_experts), device=gate_weights.device)
        )
        return max_entropy - entropy.mean()

    def compute_ohem_loss(
        self,
        loss_fn: nn.Module,
        predictions: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Online Hard Example Mining: backprop only through hardest pixels.

        Computes per-pixel CE, selects top ohem_ratio fraction, returns mean.
        """
        if self.ohem_ratio <= 0 or self.ohem_ratio >= 1.0:
            return loss_fn(predictions, target)

        # Per-pixel loss (no reduction)
        with torch.no_grad():
            if target.is_floating_point():
                # soft labels — use hard version for pixel selection
                valid = target.sum(dim=1) > 0
                hard_target = target.argmax(dim=1)
                hard_target[~valid] = self.ignore_index
            else:
                hard_target = target
            pixel_loss = F.cross_entropy(
                predictions,
                hard_target,
                ignore_index=self.ignore_index,
                reduction="none",
            )  # (B, H, W)
            valid_mask = hard_target != self.ignore_index
            # Set ignored pixels to -inf so they're never selected
            pixel_loss[~valid_mask] = -1.0

        # Select top-K hardest valid pixels
        n_valid = valid_mask.sum().item()
        if n_valid == 0:
            return loss_fn(predictions, target)

        k = max(1, int(n_valid * self.ohem_ratio))
        flat_loss = pixel_loss.view(-1)
        _, hard_indices = flat_loss.topk(k)

        # Create OHEM mask: only hard pixels contribute
        ohem_mask = torch.zeros_like(flat_loss, dtype=torch.bool)
        ohem_mask[hard_indices] = True
        ohem_mask = ohem_mask.view_as(pixel_loss)

        # Set non-selected pixels to ignore in target
        if target.is_floating_point():
            # Soft target (B,C,H,W): zero out non-selected pixels per batch
            ohem_target = target.clone()
            # ohem_mask is (B,H,W), expand to (B,1,H,W) for broadcast
            ohem_target[~ohem_mask.unsqueeze(1).expand_as(ohem_target)] = 0
            return loss_fn(predictions, ohem_target)
        else:
            ohem_target = target.clone()
            ohem_target[~ohem_mask] = self.ignore_index
            return loss_fn(predictions, ohem_target)

    def compute_expert_loss(
        self,
        expert_outputs: List[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked expert losses + L2 suppression + gate entropy reg."""
        ce_losses = []
        ce_weights = []
        suppression_losses = []
        for i, expert_out in enumerate(expert_outputs):
            masked_target = self._make_masked_target(target, i)
            if (masked_target != self.ignore_index).any():
                ce_losses.append(self.expert_criterion(expert_out, masked_target))
                ce_weights.append(self.expert_weights[i])
            suppression_losses.append(self._compute_suppression_loss(expert_out, i))

        if not ce_losses:
            ce_term = torch.zeros(
                1, device=target.device, dtype=torch.float32
            ).squeeze()
        else:
            weights_t = torch.tensor(
                ce_weights, device=target.device, dtype=torch.float32
            )
            ce_term = (torch.stack(ce_losses) * weights_t).sum() / weights_t.sum()

        suppression_term = torch.stack(suppression_losses).mean()

        total = (
            self.expert_loss_weight * ce_term
            + self.aux_suppression_weight * suppression_term
        )

        # Gate entropy regularization
        if self.gate_entropy_weight > 0 and self._gate_weights_with_grad is not None:
            gate_reg = self._compute_gate_entropy_reg(self._gate_weights_with_grad)
            total = total + self.gate_entropy_weight * gate_reg
            self._gate_weights_with_grad = None

        return total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns ensemble logits."""
        features = self.encoder(x)
        decoder_output, aux_loss = self.decoder(features)

        expert_outputs = [expert(decoder_output) for expert in self.experts]

        gate_logits = self.gate(decoder_output)
        gate_logits = F.interpolate(
            gate_logits,
            size=expert_outputs[0].shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        gate_weights = F.softmax(gate_logits, dim=1)

        if self.tail_residual:
            # Gate blends HEAD + BODY (without TAIL)
            ensemble = torch.zeros_like(expert_outputs[0])
            for i in range(self._num_gated_experts):
                ensemble = ensemble + gate_weights[:, i : i + 1] * expert_outputs[i]
            # TAIL adds residual correction on its class channels only
            tail_out = expert_outputs[self._tail_idx]
            residual = torch.zeros_like(ensemble)
            for c in self._tail_classes:
                residual[:, c] = tail_out[:, c]
            ensemble = ensemble + self.tail_scale * residual
        else:
            ensemble = torch.zeros_like(expert_outputs[0])
            for i, expert_out in enumerate(expert_outputs):
                ensemble = ensemble + gate_weights[:, i : i + 1] * expert_out

        self.last_expert_outputs = expert_outputs
        self._gate_weights_with_grad = (
            gate_weights if self.gate_entropy_weight > 0 else None
        )
        self.last_gate_weights = gate_weights.detach()
        self.last_aux_loss = aux_loss

        return ensemble

    @torch.no_grad()
    def get_medoe_diagnostics(
        self, hard_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Collect MEDOE diagnostics for TensorBoard."""
        diagnostics: Dict[str, torch.Tensor] = {}
        if self.last_gate_weights is None:
            return diagnostics

        gw = self.last_gate_weights

        for i in range(gw.shape[1]):
            name = self.group_names[i] if i < len(self.group_names) else f"expert_{i}"
            diagnostics[f"medoe/gate_{name}_mean"] = gw[:, i].mean()

        entropy = -(gw * torch.log(gw.clamp(min=1e-8))).sum(dim=1)
        diagnostics["medoe/gate_entropy"] = entropy.mean()

        # Log tail_scale if residual mode
        if self.tail_residual:
            diagnostics["medoe/tail_scale"] = self.tail_scale.detach()

        if hard_masks is not None and self.last_expert_outputs is not None:
            for i, (name, class_indices) in enumerate(self.class_groups.items()):
                expert_pred = self.last_expert_outputs[i].detach().argmax(dim=1)
                for c in class_indices:
                    pred_c = expert_pred == c
                    gt_c = hard_masks == c
                    intersection = (pred_c & gt_c).sum().float()
                    union = (pred_c | gt_c).sum().float()
                    iou = intersection / union.clamp(min=1.0)
                    diagnostics[f"medoe/expert_{name}_iou/class_{c}"] = iou

        return diagnostics
