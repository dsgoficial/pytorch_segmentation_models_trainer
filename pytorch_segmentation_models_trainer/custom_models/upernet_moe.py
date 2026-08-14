"""
UPerNet decoder with Mixture of Experts (MoE).

Replaces selected Conv2dReLU blocks in the UPerNet decoder with
MoEConv2dReLU blocks for class-specialized feature processing.

v2 improvements:
  - Expert Choice routing: balanced without auxiliary loss
  - Shared expert: dense baseline + sparse specialization
  - Backward compatible: default params reproduce v1 behavior

Usage via Hydra YAML:
    model:
      _target_: pytorch_segmentation_models_trainer.custom_models.upernet_moe.UPerNetMoE
      encoder_name: tu-convnextv2_large.fcmae_ft_in22k_in1k
      encoder_weights: imagenet
      in_channels: 3
      classes: 7
      moe_num_experts: 6
      moe_use_shared_expert: true
      moe_routing: "expert_choice"
      moe_capacity_factor: 1.25
      moe_aux_loss_weight: 0.0
      moe_at_fusion: true
      moe_at_fpn: false
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from segmentation_models_pytorch.base import (
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.upernet.decoder import (
    PSPModule,
    FPNLateralBlock,
    LayerNorm2d,
)

from pytorch_segmentation_models_trainer.custom_models.moe_layers import MoEConv2dReLU


class UPerNetMoEDecoder(nn.Module):
    """UPerNet decoder with optional MoE at fusion and/or FPN conv blocks.

    Replicates the original UPerNetDecoder structure (PSP + FPN + fusion)
    but can replace fusion_block and/or fpn_conv_blocks with MoEConv2dReLU.

    Args:
        encoder_channels: Channel counts from encoder stages.
        encoder_depth: Number of encoder stages.
        decoder_channels: Intermediate channel count (default 256).
        use_norm: Normalization type for conv blocks.
        moe_num_experts: Number of sparse experts per MoE block.
        moe_top_k: Number of active experts per token (token_choice).
        moe_noise_std: Noise multiplier for gating (token_choice).
        moe_capacity_factor: Expert capacity multiplier (expert_choice).
        moe_use_shared_expert: Add a shared dense expert.
        moe_routing: "token_choice" or "expert_choice".
        moe_at_fusion: Use MoE at the fusion block.
        moe_at_fpn: Use MoE at the FPN conv blocks.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        encoder_depth: int = 5,
        decoder_channels: int = 256,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_noise_std: float = 1.0,
        moe_capacity_factor: float = 1.25,
        moe_use_shared_expert: bool = False,
        moe_routing: str = "token_choice",
        moe_at_fusion: bool = True,
        moe_at_fpn: bool = False,
    ):
        super().__init__()

        if encoder_depth < 3:
            raise ValueError(
                f"Encoder depth for UPerNet decoder cannot be less than 3, "
                f"got {encoder_depth}."
            )

        self.moe_at_fusion = moe_at_fusion
        self.moe_at_fpn = moe_at_fpn

        # MoE kwargs shared by fusion and FPN blocks
        moe_kwargs = dict(
            num_experts=moe_num_experts,
            top_k=moe_top_k,
            noise_std=moe_noise_std,
            capacity_factor=moe_capacity_factor,
            use_shared_expert=moe_use_shared_expert,
            routing=moe_routing,
            use_norm=use_norm,
        )

        # Skip first two encoder levels (same as original UPerNet)
        encoder_channels = encoder_channels[2:]

        # LayerNorm for each encoder feature level
        self.feature_norms = nn.ModuleList(
            [LayerNorm2d(channels, eps=1e-6) for channels in encoder_channels]
        )

        # PSP Module (unchanged)
        lowest_resolution_feature_channels = encoder_channels[-1]
        self.psp = PSPModule(
            in_channels=lowest_resolution_feature_channels,
            out_channels=decoder_channels,
            sizes=(1, 2, 3, 6),
            use_norm=use_norm,
        )

        # FPN lateral + conv blocks
        lateral_channels = encoder_channels[:-1][::-1]
        self.fpn_lateral_blocks = nn.ModuleList([])
        self.fpn_conv_blocks = nn.ModuleList([])

        for channels in lateral_channels:
            block = FPNLateralBlock(
                lateral_channels=channels,
                out_channels=decoder_channels,
                use_norm=use_norm,
            )
            self.fpn_lateral_blocks.append(block)

            if moe_at_fpn:
                conv_block = MoEConv2dReLU(
                    in_channels=decoder_channels,
                    out_channels=decoder_channels,
                    kernel_size=3,
                    padding=1,
                    **moe_kwargs,
                )
            else:
                conv_block = md.Conv2dReLU(
                    in_channels=decoder_channels,
                    out_channels=decoder_channels,
                    kernel_size=3,
                    padding=1,
                    use_norm=use_norm,
                )
            self.fpn_conv_blocks.append(conv_block)

        # Fusion block
        num_blocks_to_fuse = len(self.fpn_conv_blocks) + 1
        if moe_at_fusion:
            self.fusion_block = MoEConv2dReLU(
                in_channels=num_blocks_to_fuse * decoder_channels,
                out_channels=decoder_channels,
                kernel_size=3,
                padding=1,
                **moe_kwargs,
            )
        else:
            self.fusion_block = md.Conv2dReLU(
                in_channels=num_blocks_to_fuse * decoder_channels,
                out_channels=decoder_channels,
                kernel_size=3,
                padding=1,
                use_norm=use_norm,
            )

    def forward(
        self, features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Encoder features at multiple scales.

        Returns:
            output: Decoded feature map (B, decoder_channels, H/4, W/4)
            total_aux_loss: Sum of all MoE auxiliary losses
        """
        total_aux_loss = torch.tensor(
            0.0, device=features[0].device, dtype=features[0].dtype
        )

        # Skip 1/1 and 1/2 resolution
        features = features[2:]

        # Normalize features
        for i, norm in enumerate(self.feature_norms):
            features[i] = norm(features[i])

        # PSP on lowest resolution
        psp_out = self.psp(features[-1])

        # FPN path
        fpn_lateral_features = features[:-1][::-1]
        fpn_features = [psp_out]
        for i, block in enumerate(self.fpn_lateral_blocks):
            lateral_feature = fpn_lateral_features[i]
            state_feature = fpn_features[-1]
            fpn_feature = block(state_feature, lateral_feature)
            fpn_features.append(fpn_feature)

        # FPN conv blocks (may be MoE)
        for i, conv_block in enumerate(self.fpn_conv_blocks, start=1):
            if isinstance(conv_block, MoEConv2dReLU):
                fpn_features[i], aux_loss = conv_block(fpn_features[i])
                total_aux_loss = total_aux_loss + aux_loss
            else:
                fpn_features[i] = conv_block(fpn_features[i])

        # Resize all to 1/4 resolution
        target_size = fpn_features[-1].shape[2:]
        resized_fpn_features = []
        for feature in fpn_features:
            resized_feature = F.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=False
            )
            resized_fpn_features.append(resized_feature)

        # Concatenate and fuse
        stacked_features = torch.cat(resized_fpn_features[::-1], dim=1)

        if isinstance(self.fusion_block, MoEConv2dReLU):
            output, aux_loss = self.fusion_block(stacked_features)
            total_aux_loss = total_aux_loss + aux_loss
        else:
            output = self.fusion_block(stacked_features)

        return output, total_aux_loss


class UPerNetMoE(nn.Module):
    """UPerNet with Mixture of Experts decoder.

    Follows the same encoder-decoder-head pattern as smp.SegmentationModel
    but stores auxiliary MoE loss in self.last_aux_loss for the training
    pipeline to read. forward() returns only masks for inference compatibility.

    Args:
        encoder_name: Name of timm/smp encoder.
        encoder_depth: Number of encoder stages.
        encoder_weights: Pretrained weights name.
        decoder_channels: Decoder intermediate channels.
        decoder_use_norm: Normalization for decoder conv blocks.
        in_channels: Input image channels.
        classes: Number of output classes.
        activation: Output activation function.
        upsampling: Upsampling factor for segmentation head.
        moe_num_experts: Number of sparse experts per MoE block.
        moe_top_k: Number of active experts per token (token_choice).
        moe_noise_std: Noise multiplier for gating (token_choice).
        moe_capacity_factor: Expert capacity multiplier (expert_choice).
        moe_use_shared_expert: Add a shared dense expert per MoE block.
        moe_routing: "token_choice" or "expert_choice".
        moe_aux_loss_weight: Weight for auxiliary loss (0 for expert_choice).
        moe_at_fusion: Use MoE at the fusion block.
        moe_at_fpn: Use MoE at the FPN conv blocks.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_channels: int = 256,
        decoder_use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,
        upsampling: int = 4,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_noise_std: float = 1.0,
        moe_capacity_factor: float = 1.25,
        moe_use_shared_expert: bool = False,
        moe_routing: str = "token_choice",
        moe_aux_loss_weight: float = 0.01,
        moe_at_fusion: bool = True,
        moe_at_fpn: bool = False,
        **kwargs: Any,
    ):
        super().__init__()

        self.moe_aux_loss_weight = moe_aux_loss_weight
        self.last_aux_loss = torch.tensor(0.0)
        self.num_classes = classes

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            **{k: v for k, v in kwargs.items() if v is not None},
        )

        self.decoder = UPerNetMoEDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            use_norm=decoder_use_norm,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_noise_std=moe_noise_std,
            moe_capacity_factor=moe_capacity_factor,
            moe_use_shared_expert=moe_use_shared_expert,
            moe_routing=moe_routing,
            moe_at_fusion=moe_at_fusion,
            moe_at_fpn=moe_at_fpn,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        self.classification_head = None

        self.name = f"upernet-moe-{encoder_name}"
        self._initialize()

    def _initialize(self):
        """Initialize decoder and head weights."""
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Stores weighted auxiliary MoE loss in self.last_aux_loss.
        Returns only masks for inference/TTA compatibility.

        Args:
            x: Input image (B, C, H, W)

        Returns:
            masks: Predicted segmentation masks (B, classes, H, W)
        """
        features = self.encoder(x)
        decoder_output, aux_loss = self.decoder(features)
        masks = self.segmentation_head(decoder_output)

        self.last_aux_loss = self.moe_aux_loss_weight * aux_loss

        return masks

    def _get_active_moe_blocks(self) -> List[Tuple[int, str, MoEConv2dReLU]]:
        """Return MoE blocks that have stored weight maps from the last forward pass."""
        blocks = []
        for name, module in self.decoder.named_modules():
            if isinstance(module, MoEConv2dReLU) and hasattr(
                module, "last_expert_weight_maps"
            ):
                blocks.append((len(blocks), name, module))
        return blocks

    @torch.no_grad()
    def get_moe_diagnostics(self) -> Dict[str, torch.Tensor]:
        """Collect routing diagnostics from all MoE blocks.

        Returns dict with scalar metrics (detached, ready for logging):
          - moe/expert_{e}_utilization: fraction of pixels selected by expert e
          - moe/expert_overlap: mean number of active experts per pixel
          - moe/routing_entropy: entropy of per-pixel expert distribution (0=collapse, log(E)=uniform)
          - moe/max_weight_mean: mean of the max expert weight per pixel (confidence)
        """
        diagnostics: Dict[str, torch.Tensor] = {}
        moe_blocks = self._get_active_moe_blocks()
        if not moe_blocks:
            return diagnostics

        use_prefix = len(moe_blocks) > 1
        for idx, block_name, block in moe_blocks:
            p = f"moe/b{idx}_" if use_prefix else "moe/"
            w = block.last_expert_weight_maps  # (B, E, H, W)
            E = w.shape[1]

            active = (w > 0).float()  # (B, E, H, W)
            for e in range(E):
                diagnostics[f"{p}expert_{e}_utilization"] = active[:, e].mean()

            experts_per_pixel = active.sum(dim=1)  # (B, H, W)
            diagnostics[f"{p}expert_overlap"] = experts_per_pixel.mean()

            # Routing entropy
            w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
            w_norm = w / w_sum  # (B, E, H, W)
            entropy = -(w_norm * torch.log(w_norm.clamp(min=1e-8))).sum(dim=1)
            diagnostics[f"{p}routing_entropy"] = entropy.mean()

            max_weight = w.max(dim=1).values  # (B, H, W)
            diagnostics[f"{p}max_weight_mean"] = max_weight.mean()

        return diagnostics

    @torch.no_grad()
    def get_expert_class_affinity(
        self, hard_masks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute expert-class affinity: which experts prefer which classes.

        For each expert e and class c, computes the mean routing weight
        on pixels of class c. High values mean expert e specializes in class c.

        Args:
            hard_masks: Ground truth masks (B, H, W) at model output resolution.
                        Pixels with value 255 are ignored.

        Returns:
            Dict with moe/affinity_e{e}_c{c} scalars.
        """
        num_classes = self.num_classes
        diagnostics: Dict[str, torch.Tensor] = {}
        moe_blocks = self._get_active_moe_blocks()
        if not moe_blocks:
            return diagnostics

        use_prefix = len(moe_blocks) > 1
        for idx, block_name, block in moe_blocks:
            p = f"moe/b{idx}_" if use_prefix else "moe/"
            w = block.last_expert_weight_maps  # (B, E, H_feat, W_feat)
            E = w.shape[1]

            # Resize masks to feature map resolution
            masks_resized = (
                F.interpolate(
                    hard_masks.unsqueeze(1).float(),
                    size=w.shape[2:],
                    mode="nearest",
                )
                .squeeze(1)
                .long()
            )  # (B, H_feat, W_feat)

            valid = masks_resized != 255
            # Vectorized: compute mean weight per expert per class in one pass
            # Replace invalid pixels with num_classes so they don't match any class
            safe_masks = masks_resized.clone()
            safe_masks[~valid] = num_classes
            # One-hot encode: (B, num_classes+1, H, W), drop the extra channel
            one_hot = F.one_hot(safe_masks, num_classes + 1).permute(0, 3, 1, 2).float()
            one_hot = one_hot[:, :num_classes]  # (B, C, H, W)
            pixels_per_class = one_hot.sum(dim=(0, 2, 3))  # (C,)

            for e in range(E):
                w_e = w[:, e : e + 1]  # (B, 1, H, W)
                weighted = (w_e * one_hot).sum(dim=(0, 2, 3))  # (C,)
                for c in range(num_classes):
                    if pixels_per_class[c] > 0:
                        diagnostics[f"{p}affinity_e{e}_c{c}"] = (
                            weighted[c] / pixels_per_class[c]
                        )
                    else:
                        diagnostics[f"{p}affinity_e{e}_c{c}"] = w.new_tensor(0.0)

        return diagnostics
