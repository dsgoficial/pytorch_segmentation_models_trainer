# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-11-09
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba - Cartographic Engineer
                                                            @ Brazilian Army
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   Building blocks (ConvBNReLU, GlobalLocalAttention, Block, WF,          *
 *   FeatureRefinementHead, AuxHead) ported from the reference              *
 *   implementation of UNetFormer (Wang et al., 2022,                       *
 *   "UNetFormer: A UNet-like Transformer for Efficient Semantic            *
 *   Segmentation of Remote Sensing Urban Scene Imagery"), repo             *
 *   https://github.com/WangLibo1995/GeoSeg (GPL-3.0), file                 *
 *   geoseg/models/UNetFormer.py, fetched verbatim from the `main` branch   *
 *   on 2026-11-09 and adapted here to plug into the segmentation_models_   *
 *   pytorch (smp) encoder/decoder/segmentation_head interface instead of   *
 *   timm's `features_only` wrapper -- so it's a drop-in alternative to     *
 *   `smp.Unet`/`smp.FPN` for any code that already expects                 *
 *   `backbone.encoder(x)` / `backbone.decoder(feats)` /                    *
 *   `backbone.segmentation_head(decoded)`.                                 *
 ****
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_

import segmentation_models_pytorch as smp

# ─────────────────────────────────────────────────────────────────────────
# Blocos de construcao (porte fiel do GeoSeg, so' renomeando nada)
# ─────────────────────────────────────────────────────────────────────────


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                     dilation=dilation, stride=stride,
                     padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6(),
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                     dilation=dilation, stride=stride,
                     padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                     dilation=dilation, stride=stride,
                     padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, norm_layer=nn.BatchNorm2d):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                     padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                     groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.ReLU6, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    """Atencao em janelas (local) + media-por-faixa (global aproximado) +
    ramo convolucional local, com bias de posicao relativa -- exatamente
    como no GeoSeg (ver cabecalho do arquivo)."""

    def __init__(self, dim=256, num_heads=16, qkv_bias=False, window_size=8,
                 relative_pos_embedding=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,
                                   padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1,
                                   padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding
        if relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def pad(self, x, ps):
        """Pad right/bottom up to a multiple of the window size ``ps``.

        Reflect padding, as in GeoSeg, whenever it is possible (pad smaller
        than the map); otherwise (map smaller than the window, e.g. the 1/32
        map of a 128 px input) the border is replicated. The pad is always a
        4-tuple: torch rejects a 2-tuple reflect pad on 4D tensors, so the
        original call only worked because it never triggered at 256 px.

        Args:
            x: (B, C, H, W) feature map.
            ps: window size.

        Returns:
            (B, C, H', W') tensor with H', W' multiples of ``ps`` (``x``
            itself when no padding is needed).
        """
        _, _, H, W = x.size()
        if W % ps != 0:
            p = ps - W % ps
            x = F.pad(x, (0, p, 0, 0), mode="reflect" if p < W else "replicate")
        if H % ps != 0:
            p = ps - H % ps
            x = F.pad(x, (0, 0, 0, p), mode="reflect" if p < H else "replicate")
        return x

    def pad_out(self, x):
        return F.pad(x, pad=(0, 1, 0, 1), mode="reflect")

    def forward(self, x):
        B, C, H, W = x.shape
        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(
            qkv, "b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d",
            h=self.num_heads, d=C // self.num_heads, hh=Hp // self.ws,
            ww=Wp // self.ws, qkv=3, ws1=self.ws, ws2=self.ws,
        )
        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)
            bias = bias.permute(2, 0, 1).contiguous()
            dots += bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v
        attn = rearrange(
            attn, "(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)",
            h=self.num_heads, d=C // self.num_heads, hh=Hp // self.ws,
            ww=Wp // self.ws, ws1=self.ws, ws2=self.ws,
        )
        attn = attn[:, :, :H, :W]

        out = (self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode="reflect")) +
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode="reflect")))
        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        return out[:, :, :H, :W]


class GLTBlock(nn.Module):
    """Global-Local Transformer Block (chamado so' de `Block` no GeoSeg;
    renomeado aqui so' pra nao colidir com outros `Block` do repo)."""

    def __init__(self, dim=256, num_heads=16, mlp_ratio=4.0, qkv_bias=False,
                 drop=0.0, drop_path=0.0, act_layer=nn.ReLU6,
                 norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                         window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim,
                      act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WF(nn.Module):
    """Weighted Feature fusion (soma ponderada aprendida entre o skip do
    encoder e o decoder upsampled, pesos normalizados tipo softmax-L1)."""

    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        w = nn.ReLU()(self.weights)
        fw = w / (torch.sum(w, dim=0) + self.eps)
        x = fw[0] * self.pre_conv(res) + fw[1] * x
        return self.post_conv(x)


class FeatureRefinementHead(nn.Module):
    """WF + atencao de canal (SE-like) e espacial (depthwise+sigmoid),
    usado no ultimo estagio (resolucao mais fina) do decoder."""

    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1,
                     groups=decode_channels),
            nn.Sigmoid(),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(decode_channels, decode_channels // 16, kernel_size=1),
            nn.ReLU6(),
            Conv(decode_channels // 16, decode_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        w = nn.ReLU()(self.weights)
        fw = w / (torch.sum(w, dim=0) + self.eps)
        x = fw[0] * self.pre_conv(res) + fw[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        x = self.pa(x) * x + self.ca(x) * x
        x = self.proj(x) + shortcut
        return self.act(x)


class AuxHead(nn.Module):
    """Cabeca auxiliar de supervisao profunda (deep supervision), usada so'
    durante treino no GeoSeg original. Opcional aqui (ver `use_aux_head`
    em `UNetFormerDecoder`) -- desligada por padrao pra manter o forward()
    com saida unica (1 tensor), igual ao resto do pipeline D_Hier."""

    def __init__(self, in_channels=64, num_classes=6):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────
# Adaptacao pro padrao smp: decoder(feats) -> feature map (NAO logits);
# segmentation_head(decoded) -> logits. O GeoSeg original faz os dois
# passos dentro do mesmo `Decoder.forward` -- aqui separamos pra plugar
# sem mudanca em qualquer codigo que ja espera a convencao smp (incluindo
# o D_Hier DV3Inject, que injeta contexto em `feats[inject_level]` ANTES
# de chamar o decoder, sem saber/se importar com qual decoder e').
# ─────────────────────────────────────────────────────────────────────────


class UNetFormerDecoder(nn.Module):
    """Espera `feats` no formato smp (lista de 6 estagios, indices 0..5 =
    resolucoes [1, 1/2, 1/4, 1/8, 1/16, 1/32] do input) -- usa so' os 4
    ultimos (1/4,1/8,1/16,1/32), que correspondem exatamente aos res1..res4
    do GeoSeg original (lá extraidos via timm `features_only`,
    out_indices=(1,2,3,4)). `inject_level` do D_Hier, por default 3, cai em
    feats[3] = 1/8 = res2 aqui -- a injecao acontece antes deste decoder
    ser chamado, de forma totalmente agnostica a qual decoder e' usado."""

    def __init__(self, encoder_channels, decode_channels=64, dropout=0.1,
                window_size=8, use_aux_head=False, num_classes=6):
        super().__init__()
        # encoder_channels no formato smp: (stem, l1, l2, l3, l4) = 5 valores
        # (ignora o canal "0" = in_channels da imagem). GeoSeg usa so' os
        # 4 ultimos (64,128,256,512 pro resnet18/34).
        ch = encoder_channels[-4:]
        self.use_aux_head = use_aux_head

        self.pre_conv = ConvBN(ch[-1], decode_channels, kernel_size=1)
        self.b4 = GLTBlock(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = GLTBlock(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(ch[-2], decode_channels)

        self.b2 = GLTBlock(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(ch[-3], decode_channels)

        self.p1 = FeatureRefinementHead(ch[-4], decode_channels)

        if use_aux_head:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.aux_head = AuxHead(decode_channels, num_classes)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feats):
        res1, res2, res3, res4 = feats[-4], feats[-3], feats[-2], feats[-1]
        h, w = res1.shape[-2] * 4, res1.shape[-1] * 4  # resolucao original (res1 e' 1/4)

        x = self.b4(self.pre_conv(res4))
        aux = None
        if self.use_aux_head and self.training:
            h4 = self.up4(x)

        x = self.p3(x, res3)
        x = self.b3(x)
        if self.use_aux_head and self.training:
            h3 = self.up3(x)

        x = self.p2(x, res2)
        x = self.b2(x)
        if self.use_aux_head and self.training:
            h2 = x
            aux = self.aux_head(h4 + h3 + h2, h, w)

        x = self.p1(x, res1)  # (B, decode_channels, H/4, W/4) -- segmentation_head faz o resto

        if self.use_aux_head and self.training:
            self._last_aux = aux  # recuperado por UNetFormerBackbone.forward
        return x


class UNetFormerSegmentationHead(nn.Sequential):
    """Equivalente ao `segmentation_head` + upsample final que, no GeoSeg
    original, ficava dentro do `Decoder.forward`. Separado aqui pra bater
    com a convencao smp (`backbone.segmentation_head(decoded)` -> logits
    na resolucao do input)."""

    def __init__(self, decode_channels, num_classes, dropout=0.1):
        super().__init__(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        h, w = x.shape[-2] * 4, x.shape[-1] * 4
        x = super().forward(x)
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


class UNetFormerBackbone(nn.Module):
    """Drop-in alternativo a `smp.Unet(...)`/`smp.FPN(...)`: mesmo encoder
    smp (qualquer um suportado, ex. resnet18/34) + decoder UNetFormer
    (GeoSeg). Expoe `.encoder`, `.decoder`, `.segmentation_head` com a
    MESMA interface de 3 passos que o resto do pipeline D_Hier ja' usa
    (`feats = backbone.encoder(x); decoded = backbone.decoder(feats);
    logits = backbone.segmentation_head(decoded)`), entao a injecao de
    contexto em `feats[inject_level]` funciona sem nenhuma mudanca.

    `use_aux_head=True` liga a supervisao profunda do GeoSeg original
    (saida extra `aux_logits` durante treino) -- desligado por padrao
    pra manter saida unica; ligar exige ajustar o loop de treino pra usar
    a saida extra (nao fizemos isso neste "spot-check" inicial).
    """

    def __init__(self, encoder_name="resnet18", encoder_weights="imagenet",
                in_channels=3, classes=6, decode_channels=64, dropout=0.1,
                window_size=8, use_aux_head=False):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights,
        )
        self.decoder = UNetFormerDecoder(
            encoder_channels=self.encoder.out_channels, decode_channels=decode_channels,
            dropout=dropout, window_size=window_size, use_aux_head=use_aux_head,
            num_classes=classes,
        )
        self.segmentation_head = UNetFormerSegmentationHead(decode_channels, classes, dropout)

    def forward(self, x):
        feats = self.encoder(x)
        decoded = self.decoder(feats)
        return self.segmentation_head(decoded)
