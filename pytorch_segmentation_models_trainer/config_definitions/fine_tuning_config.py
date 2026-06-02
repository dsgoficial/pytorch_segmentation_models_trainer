# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2024-01-01
        copyright            : (C) 2024 by Philipe Borba - Cartographic Engineer
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
 ****
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoraAdapterConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning.

    Attributes:
        r: LoRA rank – the inner dimension of the low-rank matrices.
            Lower values → fewer trainable parameters; higher values → more
            expressive adapters.  Typical range: 4–64.
        lora_alpha: Scaling factor applied to the LoRA output
            (output = base + (lora_alpha / r) * lora_output).
        lora_dropout: Dropout probability applied to the LoRA matrices.
        bias: Which biases to train alongside LoRA.
            ``"none"`` trains no biases, ``"all"`` trains all biases,
            ``"lora_only"`` trains only the biases of LoRA layers.
        target_modules: Names of the ``nn.Linear`` sub-modules to replace
            with LoRA variants.  Leave empty for auto-detection based on
            common attention-layer naming conventions (query, key, value,
            qkv, q_proj, …).

    Example::

        fine_tuning:
          strategy: lora
          lora_config:
            r: 16
            lora_alpha: 32
            lora_dropout: 0.1
            bias: none
            target_modules:
              - query
              - value
    """

    r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    bias: str = "none"
    target_modules: List[str] = field(default_factory=list)


@dataclass
class FineTuningConfig:
    """Top-level fine-tuning configuration.

    Attributes:
        strategy: Fine-tuning strategy.  One of:

            * ``"full"`` – train all parameters (default).
            * ``"freeze_backbone"`` – freeze encoder/backbone, train only
              the decoder and segmentation head.
            * ``"linear_probe"`` – freeze everything, train only the final
              classification head.
            * ``"lora"`` – apply PEFT LoRA adapters to attention layers;
              requires ``lora_config`` to be set and ``peft`` to be
              installed.

        lora_config: Required when ``strategy="lora"``.
        frozen_modules: Sub-strings used to identify modules to freeze
            in ``freeze_backbone`` and ``linear_probe`` strategies.
            Defaults to common backbone/encoder names.
        trainable_modules: Sub-strings used to identify modules to keep
            trainable.  Defaults to common decoder/head names.

    Example – freeze backbone::

        fine_tuning:
          strategy: freeze_backbone
          frozen_modules: [encoder, backbone]
          trainable_modules: [decoder, head, segmentation_head]

    Example – LoRA::

        fine_tuning:
          strategy: lora
          lora_config:
            r: 16
            lora_alpha: 32
            target_modules: [query, value]
    """

    strategy: str = "full"
    lora_config: Optional[LoraAdapterConfig] = None
    frozen_modules: List[str] = field(default_factory=lambda: ["encoder", "backbone"])
    trainable_modules: List[str] = field(
        default_factory=lambda: ["decoder", "head", "segmentation_head", "neck"]
    )
