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

"""
Fine-tuning strategies for segmentation models
===============================================

Provides four strategies, all applied by :func:`apply_fine_tuning_strategy`:

``"full"``
    All parameters are trainable (default, no-op).

``"freeze_backbone"``
    Freeze everything except the decoder / segmentation head.  Works for
    SMP, HuggingFace, TerraTorch, and plain timm wrappers by matching module
    names that contain ``"decoder"`` or ``"head"`` (configurable via
    ``trainable_module_names``).

``"linear_probe"``
    Freeze the entire model then unfreeze only the final layer whose name
    contains ``"head"`` or ``"segmentation_head"`` (configurable).

``"lora"``
    Apply PEFT LoRA adapters to the attention layers of the model.
    Requires ``peft`` to be installed (``pip install peft``).

All strategies log a parameter count summary via Python's ``logging`` module.
"""

import logging
from typing import Any, Dict, List

import torch.nn as nn

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_BACKBONE_NAMES = ("encoder", "backbone", "model.encoder", "model.backbone")
_DEFAULT_TRAINABLE_NAMES = ("decoder", "head", "segmentation_head", "neck")


def freeze_modules_by_name(
    model: nn.Module,
    frozen_names: tuple = _DEFAULT_BACKBONE_NAMES,
    trainable_names: tuple = _DEFAULT_TRAINABLE_NAMES,
) -> nn.Module:
    """
    Freeze parameters whose module name contains any string in *frozen_names*,
    then unfreeze those whose name contains any string in *trainable_names*.

    This two-pass approach means ``trainable_names`` always wins, so a module
    like ``"backbone.decoder"`` would be unfrozen even if ``"backbone"`` is in
    *frozen_names*.

    Args:
        model: The model to modify in-place.
        frozen_names: Sub-strings that mark modules to freeze.
        trainable_names: Sub-strings that mark modules to keep trainable.

    Returns:
        The modified model (same object, in-place operation).
    """
    # Pass 1: freeze everything that matches frozen_names
    for name, param in model.named_parameters():
        if any(fn in name for fn in frozen_names):
            param.requires_grad = False

    # Pass 2: unfreeze anything that matches trainable_names
    for name, param in model.named_parameters():
        if any(tn in name for tn in trainable_names):
            param.requires_grad = True

    return model


def get_trainable_parameter_count(model: nn.Module) -> Dict[str, int]:
    """Return a dict with ``trainable``, ``frozen``, and ``total`` counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "frozen": total - trainable, "total": total}


def _log_param_summary(model: nn.Module, strategy: str) -> None:
    counts = get_trainable_parameter_count(model)
    pct = 100.0 * counts["trainable"] / max(counts["total"], 1)
    logger.info(
        "Fine-tuning strategy='%s' | trainable=%s / %s (%.1f%%)",
        strategy,
        f"{counts['trainable']:,}",
        f"{counts['total']:,}",
        pct,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Strategy implementations
# ──────────────────────────────────────────────────────────────────────────────


def _apply_full(model: nn.Module, cfg: Any) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = True
    return model


def _apply_freeze_backbone(model: nn.Module, cfg: Any) -> nn.Module:
    frozen = tuple(getattr(cfg, "frozen_modules", None) or _DEFAULT_BACKBONE_NAMES)
    trainable = tuple(
        getattr(cfg, "trainable_modules", None) or _DEFAULT_TRAINABLE_NAMES
    )
    # Start with all params frozen
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze trainable parts
    for name, param in model.named_parameters():
        if any(tn in name for tn in trainable):
            param.requires_grad = True
        # If no backbone names match at all (e.g. a model with unusual naming),
        # also unfreeze anything NOT matching frozen names
        elif not any(fn in name for fn in frozen):
            param.requires_grad = True
    return model


def _apply_linear_probe(model: nn.Module, cfg: Any) -> nn.Module:
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    trainable_keywords = tuple(
        getattr(cfg, "trainable_modules", None)
        or ("head", "segmentation_head", "classifier")
    )
    unfrozen_any = False
    for name, param in model.named_parameters():
        if any(kw in name for kw in trainable_keywords):
            param.requires_grad = True
            unfrozen_any = True

    if not unfrozen_any:
        logger.warning(
            "linear_probe strategy: no parameters matched trainable_modules=%s. "
            "All parameters remain frozen.  Set trainable_modules in fine_tuning "
            "config to match your model's head layer names.",
            trainable_keywords,
        )
    return model


def _apply_lora(model: nn.Module, cfg: Any) -> nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError(
            "The 'peft' package is required for LoRA fine-tuning.  "
            "Install it with:  pip install peft"
        ) from e

    lora_cfg = getattr(cfg, "lora_config", None)
    if lora_cfg is None:
        raise ValueError(
            "strategy='lora' requires a 'lora_config' block in fine_tuning config."
        )

    r = int(getattr(lora_cfg, "r", 16))
    lora_alpha = float(getattr(lora_cfg, "lora_alpha", 32.0))
    lora_dropout = float(getattr(lora_cfg, "lora_dropout", 0.1))
    bias = str(getattr(lora_cfg, "bias", "none"))
    target_modules = list(getattr(lora_cfg, "target_modules", []))

    if not target_modules:
        # Auto-detect: collect names of all nn.Linear layers inside attention blocks
        target_modules = _auto_detect_attention_linears(model)
        if not target_modules:
            raise ValueError(
                "LoRA: could not auto-detect attention linear layers.  "
                "Please set lora_config.target_modules explicitly, e.g. "
                '["query", "value"] for HuggingFace or ["qkv"] for ViT models.'
            )
        logger.info("LoRA: auto-detected target_modules=%s", target_modules)

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
    )

    model = get_peft_model(model, peft_config)
    return model


def _auto_detect_attention_linears(model: nn.Module) -> List[str]:
    """
    Heuristically collect unique short names of ``nn.Linear`` layers that
    appear to belong to attention blocks (name contains common attention
    keywords).
    """
    attention_keywords = (
        "query",
        "key",
        "value",
        "q_proj",
        "k_proj",
        "v_proj",
        "qkv",
        "attn",
    )
    found = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            short = name.split(".")[-1]
            if any(kw in name.lower() for kw in attention_keywords):
                found.add(short)
    return sorted(found)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

_STRATEGY_MAP = {
    "full": _apply_full,
    "freeze_backbone": _apply_freeze_backbone,
    "linear_probe": _apply_linear_probe,
    "lora": _apply_lora,
}


def apply_fine_tuning_strategy(model: nn.Module, cfg: Any) -> nn.Module:
    """
    Apply the fine-tuning strategy described in *cfg* to *model*.

    Args:
        model: The model to modify.
        cfg: An OmegaConf config node (or any object) with at minimum a
            ``strategy`` attribute.  Supported strategies and their config
            keys are documented in the module docstring.

    Returns:
        The (potentially wrapped) model.

    Example config::

        fine_tuning:
          strategy: lora
          lora_config:
            r: 16
            lora_alpha: 32
            lora_dropout: 0.1
            bias: none
            target_modules: [query, value]
    """
    strategy = str(getattr(cfg, "strategy", "full")).lower()

    if strategy not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown fine_tuning strategy='{strategy}'.  "
            f"Choose from: {list(_STRATEGY_MAP.keys())}"
        )

    model = _STRATEGY_MAP[strategy](model, cfg)
    _log_param_summary(model, strategy)
    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA adapter weights into the base model and return a plain
    ``nn.Module`` without PEFT wrappers.  Call this before deployment /
    inference to remove the PEFT overhead.

    Args:
        model: A PEFT-wrapped model (``PeftModel``).

    Returns:
        The merged base model.
    """
    if not is_peft_model(model):
        return model
    model = model.merge_and_unload()
    logger.info("LoRA weights merged and PEFT wrapper removed.")
    return model


def is_peft_model(model: nn.Module) -> bool:
    """Return ``True`` if *model* is wrapped with a PEFT adapter."""
    try:
        from peft import PeftModel

        return isinstance(model, PeftModel)
    except ImportError:
        return False
