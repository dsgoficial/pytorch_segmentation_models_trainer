# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-04-14
        git sha              : $Format:%H$
        copyright            : (C) 2026 by Philipe Borba - Cartographic Engineer
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

from __future__ import annotations

import logging
from typing import Dict, List

import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class FeatureExtractorHook:
    """Captures intermediate feature maps from named layers via PyTorch forward hooks.

    This is a **non-module** utility class. It registers PyTorch forward hooks
    on named layers of a model and stores their outputs in a dict. The model
    itself is not modified in any way.

    Only active when a DA method has ``requires_features = True``. Created and
    owned by ``DomainAdaptationModel``; its lifecycle is tied to the model's
    lifetime.

    Usage::

        hook = FeatureExtractorHook(model, ["encoder.layer3", "encoder.layer4"])

        # Run model forward — hooks fire automatically
        output = model(x)

        # Retrieve captured features
        features = hook.get_features()
        # {"encoder.layer3": Tensor(...), "encoder.layer4": Tensor(...)}

        # Clear before next forward pass
        hook.clear()

        # Remove hooks when done (e.g. at end of training)
        hook.remove()

    Args:
        model: The ``nn.Module`` whose layers will be hooked. Typically
            ``DomainAdaptationModel.model``.
        layer_names: List of dot-separated layer name strings, e.g.
            ``["encoder.layer3"]``. Each name is resolved via
            ``_get_layer_by_name`` using ``getattr`` traversal.

    Raises:
        ValueError: If a layer name cannot be resolved on the model.
    """

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self._features: Dict[str, Tensor] = {}
        self._handles: list = []

        for name in layer_names:
            layer = self._get_layer_by_name(model, name)
            handle = layer.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
            logger.debug("FeatureExtractorHook registered on layer '%s'", name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_features(self) -> Dict[str, Tensor]:
        """Return a shallow copy of the currently captured feature tensors.

        Returns:
            Dict mapping layer name → output tensor from the most recent
            forward pass. Empty if ``clear()`` was called and no forward
            has run since.
        """
        return dict(self._features)

    def clear(self) -> None:
        """Discard all stored feature tensors.

        Call this before each forward pass to avoid accumulating stale
        tensors across batches.
        """
        self._features.clear()

    def remove(self) -> None:
        """Deregister all forward hooks from the model.

        Should be called in ``on_fit_end`` or when the hook is no longer
        needed to avoid memory leaks.
        """
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        logger.debug("FeatureExtractorHook: all hooks removed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_hook(self, name: str):
        def hook(module: nn.Module, input, output: Tensor) -> None:
            self._features[name] = output

        return hook

    @staticmethod
    def _get_layer_by_name(model: nn.Module, name: str) -> nn.Module:
        """Resolve a dot-separated layer name to an ``nn.Module``.

        Args:
            model: Root module to traverse.
            name: Dot-separated attribute path, e.g. ``"encoder.layer3"``.

        Returns:
            The resolved sub-module.

        Raises:
            ValueError: If any part of the path does not exist on the model.
        """
        parts = name.split(".")
        current = model
        for part in parts:
            if not hasattr(current, part):
                raise ValueError(
                    f"FeatureExtractorHook: layer '{name}' not found on "
                    f"{type(model).__name__}. Failed at attribute '{part}'."
                )
            current = getattr(current, part)
        if not isinstance(current, nn.Module):
            raise ValueError(
                f"FeatureExtractorHook: '{name}' resolved to "
                f"{type(current).__name__}, expected nn.Module."
            )
        return current
