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

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelOutputAdapter(nn.Module):
    """
    Wraps any segmentation model and normalises its output to a plain
    ``(B, C, H, W)`` float tensor.

    Handles the three most common non-standard output shapes produced by
    HuggingFace, TerraTorch and plain-timm models:

    * **Dataclass / NamedTuple with ``.logits``** – e.g.
      ``transformers.SegmentationModelOutput``.
    * **Dict** – keys tried in order: ``output_key``, ``"logits"``,
      ``"out"``, ``"pred"``.  Pass ``output_key`` to force a specific key.
    * **Tuple / list** – the first element is used.
    * **Plain tensor** – returned as-is (no copy).

    After extracting the tensor the adapter optionally upsamples it to match
    the spatial size of the input, which is required for models like
    Segformer that output at 1/4 resolution.

    Args:
        model: The underlying ``nn.Module`` to wrap.
        output_key: Force a specific dict key when the model returns a dict.
            Ignored for other output types.
        output_size: Target spatial size ``(H, W)`` for the output tensor.
            Pass ``None`` (default) to skip upsampling, or the string
            ``"input"`` to automatically upsample to the spatial dimensions
            of the *input* tensor passed to ``forward``.
        interpolation_mode: Interpolation mode forwarded to
            ``torch.nn.functional.interpolate``.  Defaults to
            ``"bilinear"``.
        align_corners: Passed to ``F.interpolate`` when ``interpolation_mode``
            is ``"bilinear"`` or ``"bicubic"``.  Defaults to ``False``.
    """

    _DICT_KEY_PRIORITY = ("logits", "out", "pred", "output", "masks")

    def __init__(
        self,
        model: nn.Module,
        output_key: Optional[str] = None,
        output_size: Optional[Union[str, Tuple[int, int]]] = "input",
        interpolation_mode: str = "bilinear",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.output_key = output_key
        self.output_size = output_size
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_tensor(self, raw_output) -> torch.Tensor:
        """Convert any model output type to a plain tensor."""
        if isinstance(raw_output, torch.Tensor):
            return raw_output

        # Dataclass / object with .logits (HuggingFace pattern)
        if hasattr(raw_output, "logits") and isinstance(
            raw_output.logits, torch.Tensor
        ):
            return raw_output.logits

        # Dict-like
        if isinstance(raw_output, dict):
            # Try user-specified key first
            if self.output_key is not None:
                if self.output_key in raw_output:
                    val = raw_output[self.output_key]
                    if isinstance(val, torch.Tensor):
                        return val
                raise KeyError(
                    f"ModelOutputAdapter: output_key '{self.output_key}' not found "
                    f"in model output dict.  Available keys: {list(raw_output.keys())}"
                )
            # Auto-detect key
            for key in self._DICT_KEY_PRIORITY:
                if key in raw_output and isinstance(raw_output[key], torch.Tensor):
                    return raw_output[key]
            raise KeyError(
                f"ModelOutputAdapter: could not find a tensor in the model output "
                f"dict.  Tried keys {self._DICT_KEY_PRIORITY}.  "
                f"Available keys: {list(raw_output.keys())}.  "
                f"Pass output_key='<your_key>' to the adapter."
            )

        # Tuple / list – take first element
        if isinstance(raw_output, (tuple, list)):
            first = raw_output[0]
            if isinstance(first, torch.Tensor):
                return first
            raise TypeError(
                f"ModelOutputAdapter: first element of tuple/list output is "
                f"{type(first).__name__}, expected torch.Tensor."
            )

        raise TypeError(
            f"ModelOutputAdapter: unsupported output type '{type(raw_output).__name__}'. "
            f"Expected torch.Tensor, dict, tuple/list, or an object with a "
            f"'.logits' attribute."
        )

    def _maybe_upsample(
        self, tensor: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Upsample *tensor* if self.output_size demands it."""
        if self.output_size is None:
            return tensor

        if self.output_size == "input":
            target_h, target_w = input_tensor.shape[-2], input_tensor.shape[-1]
        else:
            target_h, target_w = self.output_size

        out_h, out_w = tensor.shape[-2], tensor.shape[-1]
        if out_h == target_h and out_w == target_w:
            return tensor

        kwargs = {}
        if self.interpolation_mode in ("bilinear", "bicubic"):
            kwargs["align_corners"] = self.align_corners

        return F.interpolate(
            tensor.float(),
            size=(target_h, target_w),
            mode=self.interpolation_mode,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.model(x)
        tensor = self._extract_tensor(raw)
        tensor = self._maybe_upsample(tensor, x)
        return tensor
