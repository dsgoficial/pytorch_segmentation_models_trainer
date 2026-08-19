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

from typing import Tuple, Union

import torch
import torch.nn as nn

from pytorch_segmentation_models_trainer.custom_models.transformer_adapters import (
    ModelOutputAdapter,
)


class HuggingFaceSegmentationWrapper(nn.Module):
    """
    Wraps any HuggingFace ``AutoModelForSemanticSegmentation`` so that it
    produces a plain ``(B, num_labels, H, W)`` float tensor compatible with
    the rest of this framework.

    Key behaviours
    --------------
    * Loads the model via ``transformers.AutoModelForSemanticSegmentation``
      using ``from_pretrained`` (supports both Hub model IDs and local paths).
    * Bypasses the HuggingFace ``AutoImageProcessor`` – the pipeline's
      existing Albumentations augmentations handle normalisation.  Users must
      ensure the correct ``normalize_mean`` / ``normalize_std`` are set in
      the dataset config to match the pretrained model's expectations.
    * Output logits (typically at 1/4 input resolution for Segformer) are
      upsampled back to the input spatial size via bilinear interpolation.

    Args:
        pretrained_model_name_or_path: HuggingFace Hub model ID or local
            directory, e.g. ``"nvidia/segformer-b2-finetuned-ade-512-512"``.
        num_labels: Number of output segmentation classes.  When this differs
            from the pretrained checkpoint's head, pass
            ``ignore_mismatched_sizes=True`` to reinitialise the head.
        ignore_mismatched_sizes: Passed through to ``from_pretrained``; set
            to ``True`` when fine-tuning with a different number of classes.
        output_size: Target spatial size ``(H, W)`` for the final output.
            Defaults to ``"input"`` (upsample to match the input resolution).
        interpolation_mode: Interpolation used during upsampling.
        from_pretrained_kwargs: Any extra keyword arguments forwarded to
            ``AutoModelForSemanticSegmentation.from_pretrained``.

    Example YAML config::

        model:
          _target_: pytorch_segmentation_models_trainer.custom_models.huggingface_models.HuggingFaceSegmentationWrapper
          pretrained_model_name_or_path: nvidia/segformer-b2-finetuned-ade-512-512
          num_labels: 2
          ignore_mismatched_sizes: true
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_labels: int,
        ignore_mismatched_sizes: bool = False,
        output_size: Union[str, Tuple[int, int]] = "input",
        interpolation_mode: str = "bilinear",
        **from_pretrained_kwargs,
    ) -> None:
        super().__init__()

        try:
            from transformers import AutoModelForSemanticSegmentation
        except ImportError as e:
            raise ImportError(
                "The 'transformers' package is required to use "
                "HuggingFaceSegmentationWrapper.  "
                "Install it with:  pip install transformers"
            ) from e

        hf_model = AutoModelForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            **from_pretrained_kwargs,
        )

        self.model = ModelOutputAdapter(
            model=hf_model,
            output_size=output_size,
            interpolation_mode=interpolation_mode,
        )
        self.num_labels = num_labels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
