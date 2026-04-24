# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-04-24
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

Hydra dataclass configuration for Monte Carlo Dropout inference.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class MCDropoutInferenceProcessorConfig:
    """Configuration for ``MCDropoutInferenceProcessor``.

    The model must have ``Dropout`` / ``Dropout2d`` layers configured for MC
    Dropout to produce meaningful uncertainty estimates.  For SMP models set
    ``decoder_dropout`` in the model config (e.g. ``decoder_dropout: 0.2``).

    Args:
        n_samples: Number of stochastic forward passes per tile.  Higher
            values reduce variance in the uncertainty estimate but increase
            inference time linearly.  Typical range: 10–50.
        uncertainty_mode: How to aggregate the T samples into a scalar
            uncertainty per pixel.
            ``"entropy"`` — predictive entropy of the mean distribution;
            captures total uncertainty (epistemic + aleatoric).
            ``"mutual_information"`` — BALD; isolates epistemic uncertainty
            by measuring disagreement between samples.
        export_uncertainty_map: Whether to compute and save a per-pixel
            uncertainty raster alongside the segmentation output.  Defaults
            to ``False`` to avoid extra computation when not needed.
        num_classes: Number of segmentation output classes.
        model_input_shape: ``[H, W]`` tile size fed to the model.
        step_shape: ``[H, W]`` sliding-window step.  Smaller than
            ``model_input_shape`` produces overlapping tiles (recommended).
        output_uncertainty_dir: Directory for the uncertainty GeoTIFFs.
            If ``None``, files are placed in the same directory as the
            segmentation output.

    Example YAML::

        inference_processor:
          _target_: pytorch_segmentation_models_trainer.tools.inference.mc_dropout_inference_processor.MCDropoutInferenceProcessor
          n_samples: 10
          uncertainty_mode: entropy
          export_uncertainty_map: false
          num_classes: 5
          model_input_shape: [512, 512]
          step_shape: [256, 256]
    """

    _target_: str = (
        "pytorch_segmentation_models_trainer.tools.inference"
        ".mc_dropout_inference_processor.MCDropoutInferenceProcessor"
    )
    n_samples: int = 10
    uncertainty_mode: str = "entropy"
    export_uncertainty_map: bool = False
    num_classes: int = MISSING
    model_input_shape: List[int] = field(default_factory=lambda: [512, 512])
    step_shape: Optional[List[int]] = field(default_factory=lambda: [256, 256])
    output_uncertainty_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Register with Hydra ConfigStore
# ---------------------------------------------------------------------------

cs = ConfigStore.instance()
cs.store(
    group="inference_processor",
    name="mc_dropout",
    node=MCDropoutInferenceProcessorConfig,
)
