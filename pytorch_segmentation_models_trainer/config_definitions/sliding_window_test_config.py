# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-05-15
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

Hydra dataclass configuration for sliding-window test-phase evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class SlidingWindowTestConfig:
    """Parameters for full-image sliding-window inference during ``trainer.test()``.

    Used when ``use_sliding_window_test: true`` is set in the model config.
    The ``Model`` class reads ``cfg.sliding_window_test`` to build a
    :class:`~pytorch_segmentation_models_trainer.tools.inference.sliding_window.SlidingWindowCore`
    instance.

    Args:
        tile_size: ``[H, W]`` size of each tile fed to the model.
        tile_step: ``[H, W]`` sliding-window step.  Set smaller than
            ``tile_size`` to obtain overlapping tiles (strongly recommended).
        num_classes: Number of segmentation output classes.
        batch_size: Number of tiles per inner-loop forward-pass batch.
        tile_weight: Blending weight for overlapping tiles.
            ``"mean"`` — uniform average.
            ``"pyramid"`` — linearly increasing weight toward tile centre.
            ``"gaussian"`` — nnU-Net-style Gaussian importance map; reduces
            border artefacts from CNN context truncation.
        tta_augmentations: List of TTA augmentation names from
            ``tools/tta/tta.py``.  Use an empty list to disable TTA.
            Supported values: ``rot0``, ``rot90``, ``rot180``, ``rot270``,
            ``flip_h``, ``flip_v``, ``flip_h_rot90``, ``flip_v_rot90``.
        compute_tta_uncertainty: When ``true`` and at least two TTA
            augmentations are requested, compute the per-pixel std of softmax
            probabilities across TTA passes and store it in the output's
            ``tta_uncertainty`` field.
        use_mc_dropout: Enable Monte Carlo Dropout.  The model must contain
            ``Dropout`` / ``Dropout2d`` layers; a warning is emitted otherwise.
        n_mc_samples: Number of stochastic forward passes for MC Dropout.
        compute_mc_uncertainty: When ``true`` (and ``use_mc_dropout`` is also
            ``true``), compute per-pixel uncertainty from the MC samples.
        uncertainty_mode: Uncertainty aggregation for MC Dropout.
            ``"entropy"`` — predictive entropy (total uncertainty).
            ``"mutual_information"`` — BALD / epistemic uncertainty.
        output_dir: Directory for saving prediction and uncertainty GeoTIFFs.
            When ``None`` (default) no files are written to disk.

    YAML example — TTA + Gaussian tile weight::

        use_sliding_window_test: true
        sliding_window_test:
          tile_size: [512, 512]
          tile_step: [256, 256]
          num_classes: 5
          batch_size: 4
          tile_weight: gaussian
          tta_augmentations:
            - rot0
            - rot90
            - rot180
            - rot270
          compute_tta_uncertainty: false

    YAML example — MC Dropout only::

        use_sliding_window_test: true
        sliding_window_test:
          tile_size: [512, 512]
          tile_step: [256, 256]
          num_classes: 2
          use_mc_dropout: true
          n_mc_samples: 20
          compute_mc_uncertainty: true
          uncertainty_mode: entropy
          output_dir: /path/to/predictions

    YAML example — TTA + MC Dropout combined::

        use_sliding_window_test: true
        sliding_window_test:
          tile_size: [512, 512]
          tile_step: [256, 256]
          num_classes: 5
          tta_augmentations: [rot0, rot90, rot180, rot270]
          compute_tta_uncertainty: true
          use_mc_dropout: true
          n_mc_samples: 10
          compute_mc_uncertainty: true
          uncertainty_mode: mutual_information
    """

    tile_size: List[int] = field(default_factory=lambda: [512, 512])
    tile_step: List[int] = field(default_factory=lambda: [256, 256])
    num_classes: int = MISSING
    batch_size: int = 4
    tile_weight: str = "gaussian"
    tta_augmentations: List[str] = field(default_factory=list)
    compute_tta_uncertainty: bool = False
    use_mc_dropout: bool = False
    n_mc_samples: int = 10
    compute_mc_uncertainty: bool = False
    uncertainty_mode: str = "entropy"
    output_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Register with Hydra ConfigStore
# ---------------------------------------------------------------------------

cs = ConfigStore.instance()
cs.store(
    group="sliding_window_test",
    name="default",
    node=SlidingWindowTestConfig,
)
