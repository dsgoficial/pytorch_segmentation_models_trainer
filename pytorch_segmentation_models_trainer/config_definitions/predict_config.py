# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2025-11-06
        copyright            : (C) 2025 by Philipe Borba
        email                : philipeborba at gmail dot com
 ***************************************************************************/
/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

from dataclasses import dataclass, field
from typing import Any, Optional, List
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class InferenceImageReaderConfig:
    """
    Image reader configuration for inference.
    Used by predict.py via get_images().
    """

    _target_: str = MISSING
    input_csv_path: str = MISSING
    key: str = "image"
    root_dir: Optional[str] = None
    n_first_rows_to_read: Optional[int] = None


@dataclass
class InferenceProcessorConfig:
    """
    Single-image inference processor configuration.
    Used by predict.py via instantiate_inference_processor().

    Note: model, polygonizer, export_strategy, device, batch_size and
    mask_bands are injected automatically by the calling code and do not
    need to appear in the YAML.

    Test Time Augmentation (TTA)
    ----------------------------
    Two TTA interfaces are available depending on the processor class:

    **MultiClassInferenceProcessor** — use ``tta_mode``:

        ``tta_mode: "d4"``   — all 8 dihedral symmetries (4 rotations × 2 flips)
        ``tta_mode: "flip"`` — horizontal + vertical flip (4 passes)
        ``tta_mode: null``   — no TTA (default)

    **SingleImageInfereceProcessor / SingleImageFromFrameFieldProcessor** — use ``use_tta``:

    When ``use_tta: true``, each tile is run through the model once per
    augmentation listed in ``tta_augmentations``. The predictions are
    de-augmented (reversed) and averaged before being handed to the
    TileMerger.

    Available augmentations (D4 dihedral group — 8 symmetries of a square):

    +------------------+-----------------------------------------------------+
    | Name             | Transformation                                      |
    +==================+=====================================================+
    | ``rot0``         | Identity — original image, no transformation        |
    | ``rot90``        | 90° counter-clockwise rotation                      |
    | ``rot180``       | 180° rotation                                       |
    | ``rot270``       | 270° counter-clockwise rotation                     |
    | ``flip_h``       | Horizontal flip (left-right mirror)                 |
    | ``flip_v``       | Vertical flip (up-down mirror)                      |
    | ``flip_h_rot90`` | Horizontal flip + 90° counter-clockwise rotation    |
    | ``flip_v_rot90`` | Vertical flip + 90° counter-clockwise rotation      |
    +------------------+-----------------------------------------------------+

    Confidence maps (MultiClassInferenceProcessor only)
    ---------------------------------------------------
    ``confidence_mode: "basic"`` — saves max-probability map per pixel.
    ``confidence_mode: "full"``  — saves max_prob, entropy, and margin maps.

    Striped inference (MultiClassInferenceProcessor only)
    -----------------------------------------------------
    Images larger than ``striped_threshold_pixels`` are processed in horizontal
    stripes of height ``stripe_height`` to avoid OOM on large/BigTIFF inputs.

    YAML example — MultiClassInferenceProcessor with D4 TTA::

        inference_processor:
          _target_: ...MultiClassInferenceProcessor
          model_input_shape: [512, 512]
          step_shape: [256, 256]
          num_classes: 5
          tta_mode: d4
          tile_weight: gaussian

    YAML example — SingleImageInfereceProcessor with rotation TTA::

        inference_processor:
          _target_: ...SingleImageInfereceProcessor
          model_input_shape: [448, 448]
          step_shape: [224, 224]
          use_tta: true
          tta_augmentations:
            - rot0
            - rot90
            - rot180
            - rot270
    """

    _target_: str = MISSING
    model_input_shape: Optional[List[int]] = None
    step_shape: Optional[List[int]] = None

    # ── TTA for SingleImageInfereceProcessor / SingleImageFromFrameFieldProcessor ──
    use_tta: bool = False
    # Default: four 90° rotations.
    # Add flip_h, flip_v, flip_h_rot90, flip_v_rot90 for the full D4 group.
    tta_augmentations: List[str] = field(
        default_factory=lambda: [
            "rot0",
            "rot90",
            "rot180",
            "rot270",
        ]
    )

    # ── TTA for MultiClassInferenceProcessor ──────────────────────────────────
    # None = no TTA, "d4" = 8 dihedral transforms, "flip" = 4 flip transforms
    tta_mode: Optional[str] = None

    # ── MultiClassInferenceProcessor extras ───────────────────────────────────
    # Tile weighting: "mean" | "pyramid" | "gaussian"
    tile_weight: str = "mean"
    # Confidence maps: None | "basic" | "full"
    confidence_mode: Optional[str] = None
    # Striped inference for large images
    striped_threshold_pixels: int = 50_000_000
    stripe_height: int = 4096
    output_probs_dir: Optional[str] = None


@dataclass
class ExportStrategyConfig:
    """
    Inference export strategy configuration.
    Used by predict.py (optional).
    """

    _target_: str = MISSING
    # Remaining fields depend on the concrete strategy type


@dataclass
class PolygonizerConfig:
    """
    Polygonizer configuration.
    Used by predict.py via instantiate_polygonizer().
    """

    _target_: str = MISSING
    config: Optional[Any] = None
    data_writer: Optional[Any] = None


@dataclass
class PredictSingleImageConfig:
    """
    Full configuration for predict.py (single image processor).

    This config is used by the predict.py script, which processes images
    one at a time using inference_processor.process().

    Fields injected automatically by the code:
    - inference_processor.model          (from checkpoint)
    - inference_processor.polygonizer    (from polygonizer config)
    - inference_processor.export_strategy (from export_strategy config)
    - inference_processor.device         (from device)
    - inference_processor.batch_size     (from hyperparameters.batch_size)
    - inference_processor.mask_bands     (from seg_params)
    """

    # Checkpoint and model
    checkpoint_path: str = MISSING
    device: str = "cuda:0"

    # Inherited from train config (pl_model, hyperparameters, seg_params)
    pl_model: Any = MISSING
    hyperparameters: Any = MISSING
    seg_params: Optional[Any] = None

    # Image reader
    inference_image_reader: InferenceImageReaderConfig = field(
        default_factory=InferenceImageReaderConfig
    )

    # Inference processor
    inference_processor: InferenceProcessorConfig = field(
        default_factory=InferenceProcessorConfig
    )

    # Polygonizer (optional)
    polygonizer: Optional[PolygonizerConfig] = None

    # Export strategy (optional)
    export_strategy: Optional[ExportStrategyConfig] = None

    # Inference parameters
    inference_threshold: float = 0.5
    save_inference: bool = True

    # ── Test Time Augmentation for the model test_step ────────────────────────
    # use_tta / tta_augmentations: applies to SingleImageInfereceProcessor-style
    # augmentation strings (rot0, rot90, flip_h, …).
    # tta_mode: applies to MultiClassInferenceProcessor-style ("d4" or "flip").
    # Only one interface is active at a time; tta_mode takes precedence.
    use_tta: bool = False
    tta_augmentations: List[str] = field(
        default_factory=lambda: [
            "rot0",
            "rot90",
            "rot180",
            "rot270",
        ]
    )
    # None = no TTA, "d4" = 8 dihedral transforms, "flip" = 4 flip transforms
    tta_mode: Optional[str] = None


# Register configurations with Hydra
cs = ConfigStore.instance()
cs.store(name="predict_single_image_config", node=PredictSingleImageConfig)
cs.store(name="inference_image_reader", node=InferenceImageReaderConfig)
cs.store(name="inference_processor_single", node=InferenceProcessorConfig)
cs.store(name="polygonizer", node=PolygonizerConfig)
cs.store(name="export_strategy", node=ExportStrategyConfig)
