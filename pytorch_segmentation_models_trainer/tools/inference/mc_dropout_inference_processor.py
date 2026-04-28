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

Monte Carlo Dropout inference processor.

Extends ``MultiClassInferenceProcessor`` (sliding-window tiled inference) to
run ``n_samples`` stochastic forward passes per tile by keeping Dropout layers
active at test time.  The mean of the T softmax probability maps is used for
the final class prediction; their disagreement is exported as a per-pixel
uncertainty map when ``export_uncertainty_map=True``.

Training is completely unchanged — this processor is only used at inference
time.

Key differences from ``MultiClassInferenceProcessor``:
* ``predict_and_merge`` runs n_samples stochastic passes instead of one
  deterministic pass.
* ``export_uncertainty_map=True`` exports an additional float32 GeoTIFF
  (``{stem}_mc_uncertainty.tif``) alongside the segmentation output.

Usage (Hydra config)::

    inference_processor:
      _target_: pytorch_segmentation_models_trainer.tools.inference.mc_dropout_inference_processor.MCDropoutInferenceProcessor
      n_samples: 10
      uncertainty_mode: entropy   # "entropy" or "mutual_information"
      export_uncertainty_map: false
      num_classes: 5
      model_input_shape: [512, 512]
      step_shape: [256, 256]

Reference:
    Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation.
    ICML 2016. https://arxiv.org/abs/1506.02142
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from pytorch_toolbelt.utils.torch_utils import to_numpy
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from pytorch_segmentation_models_trainer.tools.inference.inference_processors import (
    MultiClassInferenceProcessor,
)
from pytorch_segmentation_models_trainer.utils.mc_dropout_utils import (
    compute_uncertainty,
    enable_mc_dropout,
    warn_if_no_dropout,
)

logger = logging.getLogger(__name__)


class MCDropoutInferenceProcessor(MultiClassInferenceProcessor):
    """Sliding-window inference with Monte Carlo Dropout uncertainty estimation.

    Keeps Dropout layers active during inference and runs ``n_samples``
    stochastic forward passes per tile.  The mean softmax probability across
    passes is used for argmax class prediction.  When
    ``export_uncertainty_map=True``, a per-pixel uncertainty map is computed
    from the T samples and saved as a single-band float32 GeoTIFF.

    The model must have ``Dropout`` / ``Dropout2d`` layers configured.  For
    SMP models, set ``decoder_dropout`` in the model config.  A
    ``UserWarning`` is emitted at construction time if no dropout layers are
    found (all samples will be identical in that case).

    .. note::
        Striped inference (used automatically for very large images) does not
        support the uncertainty map.  When a large image is detected the
        processor falls back to standard tiled inference and logs a warning.
        Reduce ``model_input_shape`` or use a machine with more VRAM if you
        encounter OOM errors on large images.

    Args:
        model: Segmentation model with Dropout layers.
        device: Torch device string (``"cuda"`` or ``"cpu"``).
        batch_size: Tile batch size.
        export_strategy: Export strategy for the segmentation GeoTIFF.
        n_samples: Number of stochastic forward passes per tile (default 10).
        uncertainty_mode: ``"entropy"`` (predictive entropy, captures total
            uncertainty) or ``"mutual_information"`` (BALD, captures epistemic
            uncertainty only).  Default ``"entropy"``.
        export_uncertainty_map: Whether to compute and export the uncertainty
            raster.  Default ``False`` — only the mean prediction is computed.
        All remaining kwargs are forwarded to ``MultiClassInferenceProcessor``.

    Example YAML::

        inference_processor:
          _target_: ...MCDropoutInferenceProcessor
          n_samples: 10
          uncertainty_mode: entropy
          export_uncertainty_map: true
          num_classes: 5
          model_input_shape: [512, 512]
          step_shape: [256, 256]
    """

    def __init__(
        self,
        model,
        device,
        batch_size,
        export_strategy,
        n_samples: int = 10,
        uncertainty_mode: str = "entropy",
        export_uncertainty_map: bool = False,
        **kwargs,
    ):
        # tta_mode is incompatible with MC Dropout's predict_and_merge override
        if kwargs.get("tta_mode") is not None:
            logger.warning(
                "MCDropoutInferenceProcessor: tta_mode='%s' is ignored. "
                "Use MultiClassInferenceProcessor with export_uncertainty_map=True "
                "for TTA-based uncertainty.",
                kwargs["tta_mode"],
            )
            kwargs.pop("tta_mode")

        super().__init__(
            model=model,
            device=device,
            batch_size=batch_size,
            export_strategy=export_strategy,
            uncertainty_mode=uncertainty_mode,
            export_uncertainty_map=export_uncertainty_map,
            **kwargs,
        )
        self.n_samples = n_samples
        warn_if_no_dropout(model)

    # ------------------------------------------------------------------
    # TileMerger overrides
    # ------------------------------------------------------------------

    def get_merger_dict(self, tiler: ImageSlicer) -> Dict[str, TileMerger]:
        """Create mergers for seg probabilities and (optionally) uncertainty."""
        mergers = {
            "seg": TileMerger(
                tiler.target_shape, self.num_classes, tiler.weight, device=self.device
            )
        }
        if self.export_uncertainty_map:
            mergers["uncertainty"] = TileMerger(
                tiler.target_shape, 1, tiler.weight, device=self.device
            )
        return mergers

    # ------------------------------------------------------------------
    # Core inference override
    # ------------------------------------------------------------------

    def predict_and_merge(
        self,
        tiles: List[np.ndarray],
        tiler: ImageSlicer,
        merger_dict: Dict[str, TileMerger],
    ) -> None:
        """Run n_samples MC Dropout forward passes per tile batch and merge.

        For each tile batch:

        1. Enables Dropout layers (``enable_mc_dropout``).
        2. Runs ``n_samples`` forward passes — each uses a different dropout mask.
        3. Averages the softmax probabilities → feeds to the ``"seg"`` merger.
        4. If ``export_uncertainty_map=True``, computes uncertainty from all T
           samples and feeds to the ``"uncertainty"`` merger.

        Note: uncertainty tensors are only allocated when
        ``export_uncertainty_map=True``, so the no-uncertainty path has no
        extra memory cost.
        """
        with torch.no_grad():
            for tiles_batch, coords_batch in DataLoader(
                list(zip(tiles, tiler.crops)),
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=0,
            ):
                tiles_batch = tiles_batch.float().to(self.device)

                # Enable dropout once per tile batch (stays active for all samples)
                enable_mc_dropout(self.model)

                probs_sum = None
                samples_list = [] if self.export_uncertainty_map else None

                for _ in range(self.n_samples):
                    with autocast():
                        logits = self.model(tiles_batch).float()
                    probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
                    del logits
                    probs_sum = probs if probs_sum is None else probs_sum + probs
                    if samples_list is not None:
                        samples_list.append(probs.detach())
                    else:
                        del probs

                p_mean = probs_sum / self.n_samples  # [B, C, H, W]
                del probs_sum

                merger_dict["seg"].integrate_batch(p_mean, coords_batch)
                del p_mean

                if self.export_uncertainty_map and samples_list is not None:
                    samples_tensor = torch.stack(samples_list)  # [T, B, C, H, W]
                    del samples_list
                    unc = compute_uncertainty(samples_tensor, self.uncertainty_mode)
                    del samples_tensor
                    merger_dict["uncertainty"].integrate_batch(unc, coords_batch)
                    del unc

                del tiles_batch

    # ------------------------------------------------------------------
    # Process override: skip striped inference path
    # ------------------------------------------------------------------

    def process(
        self,
        image_path: str,
        threshold: float = 0.5,
        save_inference_output: bool = True,
        polygonizer=None,
        restore_geo_transform: bool = True,
        **kwargs: Optional[Any],
    ):
        """Process a single image with MC Dropout inference.

        Striped inference (used automatically for very large images by the
        parent class) is not supported with MC Dropout because the striped
        path uses a single ``TileMerger`` and cannot merge the uncertainty
        map.  Large images are processed with standard tiled inference instead;
        a warning is logged if the image exceeds
        ``striped_threshold_pixels``.

        Args:
            image_path: Path to the source GeoTIFF.
            threshold: Binarisation threshold (passed through but unused for
                multi-class output).
            save_inference_output: If ``True``, saves the segmentation and
                (when ``export_uncertainty_map=True``) the uncertainty raster.
            polygonizer: Optional polygonizer to run on the inference output.
            restore_geo_transform: If ``True``, preserves the source CRS and
                transform in the output.

        Returns:
            ``defaultdict(list)`` with ``"inference"`` and optionally
            ``"polygons"`` keys.
        """
        if self._is_large_image(image_path):
            logger.warning(
                "MCDropoutInferenceProcessor: large image detected (%s). "
                "Striped inference is not supported with MC Dropout. "
                "Processing with standard tiled inference — consider reducing "
                "model_input_shape if memory is limited.",
                image_path,
            )

        # Bypass the striped inference routing in the parent's process()
        image, profile = self.read_image_and_profile(
            image_path, restore_geo_transform=restore_geo_transform
        )
        inference = self.make_inference(image, **kwargs)

        output_dict = defaultdict(list)
        _polygonizer = self.polygonizer if polygonizer is None else polygonizer
        if _polygonizer is not None:
            output_dict["polygons"] += self.process_polygonizer(
                _polygonizer,
                inference,
                profile,
                image_name=Path(image_path).stem
                if self.group_output_by_image_basename
                else None,
            )
        if save_inference_output:
            self._save_mc_inference(image_path, threshold, profile, inference, output_dict)
        if kwargs.get("output_inferences"):
            output_dict["inference_output"].append(inference)
        return output_dict

    def _save_mc_inference(self, image_path, threshold, profile, inference, output_dict):
        """Like save_inference but uses '_mc_uncertainty' suffix for the uncertainty raster."""
        if self.export_uncertainty_map and "uncertainty" in inference:
            self._save_uncertainty_raster(
                uncertainty_map=inference["uncertainty"],
                reference_profile=profile,
                image_path=image_path,
                suffix="_mc_uncertainty",
            )
        profile = dict(profile)
        profile["count"] = 1
        profile["dtype"] = "uint8"
        profile["input_name"] = Path(image_path).stem
        profile["suffix"] = Path(image_path).suffix
        if self.export_strategy is not None:
            output_dict["inference"].append(
                self.export_strategy.save_inference(
                    {"seg": inference.get("seg", inference)}, profile
                )
            )
