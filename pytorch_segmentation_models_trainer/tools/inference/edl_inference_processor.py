# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-04-15
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

Evidential Deep Learning inference processor.

Extends ``SingleImageInfereceProcessor`` (sliding-window tiled inference) to:

1. Run the ``EvidentialWrapper`` forward pass and collect both ``probs`` and
   ``uncertainty`` (u = K/S) at tile resolution.
2. Merge tiles back to full image resolution via separate ``TileMerger``
   instances for probabilities and uncertainty.
3. Export a single-band float32 GeoTIFF of the uncertainty map, preserving
   the CRS and transform of the source image via ``rasterio``.
4. Optionally export per-class Dirichlet alpha parameters.

Typical CLI usage (``predict.py`` detects EDL automatically)::

    python -m pytorch_segmentation_models_trainer.predict \\
        --config-name predict \\
        model_path=/path/to/edl_model.ckpt \\
        image_path=/path/to/image.tif \\
        output_path=/results/probs.tif \\
        output_uncertainty_path=/results/uncertainty.tif
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import albumentations as A
import cv2
import numpy as np
import rasterio
import rasterio.transform
import torch
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from pytorch_toolbelt.utils.torch_utils import image_to_tensor, to_numpy
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from pytorch_segmentation_models_trainer.tools.inference.inference_processors import (
    SingleImageInfereceProcessor,
)

logger = logging.getLogger(__name__)


class EvidentialInferenceProcessor(SingleImageInfereceProcessor):
    """Sliding-window inference processor for EvidentialWrapper models.

    Produces two GeoTIFF outputs per image:
    * ``output_path``            — predicted class probabilities, one band per class.
    * ``output_uncertainty_path`` — epistemic uncertainty u=K/S, single float32 band,
                                    values in (0, 1].

    The processor is a drop-in replacement for ``SingleImageInfereceProcessor``:
    it uses the same tiling/merging infrastructure so tile size, step size, TTA,
    and batch size are all configured identically.

    Args:
        model:                  An ``EvidentialWrapper`` (or any model whose
                                forward returns a dict with ``"probs"`` and
                                ``"uncertainty"`` keys).
        device:                 Torch device string (``"cuda"`` or ``"cpu"``).
        batch_size:             Tile batch size for inference.
        export_strategy:        Export strategy for the probability raster
                                (same as the parent class).
        output_uncertainty_path: Path for the single-band uncertainty GeoTIFF.
                                If ``None``, uncertainty is not saved.
        num_classes:            Number of segmentation classes K.
        export_alpha:           If ``True``, also export K-band alpha GeoTIFF.
        All remaining kwargs are forwarded to ``SingleImageInfereceProcessor``.
    """

    def __init__(
        self,
        model,
        device,
        batch_size,
        export_strategy,
        output_uncertainty_path: Optional[str] = None,
        num_classes: int = 2,
        export_alpha: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            device=device,
            batch_size=batch_size,
            export_strategy=export_strategy,
            mask_bands=num_classes,
            **kwargs,
        )
        self.output_uncertainty_path = (
            Path(output_uncertainty_path) if output_uncertainty_path else None
        )
        self.num_classes = num_classes
        self.export_alpha = export_alpha

    # ------------------------------------------------------------------
    # TileMerger overrides
    # ------------------------------------------------------------------

    def get_merger_dict(self, tiler: ImageSlicer) -> Dict[str, TileMerger]:
        """Create separate mergers for probabilities and uncertainty."""
        mergers = {
            "probs": TileMerger(
                tiler.target_shape,
                self.num_classes,  # K bands
                tiler.weight,
                device=self.device,
            ),
            "uncertainty": TileMerger(
                tiler.target_shape,
                1,  # single band
                tiler.weight,
                device=self.device,
            ),
        }
        if self.export_alpha:
            mergers["alpha"] = TileMerger(
                tiler.target_shape,
                self.num_classes,
                tiler.weight,
                device=self.device,
            )
        return mergers

    def integrate_batch(
        self,
        pred_batch,
        coords_batch,
        merger_dict: Dict[str, TileMerger],
    ) -> None:
        """Integrate a batch of tiles into the tile mergers.

        Accepts:
        * dict with ``"probs"`` and ``"uncertainty"`` keys (EvidentialWrapper output)
        * plain tensor (falls back to parent behaviour, treated as probabilities)
        """
        if isinstance(pred_batch, dict) and "probs" in pred_batch:
            merger_dict["probs"].integrate_batch(pred_batch["probs"], coords_batch)
            merger_dict["uncertainty"].integrate_batch(
                pred_batch["uncertainty"], coords_batch
            )
            if self.export_alpha and "alpha" in merger_dict and "alpha" in pred_batch:
                merger_dict["alpha"].integrate_batch(pred_batch["alpha"], coords_batch)
        else:
            # Fallback: treat as probability tensor
            merger_dict["probs"].integrate_batch(pred_batch, coords_batch)
            # Uncertainty not available — fill with zeros
            B = pred_batch.shape[0]
            dummy_unc = torch.zeros(
                B,
                1,
                pred_batch.shape[2],
                pred_batch.shape[3],
                device=pred_batch.device,
            )
            merger_dict["uncertainty"].integrate_batch(dummy_unc, coords_batch)

    def merge_masks(
        self, tiler: ImageSlicer, merger_dict: Dict[str, TileMerger]
    ) -> Dict[str, np.ndarray]:
        """Merge tiles and return a dict of HWC numpy arrays."""
        probs_merged = np.moveaxis(to_numpy(merger_dict["probs"].merge()), 0, -1)
        unc_merged = np.moveaxis(to_numpy(merger_dict["uncertainty"].merge()), 0, -1)

        result = {
            "seg": tiler.crop_to_orignal_size(probs_merged),
            "uncertainty": tiler.crop_to_orignal_size(unc_merged),
        }
        if self.export_alpha and "alpha" in merger_dict:
            alpha_merged = np.moveaxis(to_numpy(merger_dict["alpha"].merge()), 0, -1)
            result["alpha"] = tiler.crop_to_orignal_size(alpha_merged)
        return result

    # ------------------------------------------------------------------
    # Saving overrides
    # ------------------------------------------------------------------

    def save_inference(
        self,
        image_path: str,
        threshold: float,
        profile: dict,
        inference: Dict[str, np.ndarray],
        output_dict: dict,
        apply_threshold: bool = False,
    ) -> None:
        """Save probabilities (via parent export strategy) and uncertainty GeoTIFF."""
        # Save the uncertainty map first (uses original profile for geo metadata)
        if self.output_uncertainty_path is not None and "uncertainty" in inference:
            self.save_uncertainty_raster(
                uncertainty_map=inference["uncertainty"],
                reference_profile=profile,
                output_path=self.output_uncertainty_path,
            )

        # Export alpha bands if requested
        if self.export_alpha and "alpha" in inference:
            alpha_path = (
                self.output_uncertainty_path.with_name(
                    self.output_uncertainty_path.stem
                    + "_alpha"
                    + self.output_uncertainty_path.suffix
                )
                if self.output_uncertainty_path
                else None
            )
            if alpha_path:
                self._save_multiband_raster(
                    data=inference["alpha"],
                    reference_profile=profile,
                    output_path=alpha_path,
                    nodata=-1.0,
                )

        # Delegate probability export to parent (uses export_strategy)
        inference["seg"] = inference.get("seg", inference["seg"])
        super().save_inference(
            image_path=image_path,
            threshold=threshold,
            profile=profile,
            inference={"seg": inference["seg"]},
            output_dict=output_dict,
            apply_threshold=apply_threshold,
        )

    # ------------------------------------------------------------------
    # Raster export helpers
    # ------------------------------------------------------------------

    def save_uncertainty_raster(
        self,
        uncertainty_map: np.ndarray,
        reference_profile: dict,
        output_path: Union[str, Path],
    ) -> None:
        """Write a single-band float32 GeoTIFF of the uncertainty map.

        The CRS and spatial transform are taken directly from
        ``reference_profile`` (the rasterio profile of the source image), so
        georeferencing is always preserved automatically.

        Args:
            uncertainty_map:   Array of shape ``[H, W, 1]`` or ``[H, W]``,
                               values in ``(0, 1]``.
            reference_profile: Rasterio profile dict from the source image.
            output_path:       Destination file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        unc = uncertainty_map
        if unc.ndim == 3:
            unc = unc[:, :, 0]

        profile = reference_profile.copy()
        profile.update(
            {
                "count": 1,
                "dtype": "float32",
                "compress": "deflate",
                "nodata": -1.0,
                "predictor": 3,  # floating-point predictor for better compression
            }
        )
        # Remove tiling hints that may be incompatible with a new size
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("tiled", None)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(unc.astype("float32"), 1)

        logger.info(
            "EvidentialInferenceProcessor: uncertainty raster saved → %s  "
            "(shape=%s, min=%.4f, max=%.4f)",
            output_path,
            unc.shape,
            float(unc.min()),
            float(unc.max()),
        )

    def _save_multiband_raster(
        self,
        data: np.ndarray,
        reference_profile: dict,
        output_path: Union[str, Path],
        nodata: float = -1.0,
    ) -> None:
        """Save a multi-band float32 GeoTIFF (used for alpha export)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        H, W, C = data.shape
        profile = reference_profile.copy()
        profile.update(
            {
                "count": C,
                "dtype": "float32",
                "compress": "deflate",
                "nodata": nodata,
            }
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("tiled", None)

        with rasterio.open(output_path, "w", **profile) as dst:
            for band_idx in range(C):
                dst.write(data[:, :, band_idx].astype("float32"), band_idx + 1)

        logger.info(
            "EvidentialInferenceProcessor: multi-band raster saved → %s  (bands=%d)",
            output_path,
            C,
        )

    # ------------------------------------------------------------------
    # Convenience: full inference + uncertainty in one call
    # ------------------------------------------------------------------

    def predict_with_uncertainty(
        self,
        image_path: Union[str, Path],
        output_probs_path: Optional[Union[str, Path]] = None,
        output_uncertainty_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """Run tiled inference and export both probs and uncertainty rasters.

        This is a convenience wrapper around ``process()``.  It temporarily
        overrides ``self.output_uncertainty_path`` if a path is provided.

        Args:
            image_path:              Source GeoTIFF.
            output_probs_path:       Destination for multi-band probability raster.
                                     Uses ``export_strategy`` if ``None``.
            output_uncertainty_path: Destination for the uncertainty raster.
            threshold:               Threshold for binarising probabilities.
                                     Unused when ``apply_threshold=False``.

        Returns:
            Dict with ``"seg"`` (HWC numpy), ``"uncertainty"`` (HW numpy),
            and optionally ``"alpha"`` (HWC numpy).
        """
        prev_unc_path = self.output_uncertainty_path
        if output_uncertainty_path is not None:
            self.output_uncertainty_path = Path(output_uncertainty_path)

        try:
            output_dict = self.process(
                image_path=str(image_path),
                threshold=threshold,
                save_inference_output=True,
            )
        finally:
            self.output_uncertainty_path = prev_unc_path

        return output_dict
