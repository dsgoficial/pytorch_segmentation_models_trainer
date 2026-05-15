# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-05-15
        copyright            : (C) 2026 by Philipe Borba
        email                : philipeborba at gmail dot com
 ***************************************************************************/

Sliding-window inference core — pure tensor in / tensor out.

``SlidingWindowCore`` performs tiled inference on a full image tensor that
has already been normalised by the dataset pipeline.  It is the shared
computation kernel used by:

* ``Model.test_step`` — Lightning full-image evaluation with proper
  metric accumulation and optional prediction export.
* Future wrappers that need tensor I/O without file-system coupling.

Supported inference modes (all combinable):

1. **Standard** — single deterministic forward pass per tile.
2. **TTA** — *N* augmented passes, de-augmented and averaged.  Optionally
   exports the per-pixel std across TTA passes as ``tta_uncertainty``.
3. **MC Dropout** — *T* stochastic passes with Dropout layers active.
   Mean used as prediction.  Optionally exports per-pixel entropy or
   mutual information as ``mc_uncertainty``.
4. **TTA + MC Dropout** — TTA mean is the official prediction; MC
   Dropout runs independently and contributes only to ``mc_uncertainty``.

Differences from ``SingleImageInfereceProcessor`` / ``MultiClassInferenceProcessor``
------------------------------------------------------------------------------------
* Input is a normalised ``torch.Tensor [C, H, W]`` (not a raw numpy array
  read from disk).
* Output is a ``SlidingWindowOutput`` dataclass (tensors, not numpy arrays).
* No file I/O, no ``A.Normalize``.  The dataset pipeline owns normalisation.

YAML example (Model config)::

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
      use_mc_dropout: false
      n_mc_samples: 10
      compute_mc_uncertainty: false
      uncertainty_mode: entropy
      output_dir: /path/to/predictions   # optional; omit to skip saving
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from pytorch_toolbelt.utils.torch_utils import image_to_tensor, to_numpy
from torch.utils.data import DataLoader

from pytorch_segmentation_models_trainer.tools.tta.tta import (
    _VALID as _TTA_VALID,
    _augment,
    _deaugment,
)
from pytorch_segmentation_models_trainer.utils.mc_dropout_utils import (
    compute_uncertainty,
    enable_mc_dropout,
    warn_if_no_dropout,
)

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class SlidingWindowOutput:
    """Prediction and optional uncertainty maps from :class:`SlidingWindowCore`.

    All tensors live on the device that was passed to ``SlidingWindowCore``.

    Args:
        prediction: Mean prediction ``[1, num_classes, H, W]``.
        tta_uncertainty: Per-pixel std across TTA passes ``[1, 1, H, W]``,
            or ``None`` when TTA uncertainty was not requested.
        mc_uncertainty: Per-pixel uncertainty from MC Dropout ``[1, 1, H, W]``
            (entropy or mutual information), or ``None`` when not requested.
    """

    prediction: torch.Tensor
    tta_uncertainty: Optional[torch.Tensor] = None
    mc_uncertainty: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class SlidingWindowCore:
    """Tensor-in / tensor-out sliding-window inference with TTA and MC Dropout.

    Args:
        model_fn: Callable ``(batch: Tensor[B,C,H,W]) → Tensor[B,num_classes,H,W]``.
            Typically the ``LightningModule`` itself (``self``).
        device: Torch device for inference and tile mergers.
        tile_size: ``(H, W)`` size fed to the model.
        tile_step: ``(H, W)`` sliding-window step.  Overlapping tiles (step <
            tile_size) are strongly recommended.
        num_classes: Number of output channels (segmentation classes).
        batch_size: Number of tiles per inner-loop batch.
        tile_weight: Tile blending weight.  ``"mean"`` (uniform), ``"pyramid"``
            (distance-based), ``"gaussian"`` (nnU-Net-style Gaussian importance
            map — reduces border artifacts).
        tta_augmentations: Ordered list of augmentation names from
            ``tools/tta/tta.py`` (e.g. ``["rot0", "rot90", "rot180",
            "rot270"]``).  Empty list / ``None`` disables TTA.
        compute_tta_uncertainty: Compute per-pixel std of softmax probs across
            TTA passes and store in ``SlidingWindowOutput.tta_uncertainty``.
            Requires at least two augmentations.
        use_mc_dropout: Enable Monte Carlo Dropout (model must contain
            ``Dropout``/``Dropout2d`` layers).
        n_mc_samples: Number of stochastic forward passes for MC Dropout.
        compute_mc_uncertainty: Compute per-pixel uncertainty from MC samples
            and store in ``SlidingWindowOutput.mc_uncertainty``.
        uncertainty_mode: ``"entropy"`` (total uncertainty) or
            ``"mutual_information"`` (epistemic uncertainty).

    Example::

        core = SlidingWindowCore(
            model_fn=self,
            device=self.device,
            tile_size=(512, 512),
            tile_step=(256, 256),
            num_classes=5,
            tta_augmentations=["rot0", "rot90", "rot180", "rot270"],
        )
        output = core.predict(images[0])  # images[0]: [C, H, W]
    """

    def __init__(
        self,
        model_fn: Callable,
        device: torch.device,
        tile_size: Tuple[int, int],
        tile_step: Tuple[int, int],
        num_classes: int,
        batch_size: int = 4,
        tile_weight: Union[str, np.ndarray] = "mean",
        tta_augmentations: Optional[List[str]] = None,
        compute_tta_uncertainty: bool = False,
        use_mc_dropout: bool = False,
        n_mc_samples: int = 10,
        compute_mc_uncertainty: bool = False,
        uncertainty_mode: str = "entropy",
    ) -> None:
        self.model_fn = model_fn
        self.device = device
        self.tile_size: Tuple[int, int] = tuple(tile_size)  # type: ignore[assignment]
        self.tile_step: Tuple[int, int] = tuple(tile_step)  # type: ignore[assignment]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.compute_tta_uncertainty = compute_tta_uncertainty
        self.use_mc_dropout = use_mc_dropout
        self.n_mc_samples = n_mc_samples
        self.compute_mc_uncertainty = compute_mc_uncertainty

        if uncertainty_mode not in ("entropy", "mutual_information"):
            raise ValueError(
                f"uncertainty_mode must be 'entropy' or 'mutual_information', "
                f"got '{uncertainty_mode}'"
            )
        self.uncertainty_mode = uncertainty_mode

        # Validate and store TTA augmentations
        augs = list(tta_augmentations) if tta_augmentations else []
        if augs:
            invalid = set(augs) - _TTA_VALID
            if invalid:
                raise ValueError(
                    f"Unknown TTA augmentations: {sorted(invalid)}. "
                    f"Valid: {sorted(_TTA_VALID)}"
                )
        self.tta_augmentations: List[str] = augs

        self._tile_weight_arr = self._resolve_tile_weight(tile_weight)

        if use_mc_dropout and isinstance(model_fn, nn.Module):
            warn_if_no_dropout(model_fn)

    # ------------------------------------------------------------------
    # Tile weight helpers
    # ------------------------------------------------------------------

    def _resolve_tile_weight(
        self, tile_weight: Union[str, np.ndarray]
    ) -> Union[str, np.ndarray]:
        if isinstance(tile_weight, np.ndarray):
            return tile_weight
        if tile_weight in ("mean", "pyramid"):
            return tile_weight
        if tile_weight == "gaussian":
            return self._gaussian_importance_map(self.tile_size)
        raise ValueError(
            f"tile_weight must be 'mean', 'pyramid', 'gaussian', or ndarray, "
            f"got '{tile_weight}'"
        )

    @staticmethod
    def _gaussian_importance_map(
        patch_size: Tuple[int, int], sigma_scale: float = 0.125
    ) -> np.ndarray:
        """nnU-Net-style 2-D Gaussian importance map.

        Pixels near the tile centre weight ~1.0; edges near-zero.  This
        reduces border artefacts from sliding-window inference because CNN
        predictions degrade near borders due to padding/context truncation.
        """
        center = [(s - 1) / 2.0 for s in patch_size]
        sigma = [max(s * sigma_scale, 1.0) for s in patch_size]
        axes = [np.arange(s, dtype=np.float32) for s in patch_size]
        grid = np.meshgrid(*axes, indexing="ij")
        gaussian = np.ones(patch_size, dtype=np.float32)
        for ax, c, s in zip(grid, center, sigma):
            gaussian *= np.exp(-0.5 * ((ax - c) / s) ** 2)
        gaussian = np.maximum(gaussian, gaussian.max() * 1e-4)
        return gaussian

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_tensor: torch.Tensor) -> SlidingWindowOutput:
        """Run sliding-window inference on a single full image.

        Args:
            image_tensor: Normalised input tensor ``[C, H, W]`` or
                ``[1, C, H, W]`` on any device.  The tensor is moved to
                ``self.device`` internally.

        Returns:
            :class:`SlidingWindowOutput` with ``prediction`` ``[1, C, H, W]``
            and optional uncertainty tensors, all on ``self.device``.
        """
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)
        image_tensor = image_tensor.to(self.device).float()

        orig_c, orig_h, orig_w = image_tensor.shape

        # Pad to be divisible by tile_size (bottom-right only → easy crop)
        pad_h = math.ceil(orig_h / self.tile_size[0]) * self.tile_size[0]
        pad_w = math.ceil(orig_w / self.tile_size[1]) * self.tile_size[1]
        padded = F.pad(
            image_tensor,
            (0, pad_w - orig_w, 0, pad_h - orig_h),
            mode="reflect",
        )

        # Convert to numpy HWC for ImageSlicer
        padded_np = np.moveaxis(padded.cpu().numpy(), 0, -1)  # [pad_H, pad_W, C]

        tiler = ImageSlicer(
            padded_np.shape,
            tile_size=self.tile_size,
            tile_step=self.tile_step,
            weight=self._tile_weight_arr,
        )
        tiles = [image_to_tensor(tile) for tile in tiler.split(padded_np)]

        mergers = self._build_mergers(tiler)
        self._predict_and_merge(tiles, tiler, mergers)

        return self._extract_output(mergers, tiler, orig_h, orig_w)

    # ------------------------------------------------------------------
    # Internal: merger construction
    # ------------------------------------------------------------------

    def _build_mergers(self, tiler: ImageSlicer) -> dict:
        pin = self.device.type == "cuda" if hasattr(self.device, "type") else True
        mergers: dict = {
            "seg": TileMerger(
                tiler.target_shape,
                self.num_classes,
                tiler.weight,
                device=self.device,
            )
        }
        has_tta = bool(self.tta_augmentations)
        if has_tta and self.compute_tta_uncertainty and len(self.tta_augmentations) > 1:
            mergers["tta_uncertainty"] = TileMerger(
                tiler.target_shape, 1, tiler.weight, device=self.device
            )
        if self.use_mc_dropout and self.compute_mc_uncertainty:
            mergers["mc_uncertainty"] = TileMerger(
                tiler.target_shape, 1, tiler.weight, device=self.device
            )
        return mergers

    # ------------------------------------------------------------------
    # Internal: tile batching loop
    # ------------------------------------------------------------------

    def _predict_and_merge(
        self, tiles: list, tiler: ImageSlicer, mergers: dict
    ) -> None:
        pin = self.device.type == "cuda" if hasattr(self.device, "type") else False
        with torch.no_grad():
            for tiles_batch, coords_batch in DataLoader(
                list(zip(tiles, tiler.crops)),
                batch_size=self.batch_size,
                pin_memory=pin,
                num_workers=0,
            ):
                tiles_batch = tiles_batch.float().to(self.device)
                pred_batch = self._infer_tile_batch(tiles_batch, mergers, coords_batch)
                mergers["seg"].integrate_batch(pred_batch, coords_batch)
                del tiles_batch, pred_batch

    def _infer_tile_batch(
        self,
        tiles_batch: torch.Tensor,
        mergers: dict,
        coords_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Return the mean prediction for a tile batch.

        When TTA is active it is used for the official prediction.
        When MC Dropout is active it always contributes to mc_uncertainty;
        it also provides the official prediction when TTA is disabled.
        """
        has_tta = bool(self.tta_augmentations)

        if has_tta:
            pred_batch = self._infer_with_tta(tiles_batch, mergers, coords_batch)
        else:
            pred_batch = self.model_fn(tiles_batch).float()

        if self.use_mc_dropout:
            mc_mean = self._infer_with_mc_dropout(tiles_batch, mergers, coords_batch)
            if not has_tta:
                pred_batch = mc_mean

        return pred_batch

    def _infer_with_tta(
        self,
        tiles_batch: torch.Tensor,
        mergers: dict,
        coords_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Run TTA, optionally accumulating per-pixel std uncertainty."""
        preds: List[torch.Tensor] = []
        for aug in self.tta_augmentations:
            aug_batch = _augment(tiles_batch, aug)
            pred = self.model_fn(aug_batch).float()
            preds.append(_deaugment(pred, aug))
            del aug_batch, pred

        mean_pred = torch.stack(preds).mean(0)

        if "tta_uncertainty" in mergers and len(preds) > 1:
            probs = torch.stack(
                [torch.softmax(p, dim=1) for p in preds]
            )  # [T, B, C, H, W]
            unc = compute_uncertainty(probs, self.uncertainty_mode)
            mergers["tta_uncertainty"].integrate_batch(unc, coords_batch)
            del probs, unc

        del preds
        return mean_pred

    def _infer_with_mc_dropout(
        self,
        tiles_batch: torch.Tensor,
        mergers: dict,
        coords_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Run MC Dropout samples, optionally accumulating uncertainty map."""
        if isinstance(self.model_fn, nn.Module):
            enable_mc_dropout(self.model_fn)

        samples: List[torch.Tensor] = []
        for _ in range(self.n_mc_samples):
            pred = self.model_fn(tiles_batch).float()
            samples.append(torch.softmax(pred, dim=1))
            del pred

        if isinstance(self.model_fn, nn.Module):
            self.model_fn.eval()

        samples_tensor = torch.stack(samples)  # [T, B, C, H, W]
        mc_mean = samples_tensor.mean(0)  # [B, C, H, W]

        if "mc_uncertainty" in mergers:
            unc = compute_uncertainty(samples_tensor, self.uncertainty_mode)
            mergers["mc_uncertainty"].integrate_batch(unc, coords_batch)
            del unc

        del samples, samples_tensor
        return mc_mean

    # ------------------------------------------------------------------
    # Internal: crop merged tensor back to original dimensions
    # ------------------------------------------------------------------

    def _crop_merged(
        self,
        merged: torch.Tensor,
        tiler: ImageSlicer,
        orig_h: int,
        orig_w: int,
    ) -> torch.Tensor:
        """Remove ImageSlicer internal margins and our bottom-right padding.

        After ``TileMerger.merge()``, the tensor has shape
        ``[C, target_H, target_W]`` which includes margins added by
        ``ImageSlicer``.  We remove them in two steps:

        1. Slice using ``tiler.margin_top / margin_left`` to recover the
           padded-image dimensions.
        2. Slice ``:orig_h, :orig_w`` to recover original dimensions (safe
           because we padded only bottom-right).
        """
        mt = tiler.margin_top
        ml = tiler.margin_left
        cropped = merged[:, mt : mt + tiler.image_height, ml : ml + tiler.image_width]
        return cropped[:, :orig_h, :orig_w]

    def _extract_output(
        self,
        mergers: dict,
        tiler: ImageSlicer,
        orig_h: int,
        orig_w: int,
    ) -> SlidingWindowOutput:
        seg = self._crop_merged(
            mergers["seg"].merge(), tiler, orig_h, orig_w
        ).unsqueeze(
            0
        )  # [1, num_classes, H, W]

        tta_unc = None
        if "tta_uncertainty" in mergers:
            tta_unc = self._crop_merged(
                mergers["tta_uncertainty"].merge(), tiler, orig_h, orig_w
            ).unsqueeze(
                0
            )  # [1, 1, H, W]

        mc_unc = None
        if "mc_uncertainty" in mergers:
            mc_unc = self._crop_merged(
                mergers["mc_uncertainty"].merge(), tiler, orig_h, orig_w
            ).unsqueeze(
                0
            )  # [1, 1, H, W]

        return SlidingWindowOutput(
            prediction=seg,
            tta_uncertainty=tta_unc,
            mc_uncertainty=mc_unc,
        )
