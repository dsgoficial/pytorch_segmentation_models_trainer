# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-04-12
        copyright            : (C) 2026 by Philipe Borba
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

Test Time Augmentation (TTA) for segmentation models.

The eight supported augmentations form the D4 dihedral group — all symmetries
of a square.  Each augmentation has an exact inverse, so predictions are
de-augmented before averaging, ensuring the final mask is in the same spatial
orientation as the original input.

Augmentation names (use these strings in YAML configs)
-------------------------------------------------------
- ``rot0``          — identity (original image, no transformation)
- ``rot90``         — 90° counter-clockwise rotation
- ``rot180``        — 180° rotation
- ``rot270``        — 270° counter-clockwise rotation
- ``flip_h``        — horizontal flip (left-right mirror)
- ``flip_v``        — vertical flip (up-down mirror)
- ``flip_h_rot90``  — horizontal flip followed by 90° CCW rotation
- ``flip_v_rot90``  — vertical flip followed by 90° CCW rotation

YAML example — inference processor
------------------------------------
::

    inference_processor:
      _target_: ...SingleImageInfereceProcessor
      model_input_shape: [448, 448]
      step_shape: [224, 224]
      use_tta: true
      tta_augmentations:
        - rot0          # original image
        - rot90         # 90° counter-clockwise
        - rot180        # 180°
        - rot270        # 270° counter-clockwise
        - flip_h        # horizontal flip
        - flip_v        # vertical flip
        - flip_h_rot90  # horizontal flip + 90° CCW
        - flip_v_rot90  # vertical flip + 90° CCW

YAML example — test step (model config)
-----------------------------------------
::

    use_tta: true
    tta_augmentations:
      - rot0
      - rot90
      - rot180
      - rot270

Note on frame-field models (``crossfield`` output)
---------------------------------------------------
The ``crossfield`` tensor encodes tangent-field angles.  Correctly
de-augmenting it requires transforming the angle values themselves, which is
non-trivial.  TTA is therefore applied only to the ``seg`` output for
frame-field processors; ``crossfield`` is taken from the identity (``rot0``)
pass.  If ``rot0`` is not in the augmentation list the first pass is used.
"""

from __future__ import annotations

from typing import Callable, Dict, FrozenSet, List, Union

import torch

# ---------------------------------------------------------------------------
# Augmentation name constants
# ---------------------------------------------------------------------------

#: Identity — no transformation.
ROT0: str = "rot0"

#: 90° counter-clockwise rotation.
ROT90: str = "rot90"

#: 180° rotation.
ROT180: str = "rot180"

#: 270° counter-clockwise rotation.
ROT270: str = "rot270"

#: Horizontal flip (left-right mirror).
FLIP_H: str = "flip_h"

#: Vertical flip (up-down mirror).
FLIP_V: str = "flip_v"

#: Horizontal flip followed by 90° CCW rotation.
FLIP_H_ROT90: str = "flip_h_rot90"

#: Vertical flip followed by 90° CCW rotation.
FLIP_V_ROT90: str = "flip_v_rot90"

#: All eight D4 augmentations in a canonical order.
D4_AUGMENTATIONS: List[str] = [
    ROT0,
    ROT90,
    ROT180,
    ROT270,
    FLIP_H,
    FLIP_V,
    FLIP_H_ROT90,
    FLIP_V_ROT90,
]

#: Convenience preset — four 90-degree rotations only.
ROTATION_AUGMENTATIONS: List[str] = [ROT0, ROT90, ROT180, ROT270]

_VALID: FrozenSet[str] = frozenset(D4_AUGMENTATIONS)


# ---------------------------------------------------------------------------
# Low-level augment / de-augment helpers
# ---------------------------------------------------------------------------


def _augment(x: torch.Tensor, aug: str) -> torch.Tensor:
    """Apply a single named augmentation to tensor *x*.

    Args:
        x: Tensor of shape ``[B, C, H, W]`` or ``[C, H, W]``.
        aug: Augmentation name (one of the module-level string constants).

    Returns:
        Augmented tensor with the same shape as *x*.

    Raises:
        ValueError: If *aug* is not a recognised augmentation name.
    """
    if aug == ROT0:
        return x
    if aug == ROT90:
        return torch.rot90(x, 1, dims=(-2, -1))
    if aug == ROT180:
        return torch.rot90(x, 2, dims=(-2, -1))
    if aug == ROT270:
        return torch.rot90(x, 3, dims=(-2, -1))
    if aug == FLIP_H:
        return torch.flip(x, dims=[-1])
    if aug == FLIP_V:
        return torch.flip(x, dims=[-2])
    if aug == FLIP_H_ROT90:
        # horizontal flip, then 90° CCW
        return torch.rot90(torch.flip(x, dims=[-1]), 1, dims=(-2, -1))
    if aug == FLIP_V_ROT90:
        # vertical flip, then 90° CCW
        return torch.rot90(torch.flip(x, dims=[-2]), 1, dims=(-2, -1))
    raise ValueError(
        f"Unknown TTA augmentation: '{aug}'. " f"Valid options: {sorted(_VALID)}"
    )


def _deaugment(x: torch.Tensor, aug: str) -> torch.Tensor:
    """Apply the exact inverse of *aug* to tensor *x*.

    Inverse transforms
    ------------------
    - ``rot0``         → identity
    - ``rot90``        → rot270  (−90°)
    - ``rot180``       → rot180  (self-inverse)
    - ``rot270``       → rot90   (+90°)
    - ``flip_h``       → flip_h  (self-inverse)
    - ``flip_v``       → flip_v  (self-inverse)
    - ``flip_h_rot90`` → rot270 then flip_h
    - ``flip_v_rot90`` → rot270 then flip_v

    Args:
        x: Tensor of shape ``[B, C, H, W]`` or ``[C, H, W]``.
        aug: Augmentation whose inverse will be applied.

    Returns:
        De-augmented tensor with the same shape as *x*.

    Raises:
        ValueError: If *aug* is not a recognised augmentation name.
    """
    if aug == ROT0:
        return x
    if aug == ROT90:
        return torch.rot90(x, -1, dims=(-2, -1))
    if aug == ROT180:
        return torch.rot90(x, -2, dims=(-2, -1))
    if aug == ROT270:
        return torch.rot90(x, -3, dims=(-2, -1))
    if aug == FLIP_H:
        return torch.flip(x, dims=[-1])  # self-inverse
    if aug == FLIP_V:
        return torch.flip(x, dims=[-2])  # self-inverse
    if aug == FLIP_H_ROT90:
        # forward: flip_h → rot90  ⟹  inverse: rot270 → flip_h
        return torch.flip(torch.rot90(x, -1, dims=(-2, -1)), dims=[-1])
    if aug == FLIP_V_ROT90:
        # forward: flip_v → rot90  ⟹  inverse: rot270 → flip_v
        return torch.flip(torch.rot90(x, -1, dims=(-2, -1)), dims=[-2])
    raise ValueError(
        f"Unknown TTA augmentation: '{aug}'. " f"Valid options: {sorted(_VALID)}"
    )


def _deaugment_output(
    output: Union[torch.Tensor, Dict[str, torch.Tensor]],
    aug: str,
    skip_keys: FrozenSet[str] = frozenset(),
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """De-augment model output, optionally skipping certain keys.

    Keys listed in *skip_keys* are returned unchanged (their spatial
    content will be taken from the identity pass instead of being averaged).

    Args:
        output: Model output — either a plain tensor or a dict of tensors.
        aug: Augmentation to invert.
        skip_keys: Keys to leave untransformed (dict outputs only).

    Returns:
        De-augmented output with the same type and structure as *output*.
    """
    if isinstance(output, torch.Tensor):
        return _deaugment(output, aug)
    return {
        key: (value if key in skip_keys else _deaugment(value, aug))
        for key, value in output.items()
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_tta(
    model_fn: Callable,
    batch: torch.Tensor,
    augmentations: List[str],
    skip_keys: FrozenSet[str] = frozenset(),
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Run *model_fn* under each augmentation, de-augment, and average.

    For each augmentation in *augmentations*:

    1. Apply the augmentation to *batch*.
    2. Call *model_fn* on the augmented batch.
    3. Apply the inverse augmentation to the prediction.

    The de-augmented predictions are then averaged to produce the final output.

    Args:
        model_fn: Callable ``(batch) → prediction``.  The prediction can be a
            ``torch.Tensor`` or a ``dict[str, torch.Tensor]``.
        batch: Input tensor of shape ``[B, C, H, W]``.
        augmentations: Ordered list of augmentation names to apply.  Must not
            be empty.  Example: ``["rot0", "rot90", "rot180", "rot270"]``.
        skip_keys: For dict outputs, keys whose spatial content cannot be
            simply de-augmented (e.g. ``"crossfield"`` in frame-field models).
            These keys are copied from the identity (``rot0``) pass; if
            ``rot0`` is absent they are taken from the first pass.

    Returns:
        Averaged prediction with the same type and shape as a single
        *model_fn* call.

    Raises:
        ValueError: If *augmentations* is empty or contains unknown names.

    Example::

        from pytorch_segmentation_models_trainer.tools.tta.tta import apply_tta

        pred = apply_tta(
            model_fn=model,
            batch=tiles_batch,
            augmentations=["rot0", "rot90", "rot180", "rot270"],
        )
    """
    if not augmentations:
        raise ValueError("augmentations list must not be empty.")
    invalid = set(augmentations) - _VALID
    if invalid:
        raise ValueError(
            f"Unknown TTA augmentations: {sorted(invalid)}. "
            f"Valid options: {sorted(_VALID)}"
        )

    predictions: list = []
    identity_output = None

    for aug in augmentations:
        augmented_batch = _augment(batch, aug)
        output = model_fn(augmented_batch)
        if aug == ROT0:
            identity_output = output
        deaugmented = _deaugment_output(output, aug, skip_keys=skip_keys)
        predictions.append(deaugmented)

    # --- average ---
    if isinstance(predictions[0], torch.Tensor):
        return torch.stack(predictions).mean(dim=0)

    # dict output
    fallback = identity_output if identity_output is not None else predictions[0]
    result: Dict[str, torch.Tensor] = {}
    for key in predictions[0]:
        if key in skip_keys:
            # Use the un-transformed output from the identity pass
            result[key] = (
                fallback[key] if isinstance(fallback, dict) else predictions[0][key]
            )
        else:
            result[key] = torch.stack([p[key] for p in predictions]).mean(dim=0)
    return result
