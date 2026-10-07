# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-10-07
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

import json
import logging
import os

import torch
from torch import nn

logger = logging.getLogger(__name__)

_RUNNER_STATE_FILE = "runner_state.json"


def load_pretrained_weights(
    model: nn.Module,
    path: str,
    source_format: str = "pytorch_lightning",
    strict_loading: bool = True,
) -> None:
    """Load model weights (no optimizer/scheduler state) from a checkpoint.

    Args:
        model: Module receiving the weights.
        path: Checkpoint file.
        source_format: ``"pytorch_lightning"`` (``ckpt["state_dict"]``, keys
            prefixed with ``"model."``; other keys are ignored) or
            ``"pytorch"`` (the file is a plain state dict).
        strict_loading: Passed to ``load_state_dict``.

    Raises:
        ValueError: Unknown ``source_format``.
    """
    if source_format not in {"pytorch_lightning", "pytorch"}:
        raise ValueError(
            f"source_format must be 'pytorch_lightning' or 'pytorch', got {source_format!r}."
        )
    logger.info(
        "Loading pretrained weights from '%s' (format=%s, strict=%s)",
        path,
        source_format,
        strict_loading,
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if source_format == "pytorch_lightning":
        state_dict = {
            k.removeprefix("model."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=strict_loading)
    if missing:
        logger.warning("Pretrained checkpoint: missing keys: %s", missing)
    if unexpected:
        logger.warning("Pretrained checkpoint: unexpected keys: %s", unexpected)


def resolve_checkpoint_from_runner(runner_output_dir: str, seed: int) -> str:
    """Best checkpoint of the ExperimentsRunner run trained with ``seed``.

    Reads ``<runner_output_dir>/runner_state.json`` written by
    ``ExperimentsRunner`` and returns ``best_checkpoint_path`` of the only
    completed run with that seed.

    Args:
        runner_output_dir: ``experiments_runner.output_base_dir`` of the base
            model experiment.
        seed: Seed of the wanted run.

    Returns:
        Path of the checkpoint.

    Raises:
        FileNotFoundError: No state file, or the checkpoint file is gone.
        ValueError: Seed not completed, ambiguous (k-fold), or no checkpoint.
    """
    state_path = os.path.join(runner_output_dir, _RUNNER_STATE_FILE)
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"{_RUNNER_STATE_FILE} not found in {runner_output_dir}."
        )
    with open(state_path) as f:
        runs = json.load(f).get("completed_runs", [])
    matches = [r for r in runs if int(r["seed"]) == int(seed)]
    if not matches:
        done = sorted({int(r["seed"]) for r in runs})
        raise ValueError(
            f"seed {seed} has no completed run in {runner_output_dir} "
            f"(completed seeds: {done})."
        )
    if len(matches) > 1:
        raise ValueError(
            f"seed {seed} has {len(matches)} completed runs in {runner_output_dir} "
            "(k-fold?); pass an explicit checkpoint path instead."
        )
    ckpt = matches[0].get("best_checkpoint_path") or ""
    if not ckpt:
        raise ValueError(
            f"run with seed {seed} in {runner_output_dir} has no best checkpoint."
        )
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"checkpoint of seed {seed} not found: {ckpt}")
    return ckpt
