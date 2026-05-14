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

Utilities for global training seed management.

Setting a seed before model creation and data loading ensures that all
sources of randomness — weight initialisation, data augmentation, crop
sampling, DataLoader shuffling — produce identical sequences across runs
with the same seed value.

Covered sources of randomness
------------------------------
* Python ``random`` module (used internally by Albumentations)
* NumPy ``np.random`` (crop sampling, CutMix, ClassMix, class-balanced sampling)
* PyTorch CPU (weight initialisation, Dropout)
* PyTorch CUDA (GPU kernels — non-deterministic by default)
* ``PL_GLOBAL_SEED`` environment variable (DDP subprocesses inherit seed)
* ``PYTHONHASHSEED`` environment variable (dict/set ordering edge cases)
* CuDNN algorithm selection and all deterministic algorithms (optional — see ``deterministic_cudnn``)
* DataLoader shuffle sampler via the returned ``torch.Generator``
"""

import logging
import os

import torch
from pytorch_lightning import seed_everything as _pl_seed_everything

logger = logging.getLogger(__name__)


def set_training_seed(
    seed: int,
    deterministic_cudnn: bool = False,
) -> torch.Generator:
    """Seed all randomness sources for reproducible training.

    Delegates to :func:`pytorch_lightning.seed_everything` for consistent
    seeding of Python ``random``, NumPy, and PyTorch (CPU + CUDA), and sets
    ``PL_GLOBAL_SEED`` so that DDP-spawned subprocesses inherit the seed.
    ``PYTHONHASHSEED`` is also set (Lightning does not cover it).

    Call this function **before** instantiating any model or dataset so that
    weight initialisation and the first data-loading sequence are also
    deterministic.

    Args:
        seed: Integer seed value.  Any non-negative integer is valid; values
            above ``2**32 - 1`` are reduced modulo ``2**32`` before being
            passed to Lightning and NumPy.
        deterministic_cudnn: When ``True``, additionally sets
            ``torch.backends.cudnn.deterministic = True``,
            ``torch.backends.cudnn.benchmark = False``, and calls
            ``torch.use_deterministic_algorithms(True)`` — equivalent to
            ``Trainer(deterministic=True)``.  This eliminates all remaining
            sources of GPU non-determinism at the cost of slower convolutions
            on some architectures.  Defaults to ``False``.

    Returns:
        A ``torch.Generator`` seeded with *seed*.  Pass this object as the
        ``generator`` argument of every ``DataLoader`` to make shuffle
        sampling deterministic::

            loader = DataLoader(dataset, shuffle=True, generator=generator)

    Example YAML::

        seed: 42
        deterministic_cudnn: false  # set true only for full determinism

    Example Python::

        from pytorch_segmentation_models_trainer.utils.seed_utils import set_training_seed

        generator = set_training_seed(42)
        loader = DataLoader(dataset, shuffle=True, generator=generator)
    """
    seed32 = int(seed) % (2**32)

    # Covers random, numpy, torch (CPU + CUDA) and sets PL_GLOBAL_SEED
    _pl_seed_everything(seed32, workers=False)
    # Lightning does not set PYTHONHASHSEED
    os.environ["PYTHONHASHSEED"] = str(seed32)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        logger.info(
            "Training seed set to %d (deterministic mode ON — "
            "convolutions may be slower).",
            seed32,
        )
    else:
        logger.info("Training seed set to %d.", seed32)

    generator = torch.Generator()
    generator.manual_seed(seed32)
    return generator
