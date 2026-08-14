# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2026-08-13
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
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SearchParamConfig:
    """Single hyperparameter to search over.

    Args:
        key: OmegaConf dotpath to the config field to override.
            Example: ``"optimizer.lr"``, ``"loss"``.
        type: Distribution type — ``"float"``, ``"int"``, ``"categorical"``,
            or ``"config_block"``.
        low: Lower bound for ``float`` and ``int`` types.
        high: Upper bound for ``float`` and ``int`` types.
        log: When ``True`` samples on a log scale (``float`` only).
        step: Step size for ``int`` sampling.
        choices: For ``categorical``: list of scalar values to choose from.
            For ``config_block``: list of dicts with ``name`` and ``values``
            keys — the ``values`` dict is deep-merged into the config at
            ``key`` when its ``name`` is selected.

    Examples:

    .. code-block:: yaml

        # float
        - key: optimizer.lr
          type: float
          low: 1.0e-5
          high: 1.0e-2
          log: true

        # int
        - key: train_dataset.batch_size
          type: int
          low: 8
          high: 64

        # categorical
        - key: model.encoder_name
          type: categorical
          choices: [resnet34, resnet50, efficientnet-b0]

        # config_block — swaps entire config subtree
        - key: loss
          type: config_block
          choices:
            - name: cross_entropy
              values:
                _target_: torch.nn.CrossEntropyLoss
            - name: focal
              values:
                _target_: kornia.losses.FocalLoss
                alpha: 0.5
                gamma: 2.0
    """

    key: str = ""
    type: str = "float"
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    step: Optional[int] = None
    choices: Optional[List[Any]] = None


@dataclass
class OptunaSearchConfig:
    """Configuration for Optuna hyperparameter search.

    Args:
        n_trials: Number of trials to run.
        metric: Metric key to optimise, as logged by Lightning.
            Example: ``"val/JaccardIndex"``.
        direction: ``"maximize"`` or ``"minimize"``.
        sampler: Sampling algorithm — ``"TPE"`` (default), ``"GP"``,
            ``"CmaES"``, ``"Random"``, or ``"Grid"``.
        storage: Optuna storage URL. ``None`` uses in-memory storage (no
            resume). ``"sqlite:///study.db"`` enables disk persistence and
            automatic resume when the study name already exists.
        study_name: Name for the Optuna study (used as identifier in storage).
        search_space: List of :class:`SearchParamConfig` entries defining which
            hyperparameters to search and with what distribution.
        save_visualizations: When ``True`` (default) saves Plotly HTML plots
            to ``<output_base_dir>/plots/`` after the study finishes.
        save_param_importances: When ``True`` (default) saves fANOVA parameter
            importance scores to ``<output_base_dir>/param_importances.json``
            after the study finishes (requires at least 2 completed trials).

    Example YAML:

    .. code-block:: yaml

        experiments_runner:
          n_trials: 50
          output_base_dir: outputs/hpo
          optuna_search:
            n_trials: 50
            metric: val/JaccardIndex
            direction: maximize
            sampler: TPE
            storage: sqlite:///outputs/hpo/study.db
            study_name: unet_hpo
            search_space:
              - key: optimizer.lr
                type: float
                low: 1.0e-5
                high: 1.0e-2
                log: true
              - key: loss
                type: config_block
                choices:
                  - name: cross_entropy
                    values:
                      _target_: torch.nn.CrossEntropyLoss
                  - name: focal
                    values:
                      _target_: kornia.losses.FocalLoss
                      alpha: 0.5
                      gamma: 2.0
    """

    n_trials: int = 10
    metric: str = "val/loss"
    direction: str = "maximize"
    sampler: str = "TPE"
    storage: Optional[str] = None
    study_name: str = "optuna_study"
    search_space: List[Any] = field(default_factory=list)
    save_visualizations: bool = True
    save_param_importances: bool = True
