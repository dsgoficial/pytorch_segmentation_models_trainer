# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-03-02
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba - Cartographic Engineer
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
import concurrent.futures
from concurrent.futures.thread import ThreadPoolExecutor
import logging
from pathlib import Path
from pytorch_segmentation_models_trainer.predict import (
    instantiate_model_from_checkpoint,
    instantiate_polygonizer,
)
from pytorch_segmentation_models_trainer.tools.parallel_processing.process_executor import (
    Executor,
)
from typing import Dict, List

import hydra
import numpy as np
import omegaconf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from tqdm import tqdm

from pytorch_segmentation_models_trainer.tools.inference.inference_processors import (
    AbstractInferenceProcessor,
)
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import (
    TemplatePolygonizerProcessor,
)
from pytorch_segmentation_models_trainer.utils.os_utils import import_module_from_cfg
from functools import partial
import copy

logger = logging.getLogger(__name__)


def instantiate_dataloader(cfg):
    ds = instantiate(
        cfg.val_dataset, return_distance_mask=False, return_size_mask=False
    )
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.hyperparameters.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )

    return dataloader


@hydra.main()
def predict_from_batch(cfg: DictConfig):
    logger.info(
        "Starting the prediction of a model with the following configuration: \n%s",
        OmegaConf.to_yaml(cfg),
    )
    model = instantiate_model_from_checkpoint(cfg)
    dataloader = instantiate_dataloader(cfg)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        futures = []
        for batch in tqdm(dataloader, desc="Processing batches"):
            images = batch["image"].to(cfg.device)
            paths = batch["path"]
            with torch.no_grad():
                batch_predictions = model(images)
            seg_batch, crossfield_batch = batch_predictions.values()
            parent_dir_name_list = [Path(path).stem for path in paths]
            polygonizer = instantiate_polygonizer(cfg)
            new_futures = polygonizer.process(
                {"seg": seg_batch, "crossfield": crossfield_batch},
                profile=None,
                parent_dir_name=parent_dir_name_list,
                pool=pool,
            )
            futures.extend(new_futures)
        for future in tqdm(
            concurrent.futures.as_completed(futures), desc="Finishing writing polygons"
        ):
            future.result()


if __name__ == "__main__":
    predict_from_batch()
