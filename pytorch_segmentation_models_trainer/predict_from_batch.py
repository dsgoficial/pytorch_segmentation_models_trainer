# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-12-15
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
import itertools
import logging
from pathlib import Path
from pytorch_segmentation_models_trainer.dataset_loader.dataset import ImageDataset
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
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

logger = logging.getLogger(__name__)

import os
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD_SIZE = torch.cuda.device_count()


def instantiate_dataloaders(cfg):
    df = (
        pd.read_csv(
            cfg.val_dataset.input_csv_path, nrows=cfg.val_dataset.n_first_rows_to_read
        )
        if "n_first_rows_to_read" in cfg.val_dataset
        and cfg.val_dataset.n_first_rows_to_read is not None
        else pd.read_csv(cfg.val_dataset.input_csv_path)
    )
    return (
        get_grouped_dataloaders(cfg, df)
        if "center_crop" not in cfg
        else get_dataloader_with_center_crop(cfg, df)
    )


def get_grouped_dataloaders(cfg, df):
    ds_dict = ImageDataset.get_grouped_datasets(
        df,
        group_by_keys=["width", "height"],
        root_dir=cfg.val_dataset.root_dir,
        augmentation_list=A.Compose([A.Normalize(), ToTensorV2()]),
    )
    device_count = 1 if cfg.device == "cpu" else torch.cuda.device_count()
    batch_size = cfg.hyperparameters.batch_size * device_count
    return [
        torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.val_dataset.data_loader.num_workers,
            prefetch_factor=cfg.val_dataset.data_loader.prefetch_factor * device_count,
        )
        for ds in ds_dict.values()
    ]


def get_dataloader_with_center_crop(cfg, df):
    ds = ImageDataset(
        df,
        root_dir=cfg.val_dataset.root_dir,
        augmentation_list=A.Compose(
            [
                A.CenterCrop(
                    height=cfg.center_crop.height,
                    width=cfg.center_crop.width,
                    always_apply=True,
                    p=1,
                ),
                A.Normalize(),
                ToTensorV2(),
            ]
        ),
    )
    device_count = 1 if cfg.device == "cpu" else torch.cuda.device_count()
    batch_size = cfg.hyperparameters.batch_size * device_count
    return [
        torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.val_dataset.data_loader.num_workers,
            prefetch_factor=cfg.val_dataset.data_loader.prefetch_factor * device_count,
        )
    ]


def instantiate_model_from_checkpoint_distributed(cfg: DictConfig) -> torch.nn.Module:
    pl_model = import_module_from_cfg(cfg.pl_model).load_from_checkpoint(
        cfg.checkpoint_path, cfg=cfg
    )
    model = pl_model.model
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model).to(cfg.device)
    else:
        model = model.to(cfg.device)
    model.eval()
    return model


@hydra.main()
def predict_from_batch(cfg: DictConfig):
    logger.info(
        "Starting the prediction of a model with the following configuration: \n%s",
        OmegaConf.to_yaml(cfg),
    )
    model = instantiate_model_from_checkpoint_distributed(cfg)
    dataloader_list = instantiate_dataloaders(cfg)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        futures = []

        def process_batch(batch):
            images = batch["image"].to(cfg.device)
            paths = batch["path"]
            with torch.no_grad():
                batch_predictions = model(images)
            seg_batch, crossfield_batch = batch_predictions.values()
            parent_dir_name_list = [Path(path).stem for path in paths]
            polygonizer = instantiate_polygonizer(cfg)
            return polygonizer.process(
                {"seg": seg_batch, "crossfield": crossfield_batch},
                profile=None,
                parent_dir_name=parent_dir_name_list,
                pool=pool,
                convert_output_to_world_coords=False,
            )

        for batch in tqdm(
            itertools.chain.from_iterable(dataloader_list),
            total=sum(len(i) for i in dataloader_list),
            desc="Processing polygonization for each batch",
        ):
            future = process_batch(batch)
            futures.extend(future)
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            desc="Writing outputs",
            total=len(futures),
        ):
            future.result()


if __name__ == "__main__":
    predict_from_batch()
