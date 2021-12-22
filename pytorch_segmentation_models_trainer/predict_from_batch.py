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
import math
from pathlib import Path
import warnings

from pytorch_lightning.trainer.trainer import Trainer
from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import (
    FrameFieldPolygonizerCallback,
)
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    ImageDataset,
    TiledInferenceImageDataset,
)
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
import swifter

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
from rasterio.errors import NotGeoreferencedWarning

logger = logging.getLogger(__name__)

import os
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD_SIZE = torch.cuda.device_count()

logging.getLogger("shapely.geos").setLevel(logging.CRITICAL)
logging.getLogger("rasterio.errors").setLevel(logging.CRITICAL)
logging.getLogger("tensorboard").setLevel(logging.CRITICAL)
logging.getLogger("numpy").setLevel(logging.CRITICAL)
logging.getLogger("skan").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.simplefilter(action="ignore", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def instantiate_dataloaders(cfg):
    df = (
        pd.read_csv(
            cfg.val_dataset.input_csv_path, nrows=cfg.val_dataset.n_first_rows_to_read
        )
        if "n_first_rows_to_read" in cfg.val_dataset
        and cfg.val_dataset.n_first_rows_to_read is not None
        else pd.read_csv(cfg.val_dataset.input_csv_path)
    )
    return get_grouped_dataloaders(
        cfg,
        df,
        windowed=False
        if "use_inference_processor" not in cfg
        else cfg.use_inference_processor,
    )


def get_grouped_dataloaders(cfg, df, windowed=False):
    ds_dict = get_grouped_datasets(cfg, df, windowed)
    batch_size = cfg.hyperparameters.batch_size
    return [
        (
            key,
            torch.utils.data.DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=cfg.val_dataset.data_loader.num_workers,
                prefetch_factor=cfg.val_dataset.data_loader.prefetch_factor,
                collate_fn=ds.collate_fn if hasattr(ds, "collate_fn") else None,
            ),
        )
        for key, ds in sorted(
            ds_dict.items(), key=lambda x: x[0][0] * x[0][1], reverse=True
        )
    ]


def get_grouped_datasets(cfg, df, windowed):
    from tqdm import tqdm

    tqdm.pandas()
    if "skip_existing_polygons" in cfg and cfg.skip_existing_polygons:
        logger.info("Filtering out images with polygonization that already exist.")
        if (
            hasattr(cfg, "skip_if_folder_or_file_created")
            and cfg.skip_if_folder_or_file_created == "file"
        ):
            df["output_exists"] = df["image"].swifter.apply(
                lambda x: Path(
                    os.path.join(
                        cfg.polygonizer.data_writer.output_file_folder,
                        Path(x).stem,
                        "output.geojson",
                    )
                ).exists()
            )
        else:
            df["output_exists"] = df["image"].swifter.apply(
                lambda x: Path(
                    os.path.join(
                        cfg.polygonizer.data_writer.output_file_folder, Path(x).stem
                    )
                ).exists()
            )
        df = df[df["output_exists"] == False].reset_index(drop=True)

    ds_dict = (
        ImageDataset.get_grouped_datasets(
            df,
            group_by_keys=["width", "height"],
            root_dir=cfg.val_dataset.root_dir,
            augmentation_list=A.Compose([A.Normalize(), ToTensorV2()]),
        )
        if not windowed
        else TiledInferenceImageDataset.get_grouped_datasets(
            df,
            group_by_keys=["width", "height"],
            root_dir=cfg.val_dataset.root_dir,
            normalize_output=True,
            pad_if_needed=True,
            model_input_shape=tuple(cfg.inference_processor.model_input_shape),
            step_shape=tuple(cfg.inference_processor.step_shape),
        )
    )
    return ds_dict


@hydra.main()
def predict_from_batch(cfg: DictConfig):
    logger.info(
        "Starting the prediction of a model with the following configuration: \n%s",
        OmegaConf.to_yaml(cfg),
    )
    # model = instantiate_model_from_checkpoint_distributed(cfg)
    model = import_module_from_cfg(cfg.pl_model).load_from_checkpoint(
        cfg.hyperparameters.resume_from_checkpoint, cfg=cfg
    )
    dataloader_list = instantiate_dataloaders(cfg)
    trainer = Trainer(**cfg.pl_trainer, callbacks=[FrameFieldPolygonizerCallback()])
    for key, dataloader in tqdm(
        dataloader_list,
        total=len(dataloader_list),
        desc="Processing inference for each group of images",
        colour="green",
    ):
        logger.info(f"Processing inference for images of shape {key}")
        try:
            trainer.predict(model, dataloader)
        except Exception as e:
            logger.exception(e)
            logger.exception(
                f"Error occurred during inference of batch group {key}. The process will continue, but you may want to run the inference again for the missing results."
            )


if __name__ == "__main__":
    predict_from_batch()
