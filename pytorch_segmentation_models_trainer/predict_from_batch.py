# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-03-01
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba -
                                    Cartographic Engineer @ Brazilian Army
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
import logging
import warnings
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

from pytorch_lightning.trainer.trainer import Trainer
from pytorch_segmentation_models_trainer.model_loader.frame_field_model import FrameFieldSegmentationPLModel
from pytorch_segmentation_models_trainer.custom_callbacks.training_callbacks import (
    ActiveSkeletonsPolygonizerCallback,
)
from pytorch_segmentation_models_trainer.dataset_loader.dataset import (
    ImageDataset,
    TiledInferenceImageDataset,
)
from typing import Dict

import hydra
import torch
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from tqdm import tqdm

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
logging.getLogger(
    "pytorch_segmentation_models_trainer.optimizers.poly_optimizers"
).setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.simplefilter(action="ignore", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def prepare_inference_csv(cfg):
    """
    Prepara CSV de inferência.
    
    Suporta três modos:
    1. inference_dataset.input_csv_path (direto)
    2. inference_dataset.build_csv_from_folder (construir de pasta)
    3. val_dataset.input_csv_path (retrocompatibilidade)
    
    Returns:
        Tuple (csv_path, root_dir)
    """
    # Modo 1: inference_dataset com CSV direto
    if (hasattr(cfg, 'inference_dataset') and 
        hasattr(cfg.inference_dataset, 'input_csv_path') and 
        cfg.inference_dataset.input_csv_path):
        
        logger.info("Using inference_dataset.input_csv_path")
        csv_path = cfg.inference_dataset.input_csv_path
        root_dir = cfg.inference_dataset.get('root_dir', os.path.dirname(csv_path))
        
        return csv_path, root_dir
    
    # Modo 2: inference_dataset com build_csv_from_folder
    if (hasattr(cfg, 'inference_dataset') and 
        hasattr(cfg.inference_dataset, 'build_csv_from_folder') and
        cfg.inference_dataset.build_csv_from_folder.get('enabled', False)):
        
        logger.info("Building CSV from folder (inference_dataset.build_csv_from_folder)")
        
        from pytorch_segmentation_models_trainer.tools.inference.inference_csv_builder import (
            build_inference_csv_from_config
        )
        
        csv_path = build_inference_csv_from_config(
            cfg.inference_dataset.build_csv_from_folder
        )
        
        root_dir = cfg.inference_dataset.build_csv_from_folder.get(
            'root_dir',
            cfg.inference_dataset.build_csv_from_folder.images_folder
        )
        
        return csv_path, root_dir
    
    # Modo 3: val_dataset (retrocompatibilidade)
    if hasattr(cfg, 'val_dataset') and hasattr(cfg.val_dataset, 'input_csv_path'):
        logger.info("Using val_dataset.input_csv_path (legacy mode)")
        csv_path = cfg.val_dataset.input_csv_path
        root_dir = cfg.val_dataset.get('root_dir', os.path.dirname(csv_path))
        
        return csv_path, root_dir
    
    raise ValueError(
        "No valid dataset configuration found. Please provide either:\n"
        "  - inference_dataset.input_csv_path\n"
        "  - inference_dataset.build_csv_from_folder.enabled=true\n"
        "  - val_dataset.input_csv_path (legacy)"
    )


def instantiate_dataloaders(cfg):
    """
    Instancia dataloaders para inferência.
    
    Suporta inference_dataset e val_dataset (retrocompatibilidade).
    """
    # Preparar CSV
    csv_path, root_dir = prepare_inference_csv(cfg)
    
    logger.info(f"Loading inference data from: {csv_path}")
    logger.info(f"Root directory: {root_dir}")
    
    # Ler CSV
    # Verificar se há limite de linhas
    n_rows = None
    
    if hasattr(cfg, 'inference_dataset'):
        n_rows = cfg.inference_dataset.get('n_first_rows_to_read')
    elif hasattr(cfg, 'val_dataset'):
        n_rows = cfg.val_dataset.get('n_first_rows_to_read')
    
    if n_rows:
        df = pd.read_csv(csv_path, nrows=n_rows)
        logger.info(f"Reading first {n_rows} rows from CSV")
    else:
        df = pd.read_csv(csv_path)
    
    logger.info(f"Loaded {len(df)} images for inference")
    
    # Obter dataloaders
    windowed = (
        False
        if not hasattr(cfg, "use_inference_processor")
        else cfg.use_inference_processor
    )
    
    return get_grouped_dataloaders(cfg, df, root_dir, windowed)


def get_grouped_dataloaders(cfg, df, root_dir, windowed=False):
    """Cria dataloaders agrupados por dimensão."""
    ds_dict = get_grouped_datasets(cfg, df, root_dir, windowed)
    
    # Obter batch_size e configurações de dataloader
    batch_size = cfg.hyperparameters.batch_size
    
    # Tentar pegar configurações de inference_dataset, senão val_dataset
    if hasattr(cfg, 'inference_dataset') and hasattr(cfg.inference_dataset, 'data_loader'):
        dl_config = cfg.inference_dataset.data_loader
    elif hasattr(cfg, 'val_dataset') and hasattr(cfg.val_dataset, 'data_loader'):
        dl_config = cfg.val_dataset.data_loader
    else:
        # Defaults
        dl_config = {
            'num_workers': 4,
            'prefetch_factor': 2
        }
    
    num_workers = dl_config.get('num_workers', 4)
    prefetch_factor = dl_config.get('prefetch_factor', 2)
    
    return [
        (
            key,
            torch.utils.data.DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                collate_fn=ds.collate_fn if hasattr(ds, "collate_fn") else None,
            ),
        )
        for key, ds in sorted(
            ds_dict.items(), key=lambda x: x[0][0] * x[0][1], reverse=True
        )
    ]


def get_grouped_datasets(cfg, df, root_dir, windowed):
    """Cria datasets agrupados por dimensão."""
    from tqdm import tqdm

    tqdm.pandas()
    
    # Skip de polígonos existentes
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
            if (
                hasattr(cfg, "save_not_found_image_list_to_csv")
                and cfg.save_not_found_image_list_to_csv
            ):
                df[df["output_exists"] == False].to_csv(
                    cfg.polygonizer.data_writer.output_file_folder
                    + "/not_found_image_list.csv"
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
    
    # Criar datasets
    ds_dict = (
        ImageDataset.get_grouped_datasets(
            df,
            group_by_keys=["width", "height"],
            root_dir=root_dir,
            augmentation_list=A.Compose([A.Normalize(), ToTensorV2()]),
        )
        if not windowed
        else TiledInferenceImageDataset.get_grouped_datasets(
            df,
            group_by_keys=["width", "height"],
            root_dir=root_dir,
            normalize_output=True,
            pad_if_needed=True,
            model_input_shape=tuple(cfg.inference_processor.model_input_shape),
            step_shape=tuple(cfg.inference_processor.step_shape),
        )
    )
    return ds_dict


@hydra.main()
def predict_from_batch(cfg: DictConfig):
    """
    Executa predição em batch.
    
    Suporta:
    - inference_dataset.input_csv_path
    - inference_dataset.build_csv_from_folder
    - val_dataset.input_csv_path (retrocompatibilidade)
    """
    logger.info(
        "Starting the prediction of a model with the following configuration: \n%s",
        OmegaConf.to_yaml(cfg),
    )
    
    # Carregar modelo
    model = import_module_from_cfg(cfg.pl_model).load_from_checkpoint(
        cfg.checkpoint_path, cfg=cfg, inference_mode=True
    )
    model.eval()
    
    # Criar dataloaders
    dataloader_list = instantiate_dataloaders(cfg)
    callback_list = [] if not isinstance(model, FrameFieldSegmentationPLModel) \
        else [ActiveSkeletonsPolygonizerCallback()]
    
    # Criar trainer
    trainer = Trainer(
        **cfg.pl_trainer, callbacks=callback_list
    )
    
    # Executar predições
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
                f"Error occurred during inference of batch group {key}. "
                f"The process will continue, but you may want to run the "
                f"inference again for the missing results."
            )


if __name__ == "__main__":
    predict_from_batch()