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
import logging
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

logger = logging.getLogger(__name__)


def instantiate_model_from_checkpoint(cfg: DictConfig) -> torch.nn.Module:
    pl_model = import_module_from_cfg(cfg.pl_model).load_from_checkpoint(
        cfg.checkpoint_path, cfg=cfg, inference_mode=True
    )
    model = pl_model.model
    model.to(cfg.device)
    model.eval()
    return model


def instantiate_polygonizer(cfg: DictConfig) -> TemplatePolygonizerProcessor:
    polygonizer = instantiate(cfg.polygonizer) if "polygonizer" in cfg else None
    return polygonizer


def instantiate_inference_processor(cfg: DictConfig) -> AbstractInferenceProcessor:
    obj_params = dict(cfg.inference_processor)
    obj_params["model"] = instantiate_model_from_checkpoint(cfg)
    obj_params["polygonizer"] = instantiate_polygonizer(cfg)
    obj_params["export_strategy"] = (
        instantiate(cfg.export_strategy, _recursive_=False)
        if "export_strategy" in cfg
        else None
    )
    obj_params["device"] = cfg.device
    obj_params["batch_size"] = cfg.hyperparameters.batch_size
    if "MultiClassInferenceProcessor" not in cfg.inference_processor._target_:
        obj_params["mask_bands"] = sum(cfg.seg_params.values()) if "seg_params" in cfg else 1
    else:
        obj_params["num_classes"] = cfg.inference_processor.num_classes
    if "normalize_mean" in cfg.inference_processor:
        obj_params["normalize_mean"] = cfg.inference_processor.normalize_mean
    if "normalize_std" in cfg.inference_processor:
        obj_params["normalize_std"] = cfg.inference_processor.normalize_std
    obj_params.pop("_target_")
    for key, value in obj_params.items():
        if isinstance(value, omegaconf.listconfig.ListConfig):
            obj_params[key] = list(value)

    return import_module_from_cfg(cfg.inference_processor)(**obj_params)


def get_images(cfg: DictConfig) -> List[str]:
    image_reader = instantiate(cfg.inference_image_reader, _recursive_=False)
    return image_reader.get_images()


@hydra.main()
def predict(cfg: DictConfig):
    logger.info(
        "Starting the prediction of a model with the following configuration: \n%s",
        OmegaConf.to_yaml(cfg),
    )
    inference_processor = instantiate_inference_processor(cfg)
    images = get_images(cfg)
    params = {
        "save_inference_output": cfg.save_inference if "save_inference" in cfg else True
    }
    if "inference_threshold" in cfg:
        params["inference_threshold"] = cfg.inference_threshold
    compute_func = lambda image: inference_processor.process(
        image,
        **params,
    )
    for image in tqdm(images):
        compute_func(image)


if __name__ == "__main__":
    predict()
