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
from typing import Dict, List

import hydra
import numpy as np
import omegaconf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from pytorch_segmentation_models_trainer.tools.inference.inference_processors import \
    AbstractInferenceProcessor
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import \
    TemplatePolygonizerProcessor
from pytorch_segmentation_models_trainer.utils.os_utils import \
    import_module_from_cfg


def instantiate_model_from_checkpoint(cfg: DictConfig) -> torch.nn.Module:
    pl_model = import_module_from_cfg(cfg.pl_model).load_from_checkpoint(
        cfg.checkpoint_path,
        cfg=cfg
    )
    model = pl_model.model
    model.eval()
    return model

def instantiate_polygonizer(cfg: DictConfig) -> TemplatePolygonizerProcessor:
    data_writer = instantiate(cfg.polygonizer.data_writer)
    polygonizer = instantiate(
        cfg.polygonizer,
        data_writer=data_writer
    )
    polygonizer.data_writer = data_writer
    return polygonizer

def instantiate_inference_processor(cfg: DictConfig) -> AbstractInferenceProcessor:
    obj_params = dict(cfg.inference_processor)
    obj_params['model'] = instantiate_model_from_checkpoint(cfg)
    obj_params['polygonizer'] = instantiate_polygonizer(cfg)
    obj_params['export_strategy'] = instantiate(cfg.export_strategy)
    obj_params['device'] = cfg.device
    obj_params['batch_size'] = cfg.hyperparameters.batch_size
    obj_params['mask_bands'] = sum(cfg.seg_params.values())
    obj_params.pop('_target_')
    for key, value in obj_params.items():
        if isinstance(value, omegaconf.listconfig.ListConfig):
            obj_params[key] = list(value)

    return import_module_from_cfg(cfg.inference_processor)(**obj_params)

def get_images(cfg: DictConfig) -> List[str]:
    image_reader = instantiate(cfg.inference_image_reader)
    return image_reader.get_images()

@hydra.main(config_path="conf", config_name="config")
def predict(cfg: DictConfig):
    inference_processor = instantiate_inference_processor(cfg)
    images = get_images(cfg)
    for image in tqdm(images):
        inference_processor.process(
            image,
            threshold=cfg.inference_threshold
        )

if __name__=="__main__":
    predict()
