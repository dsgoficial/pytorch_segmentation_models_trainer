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
import logging
from pytorch_segmentation_models_trainer.utils.os_utils import import_module_from_cfg
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import TemplatePolygonizerProcessor
from pytorch_segmentation_models_trainer.tools.inference.inference_processors import AbstractInferenceProcessor
import hydra
import omegaconf
import rasterio
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import DictConfig, MISSING
from dataclasses import dataclass, field
from hydra.utils import instantiate

@dataclass
class InferenceParams:
    threshold: float = 0.5
    batch_size: int = 8

@dataclass
class PredictionConfig:
    training_cfg: dict = field(default_factory=dict)
    inference_params: InferenceParams = field(default_factory=InferenceParams)
    checkpoint_path: str = MISSING
    polygonizer: TemplatePolygonizerProcessor = MISSING
    inference_processor: AbstractInferenceProcessor = MISSING
    image_reader: str = MISSING #TODO: define an object structure for this part

def instantiate_model(model_cfg, checkpoint_file_path):
    pl_model = import_module_from_cfg(model_cfg).load_from_checkpoint(
        checkpoint_file_path,
        cfg=model_cfg
    )
    model = pl_model.model
    model.eval()
    return model

def instantiate_inference_processor(cfg, model, polygonizer=None):
    inference_processor_class = import_module_from_cfg(cfg)
    inference_processor = inference_processor_class(
            model=model,
            device=cfg.device,
            batch_size=8,
            export_strategy=instantiate(cfg.export_strategy),
            polygonizer=None if polygonizer is None else instantiate(polygonizer)
        )
    return inference_processor

def get_images(image_reader_cfg):
    return image_reader_cfg

@hydra.main(config_path="conf", config_name="config")
def predict(cfg: DictConfig):
    model = instantiate_model(
        cfg.training_cfg.pl_model if "pl_model" in cfg.training_cfg else cfg.training_cfg.model,
        cfg.checkpoint_path
    )
    inference_processor = instantiate_inference_processor(
        cfg.inference_processor,
        polygonizer=None if "polygonizer" not in cfg else cfg.polygonizer
    )
    images = get_images(cfg.image_reader)
    for image in images:
        inference_processor.process(
            image,
            threshold=cfg.inference_threshold
        )




if __name__=="__main__":
    predict()
