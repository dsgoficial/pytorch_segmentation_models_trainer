# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-07-19
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
from dataclasses import dataclass

from omegaconf import MISSING
from pytorch_segmentation_models_trainer.config_definitions.train_config import \
    TrainConfig
from pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader import \
    AbstractRasterPathListGetter
from pytorch_segmentation_models_trainer.tools.inference.inference_processors import \
    AbstractInferenceProcessor
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import \
    TemplatePolygonizerProcessor


@dataclass
class PredictionConfig(TrainConfig):
    checkpoint_path: str = MISSING
    polygonizer: TemplatePolygonizerProcessor = MISSING
    inference_processor: AbstractInferenceProcessor = MISSING
    image_reader: AbstractRasterPathListGetter = MISSING
