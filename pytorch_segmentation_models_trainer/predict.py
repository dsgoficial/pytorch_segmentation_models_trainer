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
import hydra
import omegaconf
import rasterio
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config")
def predict(cfg: DictConfig):
    logger.info(
        "Starting the inference of input images using: \n%s",
        omegaconf.to_yaml(cfg)
    )
    model = Model(cfg) if "pl_model" not in cfg else import_module_from_cfg(cfg.pl_model)(cfg)


if __name__=="__main__":
    predict()
