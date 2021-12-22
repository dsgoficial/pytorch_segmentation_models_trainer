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

import hydra
from omegaconf import DictConfig
from pytorch_segmentation_models_trainer.predict import predict
from pytorch_segmentation_models_trainer.train import train
from pytorch_segmentation_models_trainer.build_mask import build_masks
from pytorch_segmentation_models_trainer.config_utils import validate_config
from pytorch_segmentation_models_trainer.convert_ds import convert_dataset
from pytorch_segmentation_models_trainer.predict_from_batch import predict_from_batch
from pytorch_segmentation_models_trainer.predict_mod_polymapper_from_batch import (
    predict_mod_polymapper_from_batch,
)

import logging
import warnings
from rasterio.errors import NotGeoreferencedWarning

logging.getLogger("shapely.geos").setLevel(logging.CRITICAL)
logging.getLogger("rasterio.errors").setLevel(logging.CRITICAL)
logging.getLogger("tensorboard").setLevel(logging.CRITICAL)
logging.getLogger("numpy").setLevel(logging.CRITICAL)
logging.getLogger("skan").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.simplefilter(action="ignore", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(config_path="conf")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        return train(cfg)
    elif cfg.mode == "predict":
        return predict(cfg)
    elif cfg.mode == "predict-from-batch":
        return predict_from_batch(cfg)
    elif cfg.mode == "predict-mod-polymapper-from-batch":
        return predict_mod_polymapper_from_batch(cfg)
    elif cfg.mode == "validate-config":
        return validate_config(cfg)
    elif cfg.mode == "build-mask":
        return build_masks(cfg)
    elif cfg.mode == "convert-dataset":
        return convert_dataset(cfg)
    else:
        raise NotImplementedError


# this function is required to allow automatic detection of the module name when running
# from a binary script.
# it should be called from the executable script and not the hydra.main() function directly.
def entry():
    main()


if __name__ == "__main__":
    main()
