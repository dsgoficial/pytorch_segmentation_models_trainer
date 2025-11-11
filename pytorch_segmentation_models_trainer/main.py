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

import logging
import warnings
from rasterio.errors import NotGeoreferencedWarning
import torch.multiprocessing as mp


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    logging.getLogger("shapely.geos").setLevel(logging.CRITICAL)
    logging.getLogger("rasterio.errors").setLevel(logging.CRITICAL)
    logging.getLogger("tensorboard").setLevel(logging.CRITICAL)
    logging.getLogger("numpy").setLevel(logging.CRITICAL)
    logging.getLogger("skan").setLevel(logging.CRITICAL)
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
    warnings.simplefilter(action="ignore", category=Warning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if cfg.mode == "train":
        from pytorch_segmentation_models_trainer.train import train
        return train(cfg)
    elif cfg.mode == "predict":
        from pytorch_segmentation_models_trainer.predict import predict
        return predict(cfg)
    elif cfg.mode == "predict-from-batch":
        from pytorch_segmentation_models_trainer.predict_from_batch import predict_from_batch
        return predict_from_batch(cfg)
    elif cfg.mode == "predict-mod-polymapper-from-batch":
        from pytorch_segmentation_models_trainer.predict_mod_polymapper_from_batch import (
            predict_mod_polymapper_from_batch,
        )
        return predict_mod_polymapper_from_batch(cfg)
    elif cfg.mode == "validate-config":
        from pytorch_segmentation_models_trainer.config_utils import validate_config
        return validate_config(cfg)
    elif cfg.mode == "build-mask":
        from pytorch_segmentation_models_trainer.build_mask import build_masks
        return build_masks(cfg)
    elif cfg.mode == "convert-dataset":
        from pytorch_segmentation_models_trainer.convert_ds import convert_dataset
        return convert_dataset(cfg)
    elif cfg.mode == "evaluate-experiments":
        from pytorch_segmentation_models_trainer.evaluate_experiments import evaluate
        return evaluate(cfg)
    else:
        raise NotImplementedError


# this function is required to allow automatic detection of the module name when running
# from a binary script.
# it should be called from the executable script and not the hydra.main() function directly.
def entry():
    main()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
