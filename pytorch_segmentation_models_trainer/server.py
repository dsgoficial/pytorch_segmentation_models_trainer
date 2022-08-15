# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-08-31
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

import io
from pathlib import Path
import shutil

from fastapi.params import Depends
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import (
    TemplatePolygonizerProcessor,
)
from typing import Optional

import torch
import torchvision
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from hydra import compose, initialize
from hydra.utils import instantiate
from PIL import Image
from starlette.responses import Response
from torchvision import transforms
from shapely.geometry import mapping

from pytorch_segmentation_models_trainer.predict import (
    instantiate_inference_processor,
    instantiate_model_from_checkpoint,
    instantiate_polygonizer,
)
from functools import lru_cache
from .config import Settings
from rasterio.io import MemoryFile
from tempfile import NamedTemporaryFile


def get_hydra_config(config_path, config_name):
    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name)
    return cfg


@lru_cache()
def get_inference_processor():
    settings = Settings()
    cfg = get_hydra_config(settings.config_path, settings.config_name)
    inference_processor = instantiate_inference_processor(cfg)
    inference_processor.polygonizer.data_writer = None
    return inference_processor


app = FastAPI(
    title="pytorch-smt polygon inference service",
    description="""TODO.""",
    version="0.1.0",
)


@app.post("/polygonize/")
async def get_polygons_from_image_path(
    file_path: str,
    inference_processor: Settings = Depends(get_inference_processor),
    polygonizer: Optional[dict] = None,
):
    polygonizer = instantiate(polygonizer) if polygonizer is not None else None
    if polygonizer is not None:
        polygonizer.data_writer = None
    output_dict = inference_processor.process(
        file_path, save_inference_raster=False, polygonizer=polygonizer
    )
    return JSONResponse(
        content={
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {}, "geometry": geom}
                for geom in map(mapping, output_dict["polygons"])
            ],
        }
    )


@app.post("/polygonize_image/")
async def get_polygons_from_uploaded_image(
    file: UploadFile = File(...),
    inference_processor: Settings = Depends(get_inference_processor),
    polygonizer: Optional[dict] = None,
):
    suffix = Path(file.filename).suffix
    with NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        # tmp_path = Path(tmp.name)
        response = await get_polygons_from_image_path(
            file_path=tmp.name,
            inference_processor=inference_processor,
            polygonizer=polygonizer,
        )
    return response
