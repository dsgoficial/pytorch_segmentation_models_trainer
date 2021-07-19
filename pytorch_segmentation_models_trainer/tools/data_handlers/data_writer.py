# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-07-14
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
from copy import deepcopy
from typing import List, Union
import numpy as np
import os
import shapely
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from geopandas import GeoDataFrame, GeoSeries
from omegaconf import MISSING
import rasterio
from rasterio.plot import reshape_as_raster
from sqlalchemy.engine import create_engine
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

class AbstractDataWriter(ABC):
    @abstractmethod
    def write_data(self, input_data: np.array) -> None:
        pass

@dataclass
class RasterDataWriter(AbstractDataWriter):
    output_file_path: str = MISSING

    def write_data(self, input_data: np.array, profile: dict) -> None:
        profile = deepcopy(profile)
        profile['count'] = input_data.shape[-1]
        with rasterio.open(self.output_file_path, 'w', **profile) as out:
            out.write(reshape_as_raster(input_data))

@dataclass
class VectorFileDataWriter(AbstractDataWriter):
    output_file_path: str = MISSING
    driver: str = "GeoJSON"
    mode: str = "a"


    def write_data(self, input_data: List[Union[BaseGeometry, BaseMultipartGeometry]], profile: dict) -> None:
        geoseries = GeoSeries(input_data, crs=profile['crs'])
        gdf = GeoDataFrame.from_features(geoseries, crs=profile['crs'])
        if len(gdf) == 0:
            return
        if not os.path.isfile(self.output_file_path) and self.mode == "a":
            gdf.to_file(
                self.output_file_path,
                driver=self.driver
            )
        else:
            gdf.to_file(
                self.output_file_path,
                driver=self.driver,
                mode=self.mode
            )

@dataclass
class VectorDatabaseDataWriter(AbstractDataWriter):
    user: str = MISSING
    password: str = MISSING
    database: str = MISSING
    sql: str = MISSING
    host: str = "localhost"
    port: int = 5432
    table_name: str = "buildings"
    geometry_column: str = "geom"
    if_exists: str = "append"

    def write_data(self, input_data: List[Union[BaseGeometry, BaseMultipartGeometry]], profile: dict) -> None:
        geoseries = GeoSeries(input_data, crs=profile['crs'])
        gdf = GeoDataFrame.from_features(geoseries, crs=profile['crs'])
        if len(gdf) == 0:
            return
        gdf.rename_geometry(self.geometry_column, inplace=True)
        engine = create_engine(f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}")
        gdf.to_postgis(self.table_name, engine, if_exists=self.if_exists)