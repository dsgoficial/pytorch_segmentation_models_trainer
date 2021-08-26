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
import pathlib
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
    output_profile: dict = None

    def write_data(self, input_data: np.array, profile: dict) -> None:
        output_profile = (
            deepcopy(profile)
            if self.output_profile is None
            else dict(self.output_profile)
        )
        if "transform" not in output_profile:
            output_profile["transform"] = profile["transform"]
        if output_profile["driver"] == "JPEG" and input_data.shape[-1] == 2:
            input_data = np.dstack(
                (
                    input_data[..., 0],
                    input_data[..., 1],
                    np.zeros(input_data.shape[:-1] + (1,)),
                )
            )
        output_profile["count"] = (
            input_data.shape[-1]
            if output_profile["count"] != input_data.shape[-1]
            else output_profile["count"]
        )
        with rasterio.open(self.output_file_path, "w", **output_profile) as out:
            out.write(reshape_as_raster(input_data))


@dataclass
class VectorFileDataWriter(AbstractDataWriter):
    output_file_path: str = MISSING
    driver: str = "GeoJSON"
    mode: str = "a"

    def write_data(
        self,
        input_data: List[Union[BaseGeometry, BaseMultipartGeometry]],
        profile: dict,
    ) -> None:
        geoseries = GeoSeries(input_data, crs=profile["crs"])
        gdf = GeoDataFrame.from_features(geoseries, crs=profile["crs"])
        if len(gdf) == 0:
            return
        if not os.path.isfile(self.output_file_path) and self.mode == "a":
            gdf.to_file(self.output_file_path, driver=self.driver)
        else:
            gdf.to_file(self.output_file_path, driver=self.driver, mode=self.mode)


@dataclass
class BatchVectorFileDataWriter(VectorFileDataWriter):
    current_index: int = 0

    def _get_current_file_path(self) -> str:
        suffix = pathlib.Path(self.output_file_path).suffix
        return self.output_file_path.replace(
            suffix, f"_{self.current_index:08}{suffix}"
        )

    def write_data(
        self,
        input_data: List[Union[BaseGeometry, BaseMultipartGeometry]],
        profile: dict,
    ) -> None:
        geoseries = GeoSeries(input_data, crs=profile["crs"])
        gdf = GeoDataFrame.from_features(geoseries, crs=profile["crs"])
        if len(gdf) == 0:
            self.current_index += 1
            return
        current_file_path = self._get_current_file_path()
        if not os.path.isfile(current_file_path) and self.mode == "a":
            gdf.to_file(current_file_path, driver=self.driver)
        else:
            gdf.to_file(current_file_path, driver=self.driver, mode=self.mode)
        self.current_index += 1


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

    def write_data(
        self,
        input_data: List[Union[BaseGeometry, BaseMultipartGeometry]],
        profile: dict,
    ) -> None:
        geoseries = GeoSeries(input_data, crs=profile["crs"])
        gdf = GeoDataFrame.from_features(geoseries, crs=profile["crs"])
        if len(gdf) == 0:
            return
        gdf.rename_geometry(self.geometry_column, inplace=True)
        engine = create_engine(
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )
        gdf.to_postgis(self.table_name, engine, if_exists=self.if_exists)
