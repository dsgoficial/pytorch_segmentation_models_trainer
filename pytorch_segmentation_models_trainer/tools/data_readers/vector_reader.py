# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-04-01
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
import abc
import geopandas
import psycopg2
from geopandas import GeoDataFrame
from dataclasses import MISSING, dataclass, field
@dataclass
class GeoDF(abc.ABC):

    @abc.abstractclassmethod
    def __post_init__(self):
        pass
    
    def get_geo_df(self):
        return self.gdf

@dataclass
class FileGeoDF(GeoDF):
    file_name: str
    def __post_init__(self):
        self.gdf = geopandas.read_file(filename=self.file_name)

@dataclass
class PostgisGeoDF(GeoDF):
    user: str
    password: str
    database: str
    sql: str
    host: str = 'localhost'
    port: int = 5432

    def __post_init__(self):
        self.con = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
        self.gdf = geopandas.read_postgis(
            sql=self.sql,
            con=self.con
        )
