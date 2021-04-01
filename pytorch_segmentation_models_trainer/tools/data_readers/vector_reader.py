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
from geopandas import GeoDataFrame
from dataclasses import MISSING, dataclass, field

@dataclass
class GeoDF(abc.ABC):
    data_loader: MISSING
    params = dict

    def __post_init__(self):
        self.gdf = self.data_loader(*self.params)
    
    def get_geo_df(self):
        return self.gdf

@dataclass
class FileGeoDF(GeoDF):
    data_loader: geopandas.read_file

@dataclass
class PostgisGeoDF(GeoDF):
    data_loader: geopandas.read_postgis
