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
from dataclasses import MISSING, dataclass, field
from enum import Enum
import functools
import operator

import geopandas
import psycopg2
import shapely
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import (GeometryCollection, LineString,
                              MultiLineString, MultiPoint, MultiPolygon, Point,
                              Polygon)


class GeomType(Enum):
    POINT, LINE, POLYGON = range(3)

GeomTypeEnum = GeomType

@dataclass
class GeoDF(abc.ABC):

    @abc.abstractclassmethod
    def __post_init__(self):
        pass
    
    def get_geo_df(self):
        return self.gdf
    
    def get_features_from_bbox(self, x_min:float, x_max:float,\
        y_min:float, y_max:float, only_geom: bool=True) -> GeoSeries:
        feats = self.gdf.cx[x_min:x_max, y_min:y_max]
        return feats['geometry'] if only_geom else feats

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

def handle_features(input_features, output_type: GeomType = None) -> list:
    if output_type is None:
        return input_features
    handler = lambda x: handle_geometry(x, output_type)
    return GeoDataFrame(
        {'geometry':list(map(handler, input_features))},
        crs=input_features.crs
    )

def handle_geometry(geom, output_type):
    """Handles geometry 

    Args:
        geom ([type]): [description]
        output_type ([type]): [description]

    Returns:
        BaseGeometry: [description]
    """
    if isinstance(geom, (Point, MultiPoint)):
        return geom
    elif isinstance(geom, GeometryCollection):
        return GeometryCollection([handle_geometry(i, output_type) for i in geom])
    elif isinstance(geom, (Polygon, MultiPolygon)) and output_type == GeomType.LINE:
        # return type is LineString or MultiLinestring, depending whether the polygon
        # has holes, or if it is a MultiPolygon
        return geom.boundary
    elif isinstance(geom, (Polygon, MultiPolygon)) and output_type == GeomType.POINT:
        # return type is MultiPoint
        return MultiPoint(
            list(
                set(
                    geom.boundary.coords if isinstance(geom, Polygon)\
                        else functools.reduce(operator.iconcat, [i.coords for i in geom.boundary])
                )
            )
        )
    elif isinstance(geom, LineString) and output_type == GeomType.POINT:
        return MultiPoint(geom.coords)
    else:
        raise Exception("Invalid geometry handling")
