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
from pathlib import Path

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
        y_min:float, y_max:float, only_geom: bool=True, \
        clip_to_extent: bool=True, filter_area: float=None,\
        use_spatial_filter: bool=True
    ) -> GeoSeries:
        if filter_area is not None and (not isinstance(filter_area, float) or filter_area < 0):
            raise Exception("Filter area must be a float value")
        feats = self._get_features(x_min, x_max, y_min, y_max)
        feats = self.clip_features_to_extent(
            feats, x_min, x_max, y_min, y_max
        ) if clip_to_extent else feats
        feats = feats if filter_area is None else feats[feats.area > filter_area]
        return feats[self.gdf.geometry.name] if only_geom else feats
    
    def _get_features(self,  x_min:float, x_max:float,\
        y_min:float, y_max:float):
        if self.spatial_index is None:
            return self.gdf.cx[x_min:x_max, y_min:y_max]
        clip_polygon = Polygon(
            [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]
        )
        candidates = list(self.spatial_index.intersection(clip_polygon.bounds))
        possible_matches = self.gdf.iloc[candidates]
        return possible_matches[possible_matches.intersects(clip_polygon)]
    
    def clip_features_to_extent(self, feats, x_min:float, x_max:float,\
        y_min:float, y_max:float) -> GeoSeries:
        clip_polygon = Polygon(
            [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]
        )
        return geopandas.clip(feats, clip_polygon, keep_geom_type=True)


@dataclass
class FileGeoDF(GeoDF):
    file_name: str = MISSING
    build_spatial_index: bool = True
    def __post_init__(self):
        self.gdf = geopandas.read_file(filename=self.file_name)
        self.spatial_index = self.gdf.sindex if self.build_spatial_index else None

@dataclass
class PostgisGeoDF(GeoDF):
    user: str = MISSING
    password: str = MISSING
    database: str = MISSING
    sql: str = MISSING
    host: str = 'localhost'
    port: int = 5432
    geometry_column: str = 'geom'
    build_spatial_index: bool = True

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
            con=self.con,
            geom_col=self.geometry_column
        )
        self.spatial_index = self.gdf.sindex if self.build_spatial_index else None

@dataclass
class BatchFileGeoDF:
    root_dir: str = '/data/vectors'
    file_extension: str = 'geojson'
    def __post_init__(self):
        self.vector_dict = {
            str(p).replace('.'+str(p).split(".")[-1], ''): FileGeoDF(str(p)) \
                for p in Path(self.root_dir).glob(f"**/*.{self.file_extension}")
        }
        self.spatial_index = None
    
    def get_geodf_item(self, key: str) -> GeoDataFrame:
        return self.vector_dict[key].get_geo_df() if key in self.vector_dict else None

def handle_features(input_features, output_type: GeomType = None, return_list: bool = False) -> list:
    if output_type is None:
        return input_features
    handler = lambda x: handle_geometry(x, output_type)
    return GeoDataFrame(
        {'geometry':list(map(handler, input_features))},
        crs=input_features.crs
    ) if not return_list else list(map(handler, input_features))

def handle_geometry(geom, output_type):
    """Handles geometry 

    Args:
        geom ([type]): [description]
        output_type ([type]): [description]

    Returns:
        BaseGeometry: [description]
    """
    if isinstance(geom, (Point, MultiPoint)) or \
        (isinstance(geom, (Polygon, MultiPolygon)) and output_type == GeomType.POLYGON) or \
        (isinstance(geom, (LineString, MultiLineString)) and output_type == GeomType.LINE):
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
    elif isinstance(geom, (LineString, MultiLineString)) and output_type == GeomType.POINT:
        return MultiPoint(geom.coords)
    else:
        raise Exception("Invalid geometry handling")

if __name__ == "__main__":
    geo_df = PostgisGeoDF(
        user="postgres",
        password="postgres",
        host="localhost",
        port=5432,
        database="dataset_mestrado",
        sql="select id, geom from buildings"
    )
    geo_df.gdf
