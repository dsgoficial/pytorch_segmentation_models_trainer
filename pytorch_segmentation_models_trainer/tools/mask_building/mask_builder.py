# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-04-02
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
from abc import ABC, abstractmethod
from collections import OrderedDict
import csv
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import rasterio
from omegaconf import MISSING, DictConfig, OmegaConf
from pytorch_segmentation_models_trainer.tools.data_readers.raster_reader import (
    DatasetEntry, MaskOutputType, RasterFile)
from pytorch_segmentation_models_trainer.tools.data_readers.vector_reader import (
    FileGeoDF, GeoDF, GeomType, GeomTypeEnum)
from pytorch_segmentation_models_trainer.utils.os_utils import \
    make_path_relative
from rasterio.plot import reshape_as_image
from hydra.utils import instantiate

mask_dict = {
    GeomTypeEnum.POLYGON : "polygon_mask",
    GeomTypeEnum.LINE: "boundary_mask",
    GeomTypeEnum.POINT: "vertex_mask",
    "angle": "crossfield_mask"
}


def build_destination_dirs(input_base_path: str, output_base_path: str):
    if input_base_path == os.path.commonpath([input_base_path, output_base_path]) \
        and not output_base_path.split(
            os.path.commonpath([input_base_path, output_base_path])
        )[-1].startswith("/.."):
        raise Exception("input path must not be in output_path")
    return [
        Path(
            dirpath.replace(input_base_path, output_base_path)    
        ).mkdir(parents=True, exist_ok=True) for dirpath, _, __ in os.walk(input_base_path)
    ]

def build_output_raster_list(input_raster_path, cfg):
    image_dir_name = os.path.basename(
        os.path.normpath(cfg.image_root_dir)
    )
    return [
        str(
            os.path.join(
                getattr(cfg, f"{mask_type}_folder_name"),
                os.path.dirname(
                    os.path.normpath(
                        str(input_raster_path).split(f'{image_dir_name}/')[-1]
                    )
                )
            )
         ) for mask_type in [
             'polygon_mask', 'boundary_mask', 'vertex_mask',
             'crossfield_mask', 'distance_mask', 'size_mask'] if getattr(cfg, f"build_{mask_type}")
    ]

def build_csv_file_from_concurrent_futures_output(cfg, result_list):
    output_file = os.path.join(cfg.output_csv_path, f'{cfg.dataset_name}.csv')
    with open(output_file, 'w') as data_file:
        csv_writer = csv.writer(data_file)
        for i, result in enumerate(result_list):
            data = dataclasses.asdict(result)
            if i == 0:
                # writes header
                csv_writer.writerow(data.keys())
            csv_writer.writerow(data.values())
    return output_file



@dataclass
class VectorReaderConfig:
    pass

@dataclass
class FileGeoDFConfig:
    _target_: str = "pytorch_segmentation_models_trainer.tools.data_readers.vector_reader.FileGeoDF"
    file_name: str = "../tests/testing_data/data/vectors/test_polygons.geojson"

@dataclass
class PostgisConfig(VectorReaderConfig):
    _target_: str = "pytorch_segmentation_models_trainer.tools.data_readers.vector_reader.PostgisGeoDF"
    database: str = "dataset_mestrado"
    sql: str = "select id, geom from buildings"
    user: str = "postgres"
    password: str = "postgres"
    host: str = "localhost"
    port: int = 5432

@dataclass
class BatchFileGeoDFConfig(VectorReaderConfig):
    _target_: str = "pytorch_segmentation_models_trainer.tools.data_readers.vector_reader.BatchFileGeoDF"
    root_dir: str = "../tests/testing_data/data/vectors"
    file_extension: str = "geojson"

@dataclass
class COCOGeoDFConfig(VectorReaderConfig):
    _target_: str = "pytorch_segmentation_models_trainer.tools.data_readers.vector_reader.COCOGeoDF"
    file_name: str = MISSING

@dataclass
class TemplateMaskBuilder(ABC):
    defaults: List[Any] = field(default_factory=lambda: [{"geo_df": "batch_file"}])
    geo_df: VectorReaderConfig = MISSING
    root_dir: str = '/data'
    output_csv_path: str = '/data'
    dataset_name: str = 'dsg_dataset'
    dataset_has_relative_path: bool = True
    image_root_dir: str = 'images'
    image_extension: str = 'tif'
    image_dir_is_relative_to_root_dir: str = True
    replicate_image_folder_structure: bool = True
    relative_path_on_csv: bool = True
    build_polygon_mask: bool = True
    polygon_mask_folder_name: str = 'polygon_masks'
    build_boundary_mask: bool = True
    boundary_mask_folder_name: str = 'boundary_masks'
    build_vertex_mask: bool = True
    vertex_mask_folder_name: str = 'vertex_masks'
    build_crossfield_mask: bool = True
    crossfield_mask_folder_name: str = 'crossfield_masks'
    build_distance_mask: bool = True
    distance_mask_folder_name: str = 'distance_masks'
    build_size_mask: bool = True
    size_mask_folder_name: str = 'size_masks'
    min_polygon_area: float = 50.0
    mask_output_extension: str = 'png'

    def __post_init__(self):
        self.geo_df = instantiate(self.geo_df)
        self.output_dir_dict = self.build_dir_dict()
        self.mask_type_list = self.build_mask_type_list()

    @abstractmethod
    def process(self, input_raster_path: str, output_dir: str, \
        filter_area: float = None) -> DatasetEntry:
        pass

    def build_mask_func(self):
        return lambda x: self.process(
            input_raster_path=x[0],
            output_dir=x[1]
        )

    def build_dataset_entry(self, input_raster_path: str, raster_df: RasterFile,\
        mask_type_list:list, built_mask_dict: dict) -> DatasetEntry:
        lambda_func = lambda x: make_path_relative(x[0], x[1]) \
            if self.dataset_has_relative_path else x[0]
        args_dict = dict()
        for mask_key, file_path in built_mask_dict.items():
            arg_name = mask_key.split('_')[0] + '_mask'
            args_dict[arg_name] = lambda_func(
                [file_path, getattr(self, f'{arg_name}_folder_name')]
            )
        args_dict.update(
            raster_df.get_image_stats()
        )
        return DatasetEntry(
                image=make_path_relative(
                        input_raster_path,
                        os.path.basename(
                            os.path.normpath(self.image_root_dir)
                        )
                    ) if self.dataset_has_relative_path else input_raster_path,
                **args_dict
            )
    
    def build_mask_type_list(self):
        """[summary]
        Args:
            cfg (MaskBuilder): [description]
        Returns:
            [type]: [description]
        """
        mask_type_list = []
        if self.build_polygon_mask:
            mask_type_list.append(GeomTypeEnum.POLYGON)
        if self.build_boundary_mask:
            mask_type_list.append(GeomTypeEnum.LINE)
        if self.build_vertex_mask:
            mask_type_list.append(GeomTypeEnum.POINT)
        return mask_type_list
        
    def build_dir_dict(self):
        input_base_path = str(
            os.path.join(self.root_dir, self.image_root_dir)
        ) if self.image_dir_is_relative_to_root_dir else self.image_root_dir
        dir_dict = dict()
        for mask_type in ['polygon_mask', 'boundary_mask', 'vertex_mask', 'crossfield_mask', 'distance_mask', 'size_mask']:
            if getattr(self, f"build_{mask_type}"):
                mask_folder_name = getattr(self, f"{mask_type}_folder_name")
                output_base_path = str(
                    os.path.join(self.root_dir, mask_folder_name)
                )
                dir_dict[mask_folder_name] = output_base_path
                if self.replicate_image_folder_structure:
                    build_destination_dirs(
                        input_base_path=input_base_path,
                        output_base_path=output_base_path
                    )
        return dir_dict
    
    def build_generator(self):
        image_base_path = os.path.join(
            self.root_dir, self.image_root_dir
        ) if self.image_dir_is_relative_to_root_dir else self.image_root_dir
        return (
            (
                input_raster_path,
                self.root_dir,
                build_output_raster_list(input_raster_path, self)
            ) for input_raster_path in Path(image_base_path).glob(f'**/*.{self.image_extension}')
        )

    def get_number_of_tasks(self):
        image_base_path = os.path.join(
            self.root_dir, self.image_root_dir
        ) if self.image_dir_is_relative_to_root_dir else self.image_root_dir
        return len([
            i for i in Path(image_base_path).glob(f'**/*.{self.image_extension}')
        ])
    
    def build_mask_and_ds_entry(self, input_raster_path, input_vector_layer, output_dir, filter_area):
        raster_df = RasterFile(file_name=input_raster_path)
        built_mask_dict = raster_df.build_mask(
            input_vector_layer=input_vector_layer,
            output_dir=output_dir,
            output_dir_dict=self.output_dir_dict,
            mask_types=self.mask_type_list,
            filter_area=filter_area,
            output_extension=self.mask_output_extension,
            compute_crossfield=self.build_crossfield_mask,
            compute_distances=self.build_distance_mask,
            compute_sizes=self.build_size_mask
        )
        ds_entry = self.build_dataset_entry(
            input_raster_path=input_raster_path,
            raster_df=raster_df,
            mask_type_list=self.mask_type_list,
            built_mask_dict=built_mask_dict
        )
        del raster_df
        return ds_entry


@dataclass
class MaskBuilder(TemplateMaskBuilder):

    def process(self, input_raster_path: str, output_dir: str, \
        filter_area: float = None) -> DatasetEntry:
        """[summary]

        Args:
            input_raster_path (str): [description]
            input_vector (GeoDF): [description]
            mask_type_list (List[GeomType]): [description]
            mask_output_type (MaskOutputType): [description]
        """
        return self.build_mask_and_ds_entry(
            input_raster_path=input_raster_path,
            input_vector_layer=self.geo_df,
            output_dir=output_dir,
            filter_area=filter_area
        )

@dataclass
class COCOMaskBuilder(TemplateMaskBuilder):
    
    def process(self, input_raster_path: str, output_dir: str, \
        filter_area: float = None) -> DatasetEntry:
        key = os.path.basename(input_raster_path).split('.')[0]
        return self.build_mask_and_ds_entry(
            input_raster_path=input_raster_path,
            input_vector_layer=self.geo_df.get_geodf_item(key),
            output_dir=output_dir,
            filter_area=filter_area
        )
