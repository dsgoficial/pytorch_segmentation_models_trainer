# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-03-25
        git sha              : $Format:%H$
        copyright            : (C) 2021 by Philipe Borba -
                                    Cartographic Engineer @ Brazilian Army
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

import os
import unittest
import warnings

import geopandas
import hydra
import rasterio
import torch
from geopandas.testing import geom_almost_equals
from hydra import compose, initialize
from pytorch_segmentation_models_trainer.tools.data_handlers.data_writer import (
    VectorFileDataWriter,
)
from pytorch_segmentation_models_trainer.tools.polygonization.polygonizer import (
    ACMConfig,
    ACMPolygonizerProcessor,
    ASMConfig,
    ASMPolygonizerProcessor,
    SimplePolConfig,
    SimplePolygonizerProcessor,
)
from pytorch_segmentation_models_trainer.utils import frame_field_utils
from pytorch_segmentation_models_trainer.utils.os_utils import (
    create_folder,
    remove_folder,
)

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")

frame_field_root_dir = os.path.join(
    current_dir, "testing_data", "data", "frame_field_data"
)

device = "cpu"


class Test_Polygonize(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self.output_dir = create_folder(os.path.join(root_dir, "test_output"))
        self.frame_field_ds = self.get_frame_field_ds()
        with rasterio.open(self.frame_field_ds[0]["path"], "r") as raster_ds:
            self.crs = raster_ds.crs
            self.profile = raster_ds.profile
            self.transform = raster_ds.transform

    def tearDown(self):
        remove_folder(self.output_dir)

    def get_frame_field_ds(self):
        csv_path = os.path.join(frame_field_root_dir, "dsg_dataset.csv")
        with initialize(config_path="./test_configs"):
            cfg = compose(
                config_name="frame_field_dataset.yaml",
                overrides=[
                    "input_csv_path=" + csv_path,
                    "root_dir=" + frame_field_root_dir,
                ],
            )
            frame_field_ds = hydra.utils.instantiate(cfg, _recursive_=False)
        return frame_field_ds

    def test_polygonizer_simple_processor(self) -> None:
        config = SimplePolConfig()
        output_file_path = os.path.join(self.output_dir, "simple_polygonizer.geojson")
        data_writer = VectorFileDataWriter(
            output_file_folder=self.output_dir,
            output_file_name="simple_polygonizer.geojson",
        )
        processor = SimplePolygonizerProcessor(data_writer=data_writer, config=config)
        processor.process(
            {
                "seg": torch.movedim(
                    self.frame_field_ds[0]["gt_polygons_image"], -1, 0
                ).unsqueeze(0)
            },
            self.profile,
        )
        assert os.path.isfile(output_file_path)
        expected_output_gdf = geopandas.read_file(
            filename=os.path.join(
                root_dir, "expected_outputs", "polygonize", "simple_polygonizer.geojson"
            )
        )
        output_features_gdf = geopandas.read_file(filename=output_file_path)
        assert geom_almost_equals(
            expected_output_gdf["geometry"], output_features_gdf["geometry"]
        )

    def test_polygonizer_acm_processor(self) -> None:
        config = ACMConfig()
        output_file_path = os.path.join(self.output_dir, "acm_polygonizer.geojson")
        data_writer = VectorFileDataWriter(
            output_file_folder=self.output_dir,
            output_file_name="acm_polygonizer.geojson",
        )
        processor = ACMPolygonizerProcessor(data_writer=data_writer, config=config)
        processor.process(
            {
                "seg": torch.movedim(
                    self.frame_field_ds[0]["gt_polygons_image"], -1, 0
                ).unsqueeze(0),
                "crossfield": frame_field_utils.compute_crossfield_to_plot(
                    self.frame_field_ds[0]["gt_crossfield_angle"]
                ),
            },
            self.profile,
        )
        assert os.path.isfile(output_file_path)
        expected_output_gdf = geopandas.read_file(
            filename=os.path.join(
                root_dir, "expected_outputs", "polygonize", "acm_polygonizer.geojson"
            )
        )
        output_features_gdf = geopandas.read_file(filename=output_file_path)
        assert geom_almost_equals(
            expected_output_gdf["geometry"], output_features_gdf["geometry"]
        )

    def test_polygonizer_asm_processor(self) -> None:
        config = ASMConfig()
        output_file_path = os.path.join(self.output_dir, "asm_polygonizer.geojson")
        data_writer = VectorFileDataWriter(
            output_file_folder=self.output_dir,
            output_file_name="asm_polygonizer.geojson",
        )
        processor = ASMPolygonizerProcessor(data_writer=data_writer, config=config)
        processor.process(
            {
                "seg": torch.movedim(
                    self.frame_field_ds[0]["gt_polygons_image"], -1, 0
                ).unsqueeze(0),
                "crossfield": frame_field_utils.compute_crossfield_to_plot(
                    self.frame_field_ds[0]["gt_crossfield_angle"]
                ),
            },
            self.profile,
        )
        assert os.path.isfile(output_file_path)
        expected_output_gdf = geopandas.read_file(
            filename=os.path.join(
                root_dir, "expected_outputs", "polygonize", "asm_polygonizer.geojson"
            )
        )
        output_features_gdf = geopandas.read_file(filename=output_file_path)
        assert geom_almost_equals(
            expected_output_gdf["geometry"], output_features_gdf["geometry"]
        )
