# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-05-06
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

import hydra
import numpy as np
import pandas as pd
from hydra import compose, initialize
from parameterized import parameterized
from sqlalchemy import create_engine
import psycopg2
from pytorch_segmentation_models_trainer.build_mask import build_masks
from pytorch_segmentation_models_trainer.tools.data_handlers.raster_reader import (
    MaskOutputTypeEnum,
    RasterFile,
)
from pytorch_segmentation_models_trainer.tools.data_handlers.vector_reader import (
    FileGeoDF,
    GeomTypeEnum,
)
from pytorch_segmentation_models_trainer.tools.mask_building.mask_builder import (
    MaskBuilder,
    build_destination_dirs,
    merge_csv_datasets,
)
from pytorch_segmentation_models_trainer.utils.os_utils import (
    create_folder,
    hash_file,
    remove_folder,
)

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")

suffix_dict = {"PNG": ".png", "GTiff": ".tif", "JPEG": ".jpg"}


class Test_BuildMask(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self.output_dir = create_folder(os.path.join(root_dir, "test_output"))
        self.replicated_dir = create_folder(
            os.path.join(root_dir, "..", "replicated_dirs")
        )
        self.build_test_dataset()

    def build_test_dataset(self):
        mask_geo_df = FileGeoDF(
            os.path.join(root_dir, "data", "build_masks_data", "buildings.geojson")
        )
        engine = create_engine("postgresql://postgres:postgres@localhost:5432/test_db")
        mask_geo_df.gdf.to_postgis("buildings", engine, if_exists="replace")

    def tearDown(self):
        remove_folder(self.output_dir)
        remove_folder(self.replicated_dir)

    def test_build_output_dirs_raises_exception(self):
        output_base_path = os.path.join(self.output_dir, "replicated_dirs")
        with self.assertRaises(Exception) as context:
            build_destination_dirs(
                input_base_path=root_dir, output_base_path=output_base_path
            )
        self.assertTrue(
            "input path must not be in output_path" in str(context.exception)
        )

    def test_build_output_dirs(self):
        output_list = build_destination_dirs(
            input_base_path=root_dir, output_base_path=self.replicated_dir
        )
        assert len(output_list) > 0
        input_structure = [
            dirpath.replace(root_dir, "") for dirpath, _, __ in os.walk(root_dir)
        ]
        output_structure = [
            dirpath.replace(self.replicated_dir, "")
            for dirpath, _, __ in os.walk(self.replicated_dir)
        ]
        assert len(input_structure) == len(output_structure)
        for item in output_structure:
            assert item in input_structure

    def test_build_masks(self):
        with initialize(config_path="./test_configs"):
            image_dir = os.path.join(root_dir, "data", "build_masks_data", "images")
            expected_output_path = os.path.join(
                root_dir, "expected_outputs", "build_masks"
            )
            cfg = compose(
                config_name="build_mask.yaml",
                overrides=[
                    "mask_builder.root_dir=" + self.output_dir,
                    "mask_builder.output_csv_path=" + self.output_dir,
                    "mask_builder.image_root_dir=" + image_dir,
                    "mask_builder.geo_df.file_name="
                    + os.path.join(
                        root_dir, "data", "build_masks_data", "buildings.geojson"
                    ),
                ],
            )
            csv_output = build_masks(cfg)
            expected_df = pd.read_csv(
                os.path.join(expected_output_path, "dsg_dataset.csv")
            ).sort_values("image")
            output_df = pd.read_csv(csv_output).sort_values("image")
            pd.testing.assert_frame_equal(
                expected_df.reset_index(drop=True), output_df.reset_index(drop=True)
            )

    def test_build_masks_from_postgis(self):
        with initialize(config_path="./test_configs"):
            image_dir = os.path.join(root_dir, "data", "build_masks_data", "images")
            expected_output_path = os.path.join(
                root_dir, "expected_outputs", "build_masks"
            )
            cfg = compose(
                config_name="build_mask_postgis.yaml",
                overrides=[
                    "mask_builder.root_dir=" + self.output_dir,
                    "mask_builder.output_csv_path=" + self.output_dir,
                    "mask_builder.image_root_dir=" + image_dir,
                ],
            )
            csv_output = build_masks(cfg)
            expected_df = pd.read_csv(
                os.path.join(expected_output_path, "dsg_dataset.csv")
            ).sort_values("image")
            output_df = pd.read_csv(csv_output).sort_values("image")
            pd.testing.assert_frame_equal(
                expected_df.reset_index(drop=True), output_df.reset_index(drop=True)
            )

    def test_build_masks_coco(self):
        with initialize(config_path="./test_configs"):
            image_dir = os.path.join(
                root_dir, "data", "build_masks_data", "coco_images"
            )
            expected_output_path = os.path.join(
                root_dir, "expected_outputs", "build_masks"
            )
            cfg = compose(
                config_name="build_coco_mask.yaml",
                overrides=[
                    "mask_builder.root_dir=" + self.output_dir,
                    "mask_builder.output_csv_path=" + self.output_dir,
                    "mask_builder.image_root_dir=" + image_dir,
                    "mask_builder.geo_df.file_name="
                    + os.path.join(
                        root_dir, "data", "build_masks_data", "annotation.json"
                    ),
                ],
            )
            csv_output = build_masks(cfg)
            expected_df = pd.read_csv(
                os.path.join(expected_output_path, "coco_dataset.csv")
            ).sort_values("image")
            output_df = pd.read_csv(csv_output).sort_values("image")
            pd.testing.assert_frame_equal(
                expected_df.reset_index(drop=True), output_df.reset_index(drop=True)
            )

    def test_build_masks_coco_no_prebuild(self):
        with initialize(config_path="./test_configs"):
            image_dir = os.path.join(
                root_dir, "data", "build_masks_data", "coco_images"
            )
            expected_output_path = os.path.join(
                root_dir, "expected_outputs", "build_masks"
            )
            cfg = compose(
                config_name="build_coco_mask.yaml",
                overrides=[
                    "mask_builder.root_dir=" + self.output_dir,
                    "mask_builder.output_csv_path=" + self.output_dir,
                    "mask_builder.image_root_dir=" + image_dir,
                    "mask_builder.geo_df.file_name="
                    + os.path.join(
                        root_dir, "data", "build_masks_data", "annotation.json"
                    ),
                    "+mask_builder.geo_df.pre_build_vector_dict=False",
                ],
            )
            csv_output = build_masks(cfg)
            expected_df = pd.read_csv(
                os.path.join(expected_output_path, "coco_dataset.csv")
            ).sort_values("image")
            output_df = pd.read_csv(csv_output).sort_values("image")
            pd.testing.assert_frame_equal(
                expected_df.reset_index(drop=True), output_df.reset_index(drop=True)
            )

    def test_merge_csv_datasets(self):
        ds_csv1 = os.path.join(
            root_dir, "data", "build_masks_data", "dsg_dataset_only_polygon_masks.csv"
        )
        ds_csv2 = os.path.join(
            root_dir, "data", "build_masks_data", "dsg_dataset_other_masks.csv"
        )
        output_csv = os.path.join(self.output_dir, "dsg_dataset_merged.csv")
        merge_csv_datasets(ds_csv1, ds_csv2, "image", output_csv)
        expected_df = pd.read_csv(
            os.path.join(
                root_dir, "expected_outputs", "build_masks", "dsg_dataset_merged.csv"
            )
        ).sort_values("image")
        output_df = pd.read_csv(output_csv).sort_values("image")
        pd.testing.assert_frame_equal(
            expected_df.reset_index(drop=True), output_df.reset_index(drop=True)
        )

    def test_build_masks_merge_existing(self):
        with initialize(config_path="./test_configs"):
            image_dir = os.path.join(root_dir, "data", "build_masks_data", "images")
            expected_output_path = os.path.join(
                root_dir, "expected_outputs", "build_masks"
            )
            cfg = compose(
                config_name="build_mask.yaml",
                overrides=[
                    "mask_builder.root_dir=" + self.output_dir,
                    "mask_builder.output_csv_path=" + self.output_dir,
                    "mask_builder.image_root_dir=" + image_dir,
                    "mask_builder.build_crossfield_mask=False",
                    "mask_builder.geo_df.file_name="
                    + os.path.join(
                        root_dir, "data", "build_masks_data", "buildings.geojson"
                    ),
                ],
            )
            csv_output = build_masks(cfg)
        with initialize(config_path="./test_configs"):
            image_dir = os.path.join(root_dir, "data", "build_masks_data", "images")
            expected_output_path = os.path.join(
                root_dir, "expected_outputs", "build_masks"
            )
            cfg = compose(
                config_name="build_mask.yaml",
                overrides=[
                    "+mask_builder.merge_existing=True",
                    "mask_builder.root_dir=" + self.output_dir,
                    "mask_builder.output_csv_path=" + self.output_dir,
                    "mask_builder.image_root_dir=" + image_dir,
                    "mask_builder.build_boundary_mask=False",
                    "mask_builder.geo_df.file_name="
                    + os.path.join(
                        root_dir, "data", "build_masks_data", "buildings.geojson"
                    ),
                ],
            )
            csv_output = build_masks(cfg)
        expected_df = pd.read_csv(
            os.path.join(expected_output_path, "dsg_dataset.csv")
        ).sort_values("image")
        output_df = pd.read_csv(csv_output).sort_values("image")
        pd.testing.assert_frame_equal(
            expected_df.reset_index(drop=True), output_df.reset_index(drop=True)
        )

    def test_build_masks_with_bounding_boxes(self):
        with initialize(config_path="./test_configs"):
            image_dir = os.path.join(root_dir, "data", "build_masks_data", "images")
            expected_output_path = os.path.join(
                root_dir, "expected_outputs", "build_masks"
            )
            cfg = compose(
                config_name="build_mask.yaml",
                overrides=[
                    "mask_builder.root_dir=" + self.output_dir,
                    "mask_builder.output_csv_path=" + self.output_dir,
                    "mask_builder.image_root_dir=" + image_dir,
                    "mask_builder.build_crossfield_mask=False",
                    "mask_builder.geo_df.file_name="
                    + os.path.join(
                        root_dir, "data", "build_masks_data", "buildings.geojson"
                    ),
                    "+mask_builder.build_bounding_box_list=True",
                ],
            )
            csv_output = build_masks(cfg)
        expected_df = pd.read_csv(
            os.path.join(expected_output_path, "dsg_dataset_with_bboxes.csv")
        ).sort_values("image")
        output_df = pd.read_csv(csv_output).sort_values("image")
        pd.testing.assert_frame_equal(
            expected_df.reset_index(drop=True), output_df.reset_index(drop=True)
        )

    def test_build_masks_with_polygons(self):
        with initialize(config_path="./test_configs"):
            image_dir = os.path.join(root_dir, "data", "build_masks_data", "images")
            expected_output_path = os.path.join(
                root_dir, "expected_outputs", "build_masks"
            )
            cfg = compose(
                config_name="build_mask.yaml",
                overrides=[
                    "mask_builder.root_dir=" + self.output_dir,
                    "mask_builder.output_csv_path=" + self.output_dir,
                    "mask_builder.image_root_dir=" + image_dir,
                    "mask_builder.build_crossfield_mask=False",
                    "mask_builder.geo_df.file_name="
                    + os.path.join(
                        root_dir, "data", "build_masks_data", "buildings.geojson"
                    ),
                    "+mask_builder.build_polygon_list=True",
                ],
            )
            csv_output = build_masks(cfg)
        expected_df = pd.read_csv(
            os.path.join(expected_output_path, "dsg_dataset_with_polygons.csv")
        ).sort_values("image")
        output_df = pd.read_csv(csv_output).sort_values("image")
        pd.testing.assert_frame_equal(
            expected_df.reset_index(drop=True), output_df.reset_index(drop=True)
        )
