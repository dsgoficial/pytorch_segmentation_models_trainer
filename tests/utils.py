# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-03-01
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
import gc
import os
import shutil
import tempfile
from typing import Any, List
import unittest
import warnings
import geopandas
import hydra
import torch


def get_file_list(dir_path, extension):
    output_list = []
    for root, dirs, files in os.walk(dir_path):
        output_list += [os.path.join(root, f) for f in files if f.endswith(extension)]
    return sorted(output_list)


def create_csv_file(file_path, image_list, label_list, root_to_be_removed=None):
    csv_text = "id,image,mask,rows,columns\n"
    for idx, i in enumerate(image_list):
        csv_text += f"{idx},{i},{label_list[idx]},512,512\n"
    with open(file_path, "w") as csv_file:
        if root_to_be_removed is not None:
            csv_text = csv_text.replace(root_to_be_removed, "")
        csv_file.write(csv_text)
    return file_path


def get_config_from_hydra(config_name, overrides_list, config_path=None):
    config_path = config_path if config_path is not None else "./test_configs"
    with hydra.initialize(config_path=config_path):
        cfg = hydra.compose(config_name=config_name, overrides=overrides_list)
    return cfg


def make_hydra_config(fields: dict):
    """
    Build a synthetic OmegaConf DictConfig without reading any YAML file.

    Useful for unit tests that need a config object without disk I/O.

    Args:
        fields: dict of config keys/values (can be nested)

    Returns:
        OmegaConf DictConfig
    """
    from omegaconf import OmegaConf

    return OmegaConf.create(fields)


def load_geometry_list_from_geojson(file_path: str) -> List[Any]:
    """
    Loads data to be used in tests.
    """
    # Load data
    gdf = geopandas.read_file(file_path)
    return [i for i in gdf.geometry]


class BasicTestCase(unittest.TestCase):
    """
    Basic test case for the tests.
    """

    def setUp(self):
        """
        Setup the test case.
        """
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self._temp_dirs = []

    def make_temp_dir(self) -> str:
        """Create a temporary directory that is automatically cleaned up in tearDown."""
        tmp = tempfile.mkdtemp()
        self._temp_dirs.append(tmp)
        return tmp

    def tearDown(self) -> None:
        for tmp_dir in self._temp_dirs:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
        outputs_path = os.path.join(os.path.dirname(__file__), "..", "outputs")
        if os.path.exists(outputs_path):
            shutil.rmtree(outputs_path)
        lightning_logs_path = os.path.join(
            os.path.dirname(__file__), "..", "lightning_logs"
        )
        if os.path.exists(lightning_logs_path):
            shutil.rmtree(lightning_logs_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class CustomTestCase(BasicTestCase):
    def setUp(self):
        super(CustomTestCase, self).setUp()
        current_dir = os.path.dirname(__file__)
        self.root_dir = os.path.join(current_dir, "testing_data", "data")
        image_list = get_file_list(os.path.join(self.root_dir, "images"), ".png")
        label_list = get_file_list(os.path.join(self.root_dir, "labels"), ".png")
        label_list = get_file_list(
            os.path.join(current_dir, "testing_data", "data", "labels"), ".png"
        )
        self.csv_ds_file = create_csv_file(
            os.path.join(current_dir, "testing_data", "csv_train_ds.csv"),
            image_list[0:5],
            label_list[0:5],
        )
        self.csv_ds_file_without_root = create_csv_file(
            os.path.join(current_dir, "testing_data", "csv_train_ds_without_root.csv"),
            image_list[0:5],
            label_list[0:5],
            root_to_be_removed=self.root_dir,
        )

    def tearDown(self):
        os.remove(self.csv_ds_file)
        os.remove(self.csv_ds_file_without_root)
        super(CustomTestCase, self).tearDown()
