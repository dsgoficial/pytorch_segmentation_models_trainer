# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-04-16
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
import shutil
import unittest

from collections.abc import Iterable
from pathlib import Path
import warnings
from pytorch_segmentation_models_trainer.tools.parallel_processing.process_executor import (
    Executor,
)
from pytorch_segmentation_models_trainer.utils.os_utils import (
    create_folder,
    hash_file,
    remove_folder,
)

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")


def copy_file(filepath, destination_folder):
    filepath = next(filepath, None) if isinstance(filepath, Iterable) else filepath
    return shutil.copyfile(str(filepath), Path(destination_folder) / filepath.name)


class Test_ProcessExecutor(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", category=ImportWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        self.output_dir = create_folder(os.path.join(root_dir, "test_output"))
        create_folder(self.output_dir)

    def tearDown(self):
        remove_folder(self.output_dir)

    def test_execute_process(self) -> None:
        directory_in_str = os.path.join(root_dir, "data", "images")
        input_files_dict = {
            path.stem: path for path in Path(directory_in_str).glob("**/*.png")
        }
        lambda_func = lambda x: copy_file(x, self.output_dir)
        executor = Executor(lambda_func)
        iterator = iter(list(input_files_dict.values()))
        executor.compute_func(iterator)
        output_files_dict = {
            path.stem: path for path in Path(self.output_dir).glob("**/*.png")
        }
        for filename, file_path in input_files_dict.items():
            if filename not in output_files_dict:
                return False
            self.assertEqual(
                hash_file(file_path), hash_file(output_files_dict[filename])
            )
