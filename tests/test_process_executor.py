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

from collections.abc import Iterable
from pathlib import Path
from pytorch_segmentation_models_trainer.tools.parallel_processing.process_executor import (
    Executor,
    ProcessPoolExecutor,
)
from pytorch_segmentation_models_trainer.utils.os_utils import (
    hash_file,
)
from tests.utils import BasicTestCase

current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "testing_data")


def copy_file(filepath, destination_folder):
    filepath = next(filepath, None) if isinstance(filepath, Iterable) else filepath
    return shutil.copyfile(str(filepath), Path(destination_folder) / filepath.name)


class Test_ProcessExecutor(BasicTestCase):
    def setUp(self):
        super().setUp()
        self.output_dir = self.make_temp_dir()

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

    def test_process_pool_executor_sets_executor_class(self) -> None:
        executor = ProcessPoolExecutor(str, simultaneous_tasks=1)

        self.assertEqual(executor.simultaneous_tasks, 1)
        self.assertEqual(executor.executor_class.__name__, "ProcessPoolExecutor")

    def test_execute_tasks_processes_all_items(self) -> None:
        executor = Executor(lambda x: x * 2, simultaneous_tasks=2)

        output = executor.execute_tasks(iter([1, 2, 3, 4]), n_tasks=4)

        self.assertEqual(sorted(output), [2, 4, 6, 8])
