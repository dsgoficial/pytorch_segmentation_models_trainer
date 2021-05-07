# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pytorch_segmentation_models_trainer
                              -------------------
        begin                : 2021-04-16
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
import concurrent.futures
import os
from typing import Iterator
from tqdm import tqdm

class Executor:
    def __init__(self, compute_func, simultaneous_tasks=None) -> None:
        self.compute_func = compute_func
        self.simultaneous_tasks = os.cpu_count() if simultaneous_tasks is None \
            else simultaneous_tasks

    def execute_tasks(self, tasks):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.simultaneous_tasks) as executor:
            # Schedule the first N futures.  We don't want to schedule them all
            # at once, to avoid consuming excessive amounts of memory.

            futures = [
                executor.submit(self.compute_func, task) \
                    for task in tasks
            ]
            kwargs = {
                'total': len(futures),
                'unit': 'it',
                'unit_scale': True,
                'leave': True
            }
            for _ in tqdm(concurrent.futures.as_completed(futures), **kwargs):
                pass
        return futures
