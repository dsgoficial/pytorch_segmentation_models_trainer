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
import itertools
import os
from typing import Iterator
from tqdm import tqdm

class Executor:
    def __init__(self, compute_func, simultaneous_tasks=None) -> None:
        self.compute_func = compute_func
        self.simultaneous_tasks = os.cpu_count() if simultaneous_tasks is None \
            else simultaneous_tasks

    def execute_tasks(self, tasks, n_tasks):
        result_list = []
        with tqdm(total=n_tasks) as pbar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Schedule the first N futures.  We don't want to schedule them all
                # at once, to avoid consuming excessive amounts of memory.
                futures = {
                    executor.submit(self.compute_func, task): task
                    for task in itertools.islice(tasks, self.simultaneous_tasks)
                }
                while futures:
                    done, _ = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    new_tasks_to_schedule = 0
                    for fut in done:
                        original_task = futures.pop(fut)
                        result_list.append(fut.result())
                        del original_task, fut
                        new_tasks_to_schedule += 1
                        pbar.update(1)
                    for task in itertools.islice(tasks, new_tasks_to_schedule):
                        fut = executor.submit(self.compute_func, task)
                        futures[fut] = task
        return result_list
