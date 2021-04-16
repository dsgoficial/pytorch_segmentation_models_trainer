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
from typing import Iterator

class Executor:
    def __init__(self, compute_func, simultaneous_tasks) -> None:
        self.compute_func = compute_func
        self.simultaneous_tasks = simultaneous_tasks

    def execute_tasks(self, tasks: Iterator):
       with concurrent.futures.ThreadPoolExecutor() as executor:
            # Schedule the first N futures.  We don't want to schedule them all
            # at once, to avoid consuming excessive amounts of memory.
            futures = {
                executor.submit(self.compute_func, task)
                for task in itertools.islice(tasks, self.simultaneous_tasks)
            }

            while futures:
                # Wait for the next future to complete.
                done, futures = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )

                # Schedule the next set of futures.  We don't want more than N futures
                # in the pool at a time, to keep memory consumption down.
                for task in itertools.islice(tasks, len(done)):
                    futures.add(
                        executor.submit(self.compute_func, task)
                    )