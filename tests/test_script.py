# -*- coding: utf-8 -*-
"""
/***************************************************************************
 segmentation_models_trainer
                              -------------------
        begin                : 2021-03-04
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
import subprocess

import pytorch_lightning as pl
from hydra.experimental import compose, initialize
from pytorch_segmentation_models_trainer.train import train

from tests.utils import CustomTestCase


class Test_TestScript(CustomTestCase):
    def test_run_validate_config(self) -> None:
        return_from_process = subprocess.run(
            [
                'pytorch-segmentation-models-trainer',
                '--config-path',
                '../tests/test_configs',
                '--config-name',
                'experiment',
                '+mode=validate-config'
            ],
            check=True
        )
        self.assertEqual(return_from_process.returncode, 0)
